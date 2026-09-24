"""【服务器】Stage 0 可学性验证(PLAN v1.1 §3 第二步,决策关口)。

问题:"说话对象"与"语调"两类信号,能不能从音频里学出来?
方法:冻结特征 + 逻辑回归(故意用最简单的分类器:测"信号在不在",不测"模型够不够强")。

特征(三组,用于消融):
  prosody :手工韵律特征(句尾 F0 走势、句尾/整句 F0 比、F0 统计、能量统计、混响粗估)
  wavlm   :冻结 WavLM-base-plus 第 L 层的均值+标准差池化(保留说话人/房间/语气等副语言信息)
  both    :两者拼接
评测条件:
  val_spk  :训练时未见过的说话人(同为 CosyVoice3)
  xtts     :跨 TTS——v2 探针(edge-tts)。语调=echo_question vs echo_confirm;
             说话对象=side_speech vs 对 agent 说的类别(v2 为干声,只含内容线索)
  xtts_room:同上,但说话对象评测样本用 pyroomacoustics 镜像法重新做"转头"处理
             (与训练用的指数噪声 RIR 生成方式不同),检验声学线索跨条件泛化
输出:各任务 × 特征组 × 条件的 准确率 / AUC,写 JSON 与 markdown。

环境:/home/pxieaf/home2/envs/qwen3omni(需 scikit-learn、pyroomacoustics)
用法:python probe_learnability.py --data /home/pxieaf/home2/data/stage0 \
        --probes /home/pxieaf/home2/cs_duplex/data/probe_audio_v2 --out /home/pxieaf/home2/data/stage0/results
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "baseline"))
from render_probes import f0_track  # noqa: E402

SR = 16000
TO_AGENT_CLASSES = {"backchannel", "correction_I0", "correction_I1", "floor_claim"}


# ---------------- 特征 ----------------
def prosody_feats(w: np.ndarray) -> np.ndarray:
    f = f0_track(w)
    if len(f) < 6:
        f = np.full(6, 150.0)
    lf = np.log(f)
    n = len(lf)
    tail = lf[int(n * 0.7):]
    x = np.arange(len(tail))
    slope_tail = np.polyfit(x, tail, 1)[0] if len(tail) >= 2 else 0.0
    slope_all = np.polyfit(np.arange(n), lf, 1)[0]
    frames = w[: len(w) // 160 * 160].reshape(-1, 160)
    e = np.log(np.sqrt((frames ** 2).mean(1)) + 1e-6)
    active = e > e.max() - 4
    # 混响粗估:能量包络在语音段后的衰减速度(越慢越混响)
    env = e[active] if active.any() else e
    decay = np.polyfit(np.arange(len(env)), env, 1)[0] if len(env) >= 2 else 0.0
    return np.array([lf.mean(), lf.std(), lf.max() - lf.min(), slope_all, slope_tail,
                     tail.mean() - lf.mean(), tail[-3:].mean() - lf.mean(),
                     e[active].mean(), e[active].std(), e.max(), active.mean(), decay,
                     len(w) / SR], np.float32)


class WavLM:
    def __init__(self, layer: int, device: str):
        import torch
        from transformers import AutoFeatureExtractor, WavLMModel
        self.torch = torch
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.m = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
        self.layer, self.device = layer, device

    def __call__(self, w: np.ndarray) -> np.ndarray:
        inp = self.fe(w, sampling_rate=SR, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            hs = self.m(**inp, output_hidden_states=True).hidden_states[self.layer][0]
        return self.torch.cat([hs.mean(0), hs.std(0)]).float().cpu().numpy()


# ---------------- 数据 ----------------
def load_stage0(data: Path):
    rows = []
    for lab in sorted(data.glob("shard*/labels.jsonl")) + sorted(data.glob("labels.jsonl")):
        base = lab.parent
        for line in lab.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            sub = "addressee" if r["task"] == "addressee" else "intonation"
            r["path"] = str(base / sub / f"{r['id']}.wav")
            rows.append(r)
    return rows


def room_turn_away(w: np.ndarray, rng: random.Random) -> np.ndarray:
    """pyroomacoustics 镜像法:声源距麦克风 1.5–3m(对照训练的指数噪声 RIR)。"""
    import pyroomacoustics as pra
    dims = [rng.uniform(3, 6), rng.uniform(3, 5), rng.uniform(2.5, 3.2)]
    rt60 = rng.uniform(0.3, 0.8)
    e_abs, max_order = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=SR, materials=pra.Material(e_abs), max_order=min(max_order, 12))
    mic = [dims[0] / 2, dims[1] / 2, 1.2]
    ang = rng.uniform(0, 2 * np.pi)
    d = rng.uniform(1.5, 3.0)
    src = [min(max(mic[0] + d * np.cos(ang), 0.3), dims[0] - 0.3),
           min(max(mic[1] + d * np.sin(ang), 0.3), dims[1] - 0.3), 1.5]
    room.add_source(src, signal=w)
    room.add_microphone(mic)
    room.simulate()
    y = room.mic_array.signals[0][: len(w) + int(0.3 * SR)]
    return (y / (np.abs(y).max() + 1e-9) * np.abs(w).max() * 10 ** (rng.uniform(-10, -3) / 20)).astype(np.float32)


def load_probe_segments(probes: Path, rng: random.Random):
    """从 v2 探针的 user.wav 切出探针段。"""
    intonation, addressee, addressee_room = [], [], []
    for d in sorted(probes.glob("probe_*")):
        s = json.loads((d / "session.json").read_text(encoding="utf-8"))
        cls, p = s["probe_class"], s["probes"][0]
        w, _ = sf.read(d / "user.wav", dtype="float32")
        seg = w[int(p["t_onset"] * SR): int(p["t_end"] * SR)]
        if len(seg) < 1600:
            continue
        if cls in ("echo_question", "echo_confirm"):
            intonation.append((seg, int(cls == "echo_question")))
        elif cls == "side_speech" or cls in TO_AGENT_CLASSES:
            lab = int(cls != "side_speech")
            addressee.append((seg, lab))
            addressee_room.append((room_turn_away(seg, rng) if lab == 0 else seg, lab))
    return {"intonation": intonation, "addressee": addressee, "addressee_room": addressee_room}


# ---------------- 训练与评测 ----------------
def evaluate(Xtr, ytr, evals: dict) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5,
                                                             class_weight="balanced"))
    clf.fit(Xtr, ytr)
    out = {}
    for name, (X, y) in evals.items():
        if len(set(y)) < 2:
            continue
        p = clf.predict_proba(X)[:, 1]
        out[name] = {"bal_acc": round(balanced_accuracy_score(y, p > 0.5), 3),
                     "auc": round(roc_auc_score(y, p), 3), "n": int(len(y))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rng = random.Random(0)

    rows = load_stage0(Path(args.data))
    print(f"stage0 samples: {len(rows)}", flush=True)
    wavlm = WavLM(args.layer, args.device)

    def feats(w):
        return {"prosody": prosody_feats(w), "wavlm": wavlm(w)}

    cache = {}
    for r in rows:
        w, _ = sf.read(r["path"], dtype="float32")
        cache[r["id"]] = feats(w)
    probe_sets = load_probe_segments(Path(args.probes), rng)
    probe_feats = {k: [(feats(w), y) for w, y in v] for k, v in probe_sets.items()}
    print("probe eval sets:", {k: len(v) for k, v in probe_feats.items()}, flush=True)

    def stack(items, group):
        if group == "both":
            return np.stack([np.concatenate([f["prosody"], f["wavlm"]]) for f in items])
        return np.stack([f[group] for f in items])

    results = {}
    for task, probe_keys in (("intonation", {"xtts": "intonation"}),
                             ("addressee", {"xtts": "addressee", "xtts_room": "addressee_room"})):
        tr = [r for r in rows if r["task"] == task and r["split"] == "train"]
        va = [r for r in rows if r["task"] == task and r["split"] == "val"]
        for group in ("prosody", "wavlm", "both"):
            Xtr = stack([cache[r["id"]] for r in tr], group)
            ytr = np.array([r["label"] for r in tr])
            evals = {"val_spk": (stack([cache[r["id"]] for r in va], group),
                                 np.array([r["label"] for r in va]))}
            for cond, key in probe_keys.items():
                evals[cond] = (stack([f for f, _ in probe_feats[key]], group),
                               np.array([y for _, y in probe_feats[key]]))
            results[f"{task}/{group}"] = evaluate(Xtr, ytr, evals)
            print(task, group, results[f"{task}/{group}"], flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage0_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    lines = ["| 任务/特征 | val_spk 平衡准确率 | val_spk AUC | 跨TTS 平衡准确率 | 跨TTS AUC | 跨TTS+真实房间仿真 平衡准确率 | AUC |",
             "|---|---|---|---|---|---|---|"]
    for k, v in results.items():
        def g(c, m):
            return v.get(c, {}).get(m, "—")
        lines.append(f"| {k} | {g('val_spk','bal_acc')} | {g('val_spk','auc')} | {g('xtts','bal_acc')} | "
                     f"{g('xtts','auc')} | {g('xtts_room','bal_acc')} | {g('xtts_room','auc')} |")
    (out / "stage0_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
