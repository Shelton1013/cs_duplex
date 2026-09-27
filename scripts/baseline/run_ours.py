"""【服务器】我们的模型(Stage B)在探针集上运行,产出与各基线同格式的动作文件。

判定协议与 Qwen3-Omni 裁判完全一致(保证可比):每个 VAD 语音段内,从 onset+0.4s 起
每 0.3s 判一次(最长 2.4s 或段结束),输入 = [onset, t] 的用户音频 + agent 截至 t 的已播文本;
分数 > 阈值 → CUT,时刻 = t + 实测推理耗时。
阈值默认 0(STOP 与 CONTINUE 首 token 等概率),不在评测集上调。

用法:
  CUDA_VISIBLE_DEVICES=1 python run_ours.py --ckpt /home/pxieaf/home2/ckpt/stage_b_v0/best.pt \
      --probes data/probe_audio_v2 --out data/baselines_v2/ours_b0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen3omni_judge import FIRST, MAX_WAIT, STEP, played_text  # noqa: E402
from run_vad import energy_vad  # noqa: E402
from slowpath.decision import Decider  # noqa: E402
from slowpath.fo_encoder import fbank, load_fo_encoder  # noqa: E402
from slowpath.model import SlowPath  # noqa: E402
from slowpath.prosody import prosody_25hz  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.0)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--fo_repo", default="/home/pxieaf/Freeze-Omni")
    ap.add_argument("--fo_ckpt", default="/home/share/data_makchen/peng/models/Freeze-Omni/checkpoints/audiollm")
    ap.add_argument("--llm", default="/home/pxieaf/home2/model/Qwen3-1.7B")
    args = ap.parse_args()
    dev = torch.device("cuda")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(dev).eval()
    model = SlowPath(load_fo_encoder(args.fo_repo, args.fo_ckpt), llm, tok).to(dev)
    model.llm = get_peft_model(model.llm, LoraConfig(
        r=args.lora, lora_alpha=2 * args.lora, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    sd = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    model.eval()
    decider = Decider(model)

    def judge(seg: np.ndarray, prefix: str) -> tuple[float, float]:
        t0 = time.time()
        fb = fbank(torch.from_numpy(np.ascontiguousarray(seg))).unsqueeze(0).to(dev)
        pr = torch.from_numpy(prosody_25hz(seg)).unsqueeze(0).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = decider.score(fb, torch.tensor([fb.shape[1]], device=dev), pr, [prefix])
        torch.cuda.synchronize()
        return float(s[0]), time.time() - t0

    judge(np.zeros(16000, np.float32), "")  # 预热
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p for p in Path(args.probes).iterdir() if (p / "user.wav").exists())
    lat_all = []
    for s in sessions:
        meta = json.loads((s / "session.json").read_text(encoding="utf-8"))
        au = meta["agent_utts"][0]
        wav, sr = sf.read(s / "user.wav", dtype="float32")
        actions, trace = [], []
        for on, off in energy_vad(wav, sr):
            t = on + FIRST
            while True:
                t_eval = min(t, off)
                score, lat = judge(wav[int(on * sr): int(t_eval * sr)],
                                   played_text(au["text"], au["t_start"], au["t_end"], t_eval))
                lat_all.append(lat)
                trace.append({"t": round(t_eval, 3), "score": round(score, 3), "lat": round(lat, 3)})
                if score > args.thr:
                    actions.append({"t": round(t_eval + lat, 3), "kind": "CUT"})
                    break
                if t_eval >= off or t_eval - on >= MAX_WAIT:
                    break
                t += STEP
        (out / f"{s.name}.json").write_text(json.dumps({"actions": actions, "trace": trace}), encoding="utf-8")
    print(f"done -> {out}  judge latency p50={np.median(lat_all) * 1000:.0f}ms p90={np.percentile(lat_all, 90) * 1000:.0f}ms")


if __name__ == "__main__":
    main()
