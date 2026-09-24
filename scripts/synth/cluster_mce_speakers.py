"""【服务器】MCE 说话人聚类:MCE 同一人对应多个目录且无说话人元数据。

每个目录取 3 段 3–10s 录音,用 CAM++ 声纹模型(funasr / ModelScope
iic/speech_campplus_sv_zh-cn_16k-common)提取声纹并取平均,余弦相似度做
层次聚类(平均连接,阈值可调),输出 目录 → 说话人簇。
render_stage0 据此"每个说话人只取一个音色",并按说话人划分 train / val。

环境:服务器 funasr env
用法:python cluster_mce_speakers.py --out /home/pxieaf/home2/data/stage0/mce_speakers.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mce", default="/home/share/data_makchen/peng/datasets/MCE/MCE_Dataset/Audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.6, help="簇间平均余弦相似度 ≥ 此值则合并")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from funasr import AutoModel
    sv = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", device=args.device,
                   disable_update=True)

    def embed(path: Path) -> np.ndarray:
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        res = sv.generate(input=wav)
        e = res[0]["spk_embedding"]
        e = e.detach().cpu().numpy() if hasattr(e, "detach") else np.asarray(e)
        e = e.reshape(-1)
        return e / (np.linalg.norm(e) + 1e-9)

    folders, embs = [], []
    for d in sorted(p for p in Path(args.mce).iterdir() if p.is_dir()):
        clips = [w for w in sorted(d.glob("*.wav")) if 3.0 <= sf.info(w).duration <= 10.0][:3]
        if not clips:
            continue
        e = np.mean([embed(w) for w in clips], axis=0)
        folders.append(d.name)
        embs.append(e / (np.linalg.norm(e) + 1e-9))
    X = np.stack(embs)
    S = X @ X.T
    print(f"{len(folders)} folders embedded; sim p50={np.median(S[np.triu_indices(len(X), 1)]):.3f}")

    # 平均连接层次聚类(纯 numpy,规模 ~160)
    clusters = [[i] for i in range(len(folders))]
    while True:
        best, pair = -1.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                s = S[np.ix_(clusters[a], clusters[b])].mean()
                if s > best:
                    best, pair = s, (a, b)
        if pair is None or best < args.threshold:
            break
        a, b = pair
        clusters[a] += clusters.pop(b)
    mapping = {folders[i]: f"spk{c:03d}" for c, members in enumerate(clusters) for i in members}
    sizes = sorted((len(m) for m in clusters), reverse=True)
    print(f"{len(clusters)} speaker clusters at threshold {args.threshold}; sizes top10={sizes[:10]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"threshold": args.threshold, "folder2spk": mapping},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", args.out)


if __name__ == "__main__":
    main()
