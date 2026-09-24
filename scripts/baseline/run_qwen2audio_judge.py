"""【服务器】基线:Qwen2-Audio-7B-Instruct 作音频裁判(与 Qwen3-Omni 裁判同协议)。

复用 run_qwen3omni_judge 的提示词、检查点节奏与 STOP/CONTINUE 判定,保证可比;
回答"更小/更早的开源音频大模型是否同样在旁侧对话与语调最小对上失败"。
单卡即可(7B bf16 ≈ 16GB)。

环境:/home/pxieaf/home2/envs/qwen3omni(transformers 含 Qwen2Audio)
用法:
  CUDA_VISIBLE_DEVICES=9 python run_qwen2audio_judge.py --probes data/probe_audio_v2 \
      --out data/baselines_v2/qwen2audio --model /home/pxieaf/home2/model/Qwen2-Audio-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen3omni_judge import FIRST, MAX_WAIT, STEP, SYSTEM, played_text  # noqa: E402
from run_vad import energy_vad  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0").eval()

    def judge(audio: np.ndarray, agent_prefix: str) -> tuple[str, float]:
        conv = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": f"助手目前已经讲到:「{agent_prefix}」\n以下是麦克风收到的客户声音:"},
                {"type": "audio", "audio_url": "probe.wav"},
                {"type": "text", "text": "助手应该 STOP 还是 CONTINUE?"},
            ]},
        ]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        try:
            inputs = proc(text=text, audio=[audio], sampling_rate=16000,
                          return_tensors="pt", padding=True)
        except TypeError:  # 旧版 transformers 参数名为 audios
            inputs = proc(text=text, audios=[audio], sampling_rate=16000,
                          return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        t0 = time.time()
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        ans = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0].strip().upper()
        return ans, time.time() - t0

    judge(np.zeros(16000, np.float32), "")  # 预热
    k, n_sh = map(int, args.shard.split("/"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p for p in Path(args.probes).iterdir() if (p / "user.wav").exists())
    todo = [s for i, s in enumerate(sessions)
            if i % n_sh == k and not (out / f"{s.name}.json").exists()]
    print(f"{len(todo)} to run", flush=True)
    t_all = time.time()
    for j, s in enumerate(todo):
        meta = json.loads((s / "session.json").read_text(encoding="utf-8"))
        au = meta["agent_utts"][0]
        wav, sr = sf.read(s / "user.wav", dtype="float32")
        actions, trace = [], []
        for on, off in energy_vad(wav, sr):
            t = on + FIRST
            while True:
                t_eval = min(t, off)
                ans, lat = judge(wav[int(on * sr): int(t_eval * sr)],
                                 played_text(au["text"], au["t_start"], au["t_end"], t_eval))
                trace.append({"t": round(t_eval, 3), "ans": ans, "lat": round(lat, 3)})
                if "STOP" in ans:
                    actions.append({"t": round(t_eval + lat, 3), "kind": "CUT"})
                    break
                if t_eval >= off or t_eval - on >= MAX_WAIT:
                    break
                t += STEP
        (out / f"{s.name}.json").write_text(
            json.dumps({"actions": actions, "trace": trace}, ensure_ascii=False), encoding="utf-8")
        if (j + 1) % 25 == 0:
            print(f"{j + 1}/{len(todo)}  {(time.time() - t_all) / (j + 1):.1f}s/session", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
