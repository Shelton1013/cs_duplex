"""【服务器】基线:Qwen3-Omni-30B-A3B 作"本地最强音频理解裁判"。

Qwen3-Omni 是半双工(survey 记 T0),没有说话中途被打断的机制;这里把它当作
可本地部署的最强音频大模型,检验"直接听音频的大模型能否做对打断判断"——
尤其语调最小对(I2)与旁侧对话(说话对象)这两类文本转写解决不了的探针。

因果流程(同基线 ② 的检查点制):每个 VAD 语音段,从 onset+0.4s 起每 0.3s
把 [onset, t] 的用户音频 + agent 截至 t 已播出的文本交给模型,只让它输出
STOP / CONTINUE;STOP → CUT(时刻 = t + 模型实测推理耗时)。
只用 Thinker 生成文本(disable_talker),不生成语音。

环境:服务器 /home/pxieaf/home2/envs/qwen3omni(transformers 含 Qwen3OmniMoe)
显存:bf16 约 70GB → 2 张 48G 卡(device_map=auto)。
用法:
  CUDA_VISIBLE_DEVICES=0,6 python run_qwen3omni_judge.py --probes data/probe_audio_v2 \
      --out data/baselines_v2/qwen3omni --model /home/pxieaf/home2/model/Qwen3-Omni-30B-A3B-Instruct
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
from run_vad import energy_vad  # noqa: E402

FIRST, STEP, MAX_WAIT = 0.4, 0.3, 2.4

SYSTEM = ("你是粤语电话客服语音助手的「打断判断器」。助手正在向客户讲话时,"
          "麦克风收到了客户那边的声音。你要判断:助手应该立即停下来(STOP),"
          "还是继续讲下去(CONTINUE)。\n"
          "应 STOP:客户在纠正助手刚讲的内容、质疑(例如用升调重复一个数字表示怀疑)、"
          "或明确想插话提问。\n"
          "应 CONTINUE:客户只是应声表示在听(嗯、係、好、ok)、用降调重复内容表示确认、"
          "在跟身边其他人讲话(不是对助手讲)、或只是噪音。\n"
          "只输出一个词:STOP 或 CONTINUE。")


def played_text(agent_text: str, t_start: float, t_end: float, t: float) -> str:
    frac = min(1.0, max(0.0, (t - t_start) / max(1e-6, t_end - t_start)))
    return agent_text[: int(len(agent_text) * frac)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()

    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto")
    model.disable_talker()
    proc = Qwen3OmniMoeProcessor.from_pretrained(args.model)

    def judge(audio: np.ndarray, agent_prefix: str) -> tuple[str, float]:
        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "text", "text": f"助手目前已经讲到:「{agent_prefix}」\n以下是麦克风收到的客户声音:"},
                {"type": "audio", "audio": audio},
                {"type": "text", "text": "助手应该 STOP 还是 CONTINUE?"},
            ]},
        ]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audio=[audio], return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                 return_audio=False, thinker_return_dict_in_generate=True)
        seq = out.sequences if hasattr(out, "sequences") else out
        ans = proc.batch_decode(seq[:, inputs["input_ids"].shape[1]:],
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
                stop = "STOP" in ans
                trace.append({"t": round(t_eval, 3), "ans": ans, "lat": round(lat, 3)})
                if stop:
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
