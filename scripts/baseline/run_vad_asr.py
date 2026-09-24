"""【服务器】基线 ②:VAD + 流式 ASR + 应声词表(一个"聪明一点的级联")。

因果流程(每个 VAD 语音段):
  在 onset+0.3s 起每 0.2s 一个检查点,用 SenseVoice-Small(粤语)转写 [onset, t] 的前缀:
    - 含否定/打断词(唔係/no/wait/等等…)          → CUT
    - 去掉应声词后仍有内容                         → CUT(不是应声)
    - 说话已超过 1.2s(应声不会这么长)            → CUT
    - 只有应声词或还没转写出字                     → 继续等
  语音段结束时仍只有应声 → 不停。
CUT 时刻 = 检查点 t + ASR 耗时(实测,反映真实延迟)。

这就是业界常见的"VAD + 关键词白名单"打断策略;它能分开应声与其他话,
但分不开旁侧对话与纠正(都算"有内容")——正是要量化的地方。

环境:服务器 funasr env(已缓存 iic/SenseVoiceSmall)。
用法:
  python run_vad_asr.py --probes data/probe_audio --out data/baselines/vad_asr
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vad import energy_vad  # noqa: E402

BACKCHANNEL_TOKENS = ["明白", "okay", "ok", "yeah", "yes", "right", "uhhuh", "hmm", "mm",
                      "係", "系", "啱", "岩", "好", "嗯", "唔", "哦", "噢", "呀", "啊", "吖",
                      "對", "对",
                      "嘛", "喎", "囉", "咯", "啦"]  # 句末语气助词(「啱嘛」「係囉」)
INTERRUPT_MARKERS = ["唔係", "唔系", "唔啱", "唔岩", "等等", "等陣", "wait", "no", "not", "sorry"]
FIRST_CHECK, STEP, MAX_BC = 0.3, 0.2, 1.2


def clean(text: str) -> str:
    text = re.sub(r"<\|[^|]*\|>", "", text)
    text = re.sub(r"[^\w]", "", text.lower())
    text = re.sub(r"([a-z])\1+", r"\1", text)  # 重复字母归一:ook/okk → ok
    return text.replace("_", "")


def classify(text: str) -> str:
    """返回 'interrupt' / 'content' / 'backchannel' / 'empty'(empty=继续等)。"""
    t = clean(text)
    if not t:
        return "empty"
    if any(m in t for m in INTERRUPT_MARKERS):
        return "interrupt"
    rest = t
    for tok in sorted(BACKCHANNEL_TOKENS, key=len, reverse=True):
        rest = rest.replace(tok, "")
    if not rest:
        return "backchannel"
    # 流式前缀被截在应声词中间(「明」←明白,「o」←ok):未定,等下一个检查点
    if any(tok.startswith(rest) and tok != rest for tok in BACKCHANNEL_TOKENS):
        return "empty"
    return "content"


def build_asr(kind: str, device: str, model_path: str):
    """返回 asr(seg)->(text, seconds)。
    sensevoice:SenseVoice-Small(language=yue);env=funasr
    qwen3asr  :Qwen3-ASR-1.7B-hf(transformers 原生;language 留空以支持句内混语,
               调用方式参照 qwen3_asr_mce 工具包已验证的 runner);env=qwen3omni
    """
    if kind == "sensevoice":
        from funasr import AutoModel
        model = AutoModel(model="iic/SenseVoiceSmall", device=device, disable_update=True)

        def asr(seg: np.ndarray) -> tuple[str, float]:
            t0 = time.time()
            res = model.generate(input=seg, cache={}, language="yue", use_itn=False)
            return (res[0]["text"] if res else ""), time.time() - t0
        return asr

    if kind == "qwen3asr":
        import tempfile
        import torch
        import transformers
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_path)
        cls = next(getattr(transformers, n) for n in
                   ("AutoModelForMultimodalLM", "Qwen3ASRForConditionalGeneration",
                    "AutoModelForSpeechSeq2Seq") if hasattr(transformers, n))
        model = cls.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device).eval()
        tmp = Path(tempfile.mkdtemp()) / "seg.wav"

        def asr(seg: np.ndarray) -> tuple[str, float]:
            t0 = time.time()
            if len(seg) < 1600:  # <0.1s 转写无意义
                return "", time.time() - t0
            sf.write(tmp, seg, 16000)
            inputs = proc.apply_transcription_request(audio=str(tmp))
            inputs = inputs.to(model.device, model.dtype)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=48, do_sample=False)
            text = proc.decode(ids[:, inputs["input_ids"].shape[1]:],
                               return_format="transcription_only")
            text = text[0] if isinstance(text, list) else text
            return text.strip(), time.time() - t0
        return asr
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--asr", default="sensevoice", choices=["sensevoice", "qwen3asr"])
    ap.add_argument("--asr_model", default="/home/pxieaf/home2/model/Qwen3-ASR-1.7B-hf")
    args = ap.parse_args()

    asr = build_asr(args.asr, args.device, args.asr_model)

    asr(np.zeros(8000, np.float32))  # 预热,避免首条延迟虚高
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p for p in Path(args.probes).iterdir() if (p / "user.wav").exists())
    for j, s in enumerate(sessions):
        wav, sr = sf.read(s / "user.wav", dtype="float32")
        actions, trace = [], []
        for on, off in energy_vad(wav, sr):
            t = on + FIRST_CHECK
            while True:
                t_eval = min(t, off)
                text, lat = asr(wav[int(on * sr): int(t_eval * sr)])
                verdict = classify(text)
                trace.append({"t": round(t_eval, 3), "text": clean(text), "v": verdict,
                              "asr_s": round(lat, 3)})
                if verdict in ("interrupt", "content") or (t_eval - on) >= MAX_BC:
                    actions.append({"t": round(t_eval + lat, 3), "kind": "CUT"})
                    break
                if t_eval >= off:
                    break  # 段结束,只有应声 → 不停
                t += STEP
        (out / f"{s.name}.json").write_text(
            json.dumps({"actions": actions, "trace": trace}, ensure_ascii=False), encoding="utf-8")
        if (j + 1) % 50 == 0:
            print(f"{j + 1}/{len(sessions)}")
    print("done ->", out)


if __name__ == "__main__":
    main()
