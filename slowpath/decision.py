"""Stage B:打断决策(在 Stage A 的编码器 + 韵律旁路 + adapter 之上,LLM 加 LoRA)。

输入:「助手講到:<已播文本>」+ 客户声音(音频 token);输出 STOP / CONTINUE。
推理只比较两个答案首 token 的对数概率(一次前向,可给出连续分数供阈值校准)。
保留质疑/确认辅助头,防止决策训练冲掉韵律信息。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SYSTEM = "<|im_start|>system\n判斷助手應否停止講話。<|im_end|>\n"
POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ANSWERS = {1: "STOP", 0: "CONTINUE"}


class Decider:
    def __init__(self, slowpath):
        self.m = slowpath
        tok = slowpath.tok
        self.first_ids = {lab: tok(ans, add_special_tokens=False).input_ids[0] for lab, ans in ANSWERS.items()}
        assert self.first_ids[0] != self.first_ids[1], "STOP/CONTINUE 首 token 相同,无法打分"

    def _ids(self, s: str, dev) -> torch.Tensor:
        return self.m.tok(s, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)

    def _sequences(self, emb, lens, prefixes, targets=None):
        E = self.m.llm.get_input_embeddings()
        dev = emb.device
        post = E(self._ids(POST, dev))
        seqs, labs = [], []
        for b, prefix in enumerate(prefixes):
            pre = E(self._ids(f"{SYSTEM}<|im_start|>user\n助手講到:「{prefix}」\n客戶聲音:", dev))
            parts = [pre, emb[b, : int(lens[b])].to(pre.dtype), post]
            y = None
            if targets is not None:
                tgt = self._ids(ANSWERS[int(targets[b])] + "<|im_end|>", dev)
                parts.append(E(tgt))
            x = torch.cat(parts, 0)
            if targets is not None:
                y = torch.full((x.shape[0],), -100, dtype=torch.long, device=dev)
                y[-len(tgt):] = tgt
            seqs.append(x)
            labs.append(y)
        L = max(s.shape[0] for s in seqs)
        X = torch.stack([F.pad(s, (0, 0, L - s.shape[0], 0)) for s in seqs])  # 左填充:末位对齐,便于推理取最后一位
        att = torch.stack([F.pad(torch.ones(s.shape[0], dtype=torch.long, device=dev), (L - s.shape[0], 0))
                           for s in seqs])
        Y = None
        if targets is not None:
            Y = torch.stack([F.pad(y, (L - y.shape[0], 0), value=-100) for y in labs])
        return X, att, Y

    def loss(self, batch: dict, w_qs: float = 0.3):
        z, valid = self.m.frames(batch["fbank"], batch["fbank_len"], batch["pros"])
        emb, lens = self.m.audio_tokens(z, valid)
        X, att, Y = self._sequences(emb, lens, batch["prefix"], batch["label"])
        logits = self.m.llm(inputs_embeds=X, attention_mask=att).logits[:, :-1].float()
        l_dec = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y[:, 1:].reshape(-1), ignore_index=-100)
        parts = {"dec": float(l_dec)}
        total = l_dec
        qs_idx = [i for i, q in enumerate(batch["qs"]) if q >= 0]
        if qs_idx and w_qs > 0:
            lg = self.m.qs_logits(z[qs_idx], valid[qs_idx])
            y = torch.tensor([batch["qs"][i] for i in qs_idx], device=z.device, dtype=torch.float)
            l_qs = F.binary_cross_entropy_with_logits(lg, y)
            total = total + w_qs * l_qs
            parts["qs"] = float(l_qs)
        return total, parts

    @torch.no_grad()
    def score(self, fb, fb_len, pros, prefixes) -> torch.Tensor:
        """返回 STOP 相对 CONTINUE 的首 token 对数几率差(>0 倾向停)。"""
        z, valid = self.m.frames(fb, fb_len, pros)
        emb, lens = self.m.audio_tokens(z, valid)
        X, att, _ = self._sequences(emb, lens, prefixes)
        last = self.m.llm(inputs_embeds=X, attention_mask=att).logits[:, -1].float()
        lp = torch.log_softmax(last, -1)
        return lp[:, self.first_ids[1]] - lp[:, self.first_ids[0]]
