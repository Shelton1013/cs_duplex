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
    """fuse=True(Stage B v1):决策分数 = LLM 首 token 几率差 + w·韵律分支 logit + b,
    w、b 与各模块一起在决策 BCE 上学习;韵律分支另受质疑/确认监督,锚定其语调含义。"""

    def __init__(self, slowpath, fuse: bool = False):
        self.m = slowpath
        self.fuse = fuse
        if fuse and not hasattr(slowpath, "fuse_w"):
            dev = next(slowpath.parameters()).device
            slowpath.fuse_w = torch.nn.Parameter(torch.tensor(0.5, device=dev))
            slowpath.fuse_b = torch.nn.Parameter(torch.tensor(0.0, device=dev))
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

    def _answer_margin(self, logits_last: torch.Tensor) -> torch.Tensor:
        lp = torch.log_softmax(logits_last.float(), -1)
        return lp[:, self.first_ids[1]] - lp[:, self.first_ids[0]]

    def loss(self, batch: dict, w_qs: float = 0.3):
        z, valid = self.m.frames(batch["fbank"], batch["fbank_len"], batch["pros"])
        emb, lens = self.m.audio_tokens(z, valid)
        X, att, Y = self._sequences(emb, lens, batch["prefix"], batch["label"])
        logits = self.m.llm(inputs_embeds=X, attention_mask=att).logits
        lg = logits[:, :-1].float()
        l_dec = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), Y[:, 1:].reshape(-1), ignore_index=-100)
        parts = {"dec": float(l_dec)}
        total = l_dec
        y_dec = torch.tensor(batch["label"], device=z.device, dtype=torch.float)
        qs_idx = [i for i, q in enumerate(batch["qs"]) if q >= 0]
        if self.fuse:
            # 答案首 token 的位置:目标序列前一位(左填充,答案占 len(tgt) 个位置,末位为 <|im_end|>)
            first_pos = (Y != -100).float().argmax(1) - 1
            margin = self._answer_margin(logits[torch.arange(len(first_pos)), first_pos])
            pl = self.m.pros_logits(batch["pros"], batch["fbank_len"])
            fused = margin + self.m.fuse_w * pl + self.m.fuse_b
            l_fuse = F.binary_cross_entropy_with_logits(fused, y_dec)
            total = total + l_fuse
            parts["fuse"] = float(l_fuse)
            if qs_idx:
                yq = torch.tensor([batch["qs"][i] for i in qs_idx], device=z.device, dtype=torch.float)
                l_pq = F.binary_cross_entropy_with_logits(pl[qs_idx], yq)
                total = total + l_pq
                parts["pros_qs"] = float(l_pq)
        if qs_idx and w_qs > 0:
            lq = self.m.qs_logits(z[qs_idx], valid[qs_idx])
            yq = torch.tensor([batch["qs"][i] for i in qs_idx], device=z.device, dtype=torch.float)
            l_qs = F.binary_cross_entropy_with_logits(lq, yq)
            total = total + w_qs * l_qs
            parts["qs"] = float(l_qs)
        return total, parts

    @torch.no_grad()
    def score(self, fb, fb_len, pros, prefixes) -> torch.Tensor:
        """返回 STOP 相对 CONTINUE 的首 token 对数几率差(>0 倾向停)。"""
        z, valid = self.m.frames(fb, fb_len, pros)
        emb, lens = self.m.audio_tokens(z, valid)
        X, att, _ = self._sequences(emb, lens, prefixes)
        margin = self._answer_margin(self.m.llm(inputs_embeds=X, attention_mask=att).logits[:, -1])
        if self.fuse:
            margin = margin + self.m.fuse_w * self.m.pros_logits(pros, fb_len) + self.m.fuse_b
        return margin
