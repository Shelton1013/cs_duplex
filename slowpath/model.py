"""慢通路模型(Stage A:模态对齐 + 韵律保持)。

音频 → FO 编码器(1024@25Hz) ⊕ 韵律旁路(8→64 @25Hz) → 相邻两帧拼接(12.5Hz)
     → 2 层 MLP adapter → LLM 词向量空间 → 以 inputs_embeds 注入冻结的 Qwen3-1.7B。
Stage A 三个目标(PLAN §2.3 "两个丢失点"):
  asr  :LLM 输出转写(主任务,把音频对齐到 LLM 语义空间)
  tone :编码器⊕韵律帧上的声调 CTC(1–6 声 + 英文/其他),迫使表征保留音高
  qs   :池化后的质疑/确认二分类(用 Stage 0 语调数据训练集说话人),直接监督语调
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_TONE = 7  # 1..6 粤语声调, 7 = 英文词/非粤语;CTC blank = 0

PROMPT_PRE = "<|im_start|>user\n"
PROMPT_POST = "请转写这段音频。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class SlowPath(nn.Module):
    def __init__(self, encoder: nn.Module, llm: nn.Module, tokenizer, enc_dim: int = 1024,
                 pros_in: int = 8, pros_dim: int = 64, stack: int = 2):
        super().__init__()
        self.encoder, self.llm, self.tok = encoder, llm, tokenizer
        self.stack = stack
        llm_dim = llm.get_input_embeddings().weight.shape[1]
        feat = enc_dim + pros_dim
        self.pros_mlp = nn.Sequential(nn.Linear(pros_in, pros_dim), nn.GELU(), nn.Linear(pros_dim, pros_dim))
        self.adapter = nn.Sequential(nn.Linear(feat * stack, llm_dim), nn.GELU(),
                                     nn.Linear(llm_dim, llm_dim), nn.LayerNorm(llm_dim))
        self.tone_head = nn.Linear(feat, N_TONE + 1)
        self.qs_head = nn.Sequential(nn.Linear(feat * 2, 256), nn.GELU(), nn.Linear(256, 1))
        for p in self.llm.parameters():
            p.requires_grad = False
        ids = lambda s: tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids[0]
        self.register_buffer("pre_ids", ids(PROMPT_PRE), persistent=False)
        self.register_buffer("post_ids", ids(PROMPT_POST), persistent=False)

    # ---- 音频 → 帧特征 ----
    def frames(self, fb: torch.Tensor, fb_len: torch.Tensor, pros: torch.Tensor):
        """fb (B,T100,80), pros (B,T25,8) → z (B,T25',1088), valid (B,T25')"""
        h, mask = self.encoder(fb, fb_len)
        valid = mask.squeeze(1)
        T = h.shape[1]
        if pros.shape[1] < T:
            pros = F.pad(pros, (0, 0, 0, T - pros.shape[1]))
        p = self.pros_mlp(pros[:, :T].to(h.dtype))
        return torch.cat([h, p], -1), valid

    def audio_tokens(self, z: torch.Tensor, valid: torch.Tensor):
        B, T, D = z.shape
        T2 = T // self.stack
        zz = z[:, : T2 * self.stack].reshape(B, T2, D * self.stack)
        lens = valid[:, : T2 * self.stack].reshape(B, T2, self.stack).all(-1).sum(-1)
        return self.adapter(zz), lens

    # ---- 损失 ----
    def asr_loss(self, emb: torch.Tensor, lens: torch.Tensor, texts: list[str]):
        E = self.llm.get_input_embeddings()
        pre, post = E(self.pre_ids), E(self.post_ids)
        seqs, labs = [], []
        for b, text in enumerate(texts):
            tgt = self.tok(text + "<|im_end|>", add_special_tokens=False,
                           return_tensors="pt").input_ids[0].to(emb.device)
            a = emb[b, : int(lens[b])]
            x = torch.cat([pre, a.to(pre.dtype), post, E(tgt)], 0)
            y = torch.full((x.shape[0],), -100, dtype=torch.long, device=emb.device)
            y[-len(tgt):] = tgt
            seqs.append(x)
            labs.append(y)
        L = max(s.shape[0] for s in seqs)
        X = torch.stack([F.pad(s, (0, 0, 0, L - s.shape[0])) for s in seqs])
        Y = torch.stack([F.pad(y, (0, L - y.shape[0]), value=-100) for y in labs])
        att = torch.stack([F.pad(torch.ones(s.shape[0], device=emb.device, dtype=torch.long),
                                 (0, L - s.shape[0])) for s in seqs])
        out = self.llm(inputs_embeds=X, attention_mask=att)
        logits = out.logits[:, :-1].float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y[:, 1:].reshape(-1),
                               ignore_index=-100)

    def tone_loss(self, z, valid, tones: list[list[int]]):
        idx = [i for i, t in enumerate(tones) if t]
        if not idx:
            return z.new_zeros(())
        logp = F.log_softmax(self.tone_head(z[idx]).float(), -1).transpose(0, 1)  # (T,B,C)
        in_len = valid[idx].sum(-1)
        tgt = [torch.tensor(tones[i], device=z.device) for i in idx]
        tgt_len = torch.tensor([len(t) for t in tgt], device=z.device)
        return F.ctc_loss(logp, torch.cat(tgt), in_len, tgt_len, blank=0, zero_infinity=True)

    def qs_logits(self, z, valid):
        m = valid.unsqueeze(-1).to(z.dtype)
        mean = (z * m).sum(1) / m.sum(1).clamp(min=1)
        # 句尾池化:最后 1/3 有效帧,语调信息主要在句尾
        tails = []
        for b in range(z.shape[0]):
            n = int(valid[b].sum())
            tails.append(z[b, max(0, n - max(1, n // 3)): n].mean(0))
        return self.qs_head(torch.cat([mean, torch.stack(tails)], -1).float()).squeeze(-1)

    def forward(self, batch: dict, w_tone: float = 0.3, w_qs: float = 0.3):
        z, valid = self.frames(batch["fbank"], batch["fbank_len"], batch["pros"])
        losses = {}
        asr_idx = [i for i, t in enumerate(batch["text"]) if t]
        if asr_idx:
            emb, lens = self.audio_tokens(z[asr_idx], valid[asr_idx])
            losses["asr"] = self.asr_loss(emb, lens, [batch["text"][i] for i in asr_idx])
            losses["tone"] = self.tone_loss(z[asr_idx], valid[asr_idx], [batch["tones"][i] for i in asr_idx])
        qs_idx = [i for i, q in enumerate(batch["qs"]) if q >= 0]
        if qs_idx:
            lg = self.qs_logits(z[qs_idx], valid[qs_idx])
            y = torch.tensor([batch["qs"][i] for i in qs_idx], device=z.device, dtype=torch.float)
            losses["qs"] = F.binary_cross_entropy_with_logits(lg, y)
        total = losses.get("asr", 0) + w_tone * losses.get("tone", 0) + w_qs * losses.get("qs", 0)
        return total, {k: float(v) for k, v in losses.items()}

    @torch.no_grad()
    def transcribe(self, fb, fb_len, pros, max_new_tokens: int = 96) -> list[str]:
        z, valid = self.frames(fb, fb_len, pros)
        emb, lens = self.audio_tokens(z, valid)
        E = self.llm.get_input_embeddings()
        pre, post = E(self.pre_ids), E(self.post_ids)
        outs = []
        for b in range(emb.shape[0]):
            x = torch.cat([pre, emb[b, : int(lens[b])].to(pre.dtype), post], 0).unsqueeze(0)
            ids = self.llm.generate(inputs_embeds=x, max_new_tokens=max_new_tokens, do_sample=False,
                                    eos_token_id=self.tok.convert_tokens_to_ids("<|im_end|>"))
            outs.append(self.tok.decode(ids[0], skip_special_tokens=True).strip())
        return outs
