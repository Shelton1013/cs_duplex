# 环境配置

项目分三套环境,按需装——**核心代码零依赖**,大多数脚本装完 Python 就能跑。

## 0. 系统工具(所有机器)

- **ffmpeg**:音频转码(diar/爬取管线用)。Windows 下载 gyan.dev 全量包;Linux `apt install ffmpeg` / conda `conda install -c conda-forge ffmpeg`
- **yt-dlp**(仅爬 YouTube 的机器):用独立版而非 pip(pip 版受 Python 版本限制,py3.9 装不到新版会撞 YouTube 反爬):
  ```
  curl -L -o scripts/crawl/yt-dlp.exe https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe   # Windows
  curl -L -o scripts/crawl/yt-dlp https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp && chmod +x scripts/crawl/yt-dlp   # Linux
  ```
  `fetch_youtube.py` 自动优先用同目录的独立版。

## 1. 核心环境(harness / synth / crawl / timing_stats / writer_llm 客户端)

纯标准库,Python ≥ 3.9 即可(本仓库在 3.9 与 3.10 上均验证过):

```bash
conda create -n cduplex-core python=3.10 -y
conda activate cduplex-core
pip install -r requirements.txt        # 只有 pytest
python -m pytest tests -q              # 应 15 passed
```

爬取脚本在内地网络需本地代理(默认 `http://127.0.0.1:7897`,可 `--proxy` 覆盖;RTHK 直连会被重置)。

## 2. diarization 环境(scripts/diar/run_diar.py,需 GPU,8GB 显存足够)

版本组合已踩坑锁死,**安装顺序不能换**(pyannote 会拖 CPU 版 torch,必须最后覆盖回 CUDA 版):

```bash
conda create -n cduplex-diar python=3.10 -y
conda activate cduplex-diar
pip install -r requirements-diar.txt                 # pyannote 3.3.2 + hub 0.27.1
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # 必须 True
```

首次运行需要 HuggingFace 授权(gated 模型):
1. 网页登录 HF,在 https://hf.co/pyannote/speaker-diarization-3.1 与 https://hf.co/pyannote/segmentation-3.0 各点一次同意;
2. `python -c "from huggingface_hub import login; login('hf_你的token')"`

验证:`python scripts/diar/run_diar.py --limit 2`(RTX 4060 上 RTF≈0.013)。

## 3. 服务器 vLLM 环境(数据生成,2×48G)

```bash
conda create -n cduplex-vllm python=3.10 -y
conda activate cduplex-vllm
pip install vllm modelscope
# 模型下载(国内免代理):
modelscope download --model Qwen/Qwen3-30B-A3B-Instruct-2507 --local_dir ./Qwen3-30B-A3B-Instruct-2507
# 起服务:
bash scripts/synth/serve_qwen.sh ./Qwen3-30B-A3B-Instruct-2507 8002 0,1
```

已知坑:flashinfer wheel 损坏会崩 worker → `pip uninstall flashinfer` 回退 NCCL。
`writer_llm.py` 客户端是纯标准库,在核心环境甚至系统 Python 里跑都行。

## 环境-脚本对照表

| 脚本 | 环境 | GPU |
|---|---|---|
| tests / gen_v0 / timing_stats / fetch_* / writer_llm(客户端) | core | 否 |
| scripts/diar/run_diar.py | diar | 是(≥4GB) |
| scripts/synth/serve_qwen.sh | vllm(服务器) | 2×48G |
