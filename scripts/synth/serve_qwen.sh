#!/usr/bin/env bash
# 服务器端:起本地 Qwen 的 OpenAI 兼容服务(vLLM),给 writer_llm.py 用。
#
# 数据生成推荐模型(2×48G 即可,不用大模型):
#   首选: Qwen/Qwen3-30B-A3B-Instruct-2507   (MoE 激活3B,快,口语质量够)
#   备选: Qwen/Qwen3-32B                      (dense,质量略稳,慢几倍)
# 下载(服务器国内网络用 modelscope 免代理):
#   pip install modelscope && modelscope download --model Qwen/Qwen3-30B-A3B-Instruct-2507 --local_dir ./Qwen3-30B-A3B-Instruct-2507
#
# 用法:
#   bash serve_qwen.sh <模型路径> [port=8002] [gpus=0,1]
# 例:
#   bash serve_qwen.sh ./Qwen3-30B-A3B-Instruct-2507 8002 0,1
#
# 说明:
#   - served-model-name 固定叫 qwen,客户端 --model qwen
#   - tensor-parallel-size 自动 = GPU 数
#   - --max-model-len 8192:writer 的 prompt+输出都很短,限长省 KV 显存
# 已知坑:
#   - flashinfer wheel 坏会崩 worker:`pip uninstall flashinfer` 回退 NCCL
#   - Instruct-2507 是无 thinking 版;客户端的 enable_thinking 参数若不被
#     模板接受,writer_llm.py 会自动去掉重试,无需处理

set -e
MODEL="${1:?用法: bash serve_qwen.sh <模型路径> [port] [gpus]}"
PORT="${2:-8002}"
GPUS="${3:-0,1}"
TP=$(echo "$GPUS" | awk -F, '{print NF}')

echo "model=$MODEL port=$PORT gpus=$GPUS tp=$TP"
CUDA_VISIBLE_DEVICES="$GPUS" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen \
  --tensor-parallel-size "$TP" \
  --max-model-len 8192 \
  --port "$PORT"
