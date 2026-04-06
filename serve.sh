#!/bin/bash
# Serve REAP-21B with OpenAI-compatible API
# Usage: ./serve.sh [port]
#
# Then use with any OpenAI client:
#   curl http://localhost:8080/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}]}'

PORT=${1:-8080}
MODEL="ukint-vs/gemma-4-21b-a4b-it-REAP-MLX-4bit"

echo "Starting REAP-21B server on port $PORT..."
echo "API: http://localhost:$PORT/v1/chat/completions"
echo ""

python -m mlx_vlm.server --model "$MODEL" --port "$PORT"
