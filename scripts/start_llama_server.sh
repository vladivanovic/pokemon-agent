#!/usr/bin/env bash
# Start the local llama.cpp server that backs Hermes' high-level brain.
#
# WHY --reasoning off MATTERS (do not drop it):
#   Qwen3.5 emits its chain of thought via the chat template. With thinking
#   enabled the model spends the ENTIRE token budget on `reasoning_content` and
#   returns `content: ""` with `finish_reason: "length"` - measured at 120
#   tokens, 4.96s, and NO usable text. Hermes then retries and burns its turn
#   timeout, which is what produced the "hermes timed out after 240s" lines in
#   the autopilot log. With `--reasoning off` the same prompt returns
#   content "PONG" in 3 tokens / 0.21s.
#
#   The Hermes side already passes --reasoning none to its own CLI, but that
#   does NOT change what the server's chat template does - this flag does.
#
# Usage:  scripts/start_llama_server.sh          (foreground)
#         nohup scripts/start_llama_server.sh &  (background)
set -euo pipefail

MODEL="${POKEMON_LLAMA_MODEL:-/home/vlad/models/unsloth-Qwen3.5-2B-GGUF/Qwen3.5-2B-UD-Q8_K_XL.gguf}"
PORT="${POKEMON_LLAMA_PORT:-8000}"
CTX="${POKEMON_LLAMA_CTX:-65536}"
LLAMA_DIR="${POKEMON_LLAMA_DIR:-/home/vlad/llama.cpp}"
LOG="${POKEMON_LLAMA_LOG:-/tmp/llama_qwen65k.log}"

if [ ! -x "$LLAMA_DIR/build/bin/llama-server" ]; then
    echo "llama-server not found at $LLAMA_DIR/build/bin/llama-server" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "model not found: $MODEL" >&2
    exit 1
fi

echo "[llama] model=$(basename "$MODEL") port=$PORT ctx=$CTX log=$LOG"
cd "$LLAMA_DIR"
# -ngl 999 / -t 8 / -tb 8 are for this Orin box; -fa on + q8_0 KV halves memory.
exec env LLAMA_ARG_N_PREDICT=1500 ./build/bin/llama-server \
    -m "$MODEL" \
    -c "$CTX" \
    -fa on \
    -ctk q8_0 -ctv q8_0 \
    -ngl 999 \
    -t 8 -tb 8 \
    -np 1 \
    --n-predict 1500 \
    --reasoning off \
    --host 0.0.0.0 \
    --port "$PORT" >> "$LOG" 2>&1
