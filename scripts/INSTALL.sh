#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${NIODOO_MODEL_PATH:-$ROOT_DIR/model/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf}"
PARTICLES_PATH="${NIODOO_PARTICLES_PATH:-$ROOT_DIR/universe_top60000.safetensors}"
TOKEN_MAP_PATH="${NIODOO_TOKEN_MAP_PATH:-$ROOT_DIR/universe_top60000_token_map.json}"

cd "$ROOT_DIR"

echo "[1/3] Checking local assets..."
missing=0

for path in "$MODEL_PATH" "$PARTICLES_PATH" "$TOKEN_MAP_PATH"; do
  if [ ! -f "$path" ]; then
    echo "missing: $path"
    missing=1
  fi
done

if [ "$missing" -ne 0 ]; then
  cat <<'EOF'

Local assets are missing.

Expected files:
- model/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
- universe_top60000.safetensors
- universe_top60000_token_map.json

You can place them anywhere and override with:
- NIODOO_MODEL_PATH
- NIODOO_PARTICLES_PATH
- NIODOO_TOKEN_MAP_PATH
EOF
  exit 1
fi

echo "[2/3] Building release binary..."
cargo build --release --bin niodoo --offline

echo "[3/3] Ready."
echo
echo "Recommended model:"
echo "  Bartowski Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"
echo
echo "Note:"
echo "  Niodoo is still experimental. Steering behavior depends on the model and quantization."
echo
echo "Chat:"
echo "  python3 scripts/chat_raw.py --max-steps 512"
echo
echo "Direct:"
echo "  ./target/release/niodoo --model-path \"$MODEL_PATH\" --particles-path \"$PARTICLES_PATH\" --n 60000 --max-steps 512 --prompt \"Hello\""
