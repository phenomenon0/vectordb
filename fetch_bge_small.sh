#!/usr/bin/env bash
set -euo pipefail

dest="vectordb/models/bge-small-en-v1.5"
mkdir -p "$dest"

echo "Downloading BGE Small (ONNX) and tokenizer into $dest"

try_fetch() {
  url="$1"
  out="$2"
  echo "  -> $out from $url"
  if curl -L --fail -o "$out" "$url"; then
    echo "     ok"
    return 0
  fi
  echo "     failed"
  return 1
}

# Model candidates (pick first that works)
model_targets=(
  "https://huggingface.co/Teradata/bge-small-en-v1.5/resolve/main/onnx/model.onnx?download=1"
  "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx?download=1"
  "https://huggingface.co/BAAI/bge-small-en/resolve/main/onnx/model.onnx?download=1"
)

tokenizer_targets=(
  "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/tokenizer.json?download=1"
  "https://huggingface.co/Teradata/bge-small-en-v1.5/resolve/main/tokenizer.json?download=1"
  "https://huggingface.co/BAAI/bge-small-en/resolve/main/tokenizer.json?download=1"
)

ok=0
for u in "${model_targets[@]}"; do
  if try_fetch "$u" "$dest/model.onnx"; then
    ok=1
    break
  fi
done

tok_ok=0
for u in "${tokenizer_targets[@]}"; do
  if try_fetch "$u" "$dest/tokenizer.json"; then
    tok_ok=1
    break
  fi
done

if [[ $ok -ne 1 || $tok_ok -ne 1 ]]; then
  echo "Warning: downloads failed; you may need a Hugging Face token or different URLs."
  exit 1
fi

echo "Done."
