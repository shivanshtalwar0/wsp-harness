#!/usr/bin/env bash
# End-to-end verification of the weight-side pipeline against a local checkpoint.
#
#   ./scripts/smoke.sh              # read-only: manifest, quant check, atlas, dry-run
#   ./scripts/smoke.sh --apply      # also writes a patched copy (~16 GB) and diffs it
#
set -euo pipefail

M=${M:-$HOME/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-4bit}
TENSOR=language_model.model.layers.19.mlp.down_proj
OUT=runs

echo "== model: $M"
[[ -d "$M" ]] || { echo "set M=<model dir>"; exit 2; }
mkdir -p "$OUT"

echo "== 1/6 unit tests"
python -m pytest -q

echo "== 2/6 quantizer cross-check against mlx"
python -m wsp.cli verify-mlx

echo "== 3/6 manifest"
python -m wsp.cli manifest "$M" | tee "$OUT/manifest.json" | head -12

echo "== 4/6 atlas (layers 15,19 mlp.down_proj)"
python -m wsp.cli atlas "$M" --layers 15,19 --suffix mlp.down_proj --out "$OUT/atlas.json"

echo "== 5/6 analytic patch"
python -m wsp.cli analytic "$M" --tensor "$TENSOR" --out "$OUT/patch.json" | head -4

echo "== 6/6 dry-run validation"
python -m wsp.cli apply "$M" "$OUT/patch.json" --dry-run

if [[ "${1:-}" == "--apply" ]]; then
  echo "== extra: applying for real -> $OUT/patched"
  python -m wsp.cli apply "$M" "$OUT/patch.json" --out "$OUT/patched" | tail -20
  echo "== extra: code-level diff"
  python -m wsp.cli diff "$M" "$OUT/patched"
  echo
  echo "Patched checkpoint at $OUT/patched (~16 GB)."
  echo "Load it with:  python -c \"from mlx_lm import load; load('$OUT/patched')\""
fi

echo "OK"
