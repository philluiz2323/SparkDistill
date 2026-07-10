#!/usr/bin/env bash
# Train a student checkpoint from an Axolotl recipe.
#
#   scripts/train.sh recipes/qwen3.5-4b-phase1/sft.yaml [--dry-run] [extra axolotl args...]
#
# --dry-run validates the recipe file and prints the command without running
# training (axolotl itself is an external dependency, installed separately —
# see README).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

recipe="${1:?usage: scripts/train.sh <recipe.yaml> [--dry-run] [extra axolotl args...]}"
shift

if [ ! -f "$recipe" ]; then
  echo "recipe not found: $recipe" >&2
  exit 1
fi
python3 -c "import yaml, sys; yaml.safe_load(open(sys.argv[1]))" "$recipe"

dry_run=false
args=()
for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    dry_run=true
  else
    args+=("$arg")
  fi
done

if [ "$dry_run" = true ]; then
  echo "[dry-run] would run: axolotl train $recipe ${args[*]:-}"
  exit 0
fi

exec axolotl train "$recipe" "${args[@]:-}"
