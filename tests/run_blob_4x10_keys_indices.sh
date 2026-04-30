#!/usr/bin/env bash
# Bash wrapper that runs tests/test_blob_4x10_keys_indices.py and
# extracts keys/indices via grep.
#
# Usage:
#   bash tests/run_blob_4x10_keys_indices.sh
#
# Outputs (under tests/_out/):
#   blob_4x10_full.log     -- full stdout/stderr of the run
#   blob_4x10_neurons.txt  -- Equivalent_Neurons_Index entries
#   blob_4x10_weights.txt  -- weight entries (front/back/plain)
#   blob_4x10_betas.txt    -- Equivalent_Betas_Index entries
#   blob_4x10_elements.txt -- ElementsinConstraintsObjectives entries

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OUT_DIR="$PROJECT_ROOT/tests/_out"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/blob_4x10_full.log"
NEURONS_FILE="$OUT_DIR/blob_4x10_neurons.txt"
WEIGHTS_FILE="$OUT_DIR/blob_4x10_weights.txt"
BETAS_FILE="$OUT_DIR/blob_4x10_betas.txt"
ELEMENTS_FILE="$OUT_DIR/blob_4x10_elements.txt"

# Activate conda env (per CLAUDE.md)
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate certif
fi

# Run the python test, tee output to the log
python -u "$PROJECT_ROOT/tests/test_blob_4x10_keys_indices.py" 2>&1 | tee "$LOG_FILE"

# Extract specific tags into separate files
grep -E '^\[KI-NEURON\]'  "$LOG_FILE" > "$NEURONS_FILE"  || true
grep -E '^\[KI-WEIGHT\]'  "$LOG_FILE" > "$WEIGHTS_FILE"  || true
grep -E '^\[KI-BETA\]'    "$LOG_FILE" > "$BETAS_FILE"    || true
grep -E '^\[KI-ELEM\]'    "$LOG_FILE" > "$ELEMENTS_FILE" || true

echo "------------------------------------------------------------"
echo "Full log     : $LOG_FILE"
echo "Neurons      : $NEURONS_FILE  ($(wc -l < "$NEURONS_FILE")  lines)"
echo "Weights      : $WEIGHTS_FILE  ($(wc -l < "$WEIGHTS_FILE")  lines)"
echo "Betas        : $BETAS_FILE    ($(wc -l < "$BETAS_FILE")    lines)"
echo "Elements     : $ELEMENTS_FILE ($(wc -l < "$ELEMENTS_FILE") lines)"
echo "------------------------------------------------------------"
