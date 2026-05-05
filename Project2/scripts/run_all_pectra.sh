#!/usr/bin/env bash
# Run every training notebook in order on pectra (or anywhere uv + CUDA work).
#
# Usage:
#   bash scripts/run_all_pectra.sh                # full run (8-15h on RTX 5090)
#   EPOCH_OVERRIDE=2 bash scripts/run_all_pectra.sh   # quick smoke (~30 min)
#   ONLY=07,08 bash scripts/run_all_pectra.sh     # just scratch notebooks
#   KEEP_GOING=1 bash scripts/run_all_pectra.sh   # don't fail-fast on a single notebook
#
# Recommended: run inside tmux so it survives disconnects.
#   tmux new -s train
#   bash scripts/run_all_pectra.sh
#   # detach: Ctrl+b then d   |   reattach: tmux attach -t train

set -u  # treat unset vars as errors
[[ "${KEEP_GOING:-0}" == "1" ]] || set -e   # fail-fast unless KEEP_GOING=1

PROJECT_DIR="${PROJECT_DIR:-$HOME/Project2}"

# Make uv visible even if .bashrc isn't sourced
export PATH="$HOME/.local/bin:$PATH"

cd "$PROJECT_DIR"

# Notebook order (LoRA + full FT first, then scratch). Adjust ONLY=06,07 etc. to subset.
ALL_NOTEBOOKS=(
    "06_finetune_lora"
    "05_finetune_full"
    "07_scratch_vit"
    "08_scratch_cnn"
)

if [[ -n "${ONLY:-}" ]]; then
    IFS=',' read -ra ONLY_ARR <<< "$ONLY"
    NOTEBOOKS=()
    for prefix in "${ONLY_ARR[@]}"; do
        for nb in "${ALL_NOTEBOOKS[@]}"; do
            [[ "$nb" == "$prefix"* ]] && NOTEBOOKS+=("$nb")
        done
    done
else
    NOTEBOOKS=("${ALL_NOTEBOOKS[@]}")
fi

START_TS=$(date +%s)
echo "================================================================"
echo "  run_all_pectra.sh"
echo "  PROJECT_DIR    = $PROJECT_DIR"
echo "  NOTEBOOKS      = ${NOTEBOOKS[*]}"
echo "  EPOCH_OVERRIDE = ${EPOCH_OVERRIDE:-(notebook default)}"
echo "  KEEP_GOING     = ${KEEP_GOING:-0}"
echo "  START          = $(date -Iseconds)"
echo "================================================================"

# Sanity checks
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not on PATH. Source ~/.bashrc or install uv." >&2
    exit 1
fi
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('cuda OK:', torch.cuda.get_device_name(0))"
uv run wandb login --verify >/dev/null 2>&1 || {
    echo "WARNING: wandb not logged in. Run 'uv run wandb login' first or set WANDB_API_KEY." >&2
    [[ "${KEEP_GOING:-0}" == "1" ]] || exit 1
}

declare -A STATUS
declare -A DURATION

for stem in "${NOTEBOOKS[@]}"; do
    nb="notebooks/${stem}.ipynb"
    log="notebooks/${stem}_run.log"

    if [[ ! -f "$nb" ]]; then
        echo "WARN: $nb not found, skipping" >&2
        STATUS[$stem]="MISSING"
        continue
    fi

    nb_start=$(date +%s)
    echo
    echo "================================================================"
    echo "  >>> $nb"
    echo "  log: $log"
    echo "  $(date -Iseconds)"
    echo "================================================================"

    # nbconvert buffers stdout until each cell finishes. Notebooks 07/08
    # only print one line per epoch (and use wandb for live metrics) so this
    # is fine — terminal stays quiet during a cell, then dumps the per-epoch
    # log when the cell completes. Watch wandb for real-time progress.
    if uv run jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=-1 \
            "$nb" 2>&1 | tee "$log"; then
        STATUS[$stem]="OK"
    else
        STATUS[$stem]="FAILED"
        echo "ERROR: $nb failed (see $log for details)" >&2
        if [[ "${KEEP_GOING:-0}" != "1" ]]; then
            echo "Aborting (set KEEP_GOING=1 to continue past failures)." >&2
            break
        fi
    fi
    DURATION[$stem]=$(( $(date +%s) - nb_start ))
done

# Final consolidated comparison: re-run NB 06 cells 16+17 only? Simpler: just
# run the bidirectional + comparison block on whatever results exist by
# executing the lightweight no-train post-hoc helper. We keep it simple here
# and just print a summary of results json files actually produced.

END_TS=$(date +%s)
TOTAL=$(( END_TS - START_TS ))
HRS=$(( TOTAL / 3600 ))
MINS=$(( (TOTAL % 3600) / 60 ))

echo
echo "================================================================"
echo "  SUMMARY  (total: ${HRS}h ${MINS}m)"
echo "================================================================"
for stem in "${NOTEBOOKS[@]}"; do
    s="${STATUS[$stem]:-PENDING}"
    d="${DURATION[$stem]:-0}"
    dh=$((d / 3600))
    dm=$(( (d % 3600) / 60 ))
    printf "  %-25s %-8s  %dh %dm\n" "$stem" "$s" "$dh" "$dm"
done

echo
echo "  Result JSONs:"
ls -1 data/results/*.json 2>/dev/null | sed 's/^/    /' || echo "    (none)"
echo
echo "  Model dirs:"
ls -1d data/models/*/ 2>/dev/null | sed 's/^/    /' || echo "    (none)"
echo
echo "  Cached test embeddings:"
ls -1 data/embeddings/*_test.npy 2>/dev/null | sed 's/^/    /' || echo "    (none)"
echo
echo "  WandB project: https://wandb.ai/vysnyandrej-slovak-university-of-technology-in-bratislava/nsiete-flickr30k-clip"
echo "================================================================"

# Exit code: 0 if all OK, else 1
ALL_OK=1
for stem in "${NOTEBOOKS[@]}"; do
    [[ "${STATUS[$stem]:-}" == "OK" ]] || ALL_OK=0
done
[[ $ALL_OK -eq 1 ]] && exit 0 || exit 1
