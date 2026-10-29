#!/bin/bash
# Clinica t1-linear on all IXI subjects, in resumable batches (Linux / WSL Ubuntu, CPU only).
#
#   bash scripts/t1_linear_wsl.sh
#
# Running the same command again resumes where it stopped: subjects whose cropped image already
# exists are skipped. No internet connection is needed during the run.
#
# Outputs (data/ is never committed):
#   data/IXI/caps_t1linear/      cropped 1 mm images (the uncropped ones are deleted to save disk)
#   data/IXI/caps_t1linear_2mm/  compact 2 mm float32 copy, the folder to upload to Drive
#
# Settings can be overridden with environment variables, e.g. NPROCS=4 bash scripts/t1_linear_wsl.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIDS="$REPO/data/IXI/bids"
CAPS="${CAPS:-$REPO/data/IXI/caps_t1linear}"
CAPS_2MM="${CAPS_2MM:-$REPO/data/IXI/caps_t1linear_2mm}"
WORK="${WORK:-$HOME/t1linear_run}"          # intermediate files, deleted after each batch
CLINICA_ENV="${CLINICA_ENV:-$HOME/clinica_env}"
NPROCS="${NPROCS:-6}"                        # images processed in parallel
THREADS="${THREADS:-2}"                      # threads per image (NPROCS x THREADS = CPU threads)
BATCH="${BATCH:-18}"                         # subjects per checkpoint (3 rounds of 6, ~10 min)
LIMIT="${LIMIT:-0}"                          # > 0: only the first LIMIT subjects (tests)

export MPLBACKEND=Agg ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$THREADS"
if [ ! -f "$BIDS/participants_sessions_age.tsv" ]; then
    echo "ERROR: $BIDS/participants_sessions_age.tsv not found (run scripts/ixi_to_bids.py first)"; exit 1
fi
if [ ! -x "$CLINICA_ENV/bin/clinica" ]; then
    echo "ERROR: Clinica not found in $CLINICA_ENV"; exit 1
fi
mkdir -p "$WORK"
# ANTsPy writes ~180 MB of registration temp files per subject and never deletes them:
# keep them in a private folder that is emptied after each batch
export TMPDIR="$WORK/tmp"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
LOG="$WORK/progress.log"
log() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }
crop_path() { echo "$CAPS/subjects/$1/$2/t1_linear/${1}_${2}_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"; }

# all (participant, session) pairs, from the age table written by ixi_to_bids.py
ALL="$WORK/all_subjects.tsv"
awk -F'\t' 'NR == 1 { for (i = 1; i <= NF; i++) { if ($i == "participant_id") p = i; if ($i == "session_id") s = i }; next }
            { print $p "\t" $s }' "$BIDS/participants_sessions_age.tsv" > "$ALL"
if [ "$LIMIT" -gt 0 ]; then head -n "$LIMIT" "$ALL" > "$ALL.tmp" && mv "$ALL.tmp" "$ALL"; fi
TOTAL=$(wc -l < "$ALL")
if [ "$TOTAL" -eq 0 ]; then echo "ERROR: no subject found in the age table"; exit 1; fi

pending() { while IFS=$'\t' read -r p s; do [ -f "$(crop_path "$p" "$s")" ] || printf "%s\t%s\n" "$p" "$s"; done < "$ALL"; }

START=$(date +%s)
LEFT=$(pending | wc -l)
DONE_AT_START=$(( TOTAL - LEFT ))
log "Start: $TOTAL subjects, $DONE_AT_START already done, $NPROCS images in parallel x $THREADS threads, checkpoint every $BATCH subjects"

free_gb() { df -P -BG "$1" | awk 'NR == 2 { gsub("G", "", $4); print $4 }'; }

while [ "$LEFT" -gt 0 ]; do
    for dir in "$WORK" "$CAPS_2MM/.."; do
        if [ "$(free_gb "$dir")" -lt "$MIN_FREE_GB" ]; then
            log "Less than $MIN_FREE_GB GB free on the disk of $dir: stopping before the next batch. Free some space, then relaunch."
            exit 1
        fi
    done
    pending | head -n "$BATCH" > "$WORK/batch_list.tsv"
    { printf "participant_id\tsession_id\n"; cat "$WORK/batch_list.tsv"; } > "$WORK/batch.tsv"
    log "Batch: $(wc -l < "$WORK/batch_list.tsv") subjects, from $(head -1 "$WORK/batch_list.tsv" | cut -f1)"

    rm -rf "$WORK/wd" "$TMPDIR" && mkdir -p "$TMPDIR"
    "$CLINICA_ENV/bin/clinica" run t1-linear "$BIDS" "$CAPS" -tsv "$WORK/batch.tsv" \
        --use-antspy --random_seed 42 -np "$NPROCS" -wd "$WORK/wd" >> "$WORK/clinica.log" 2>&1 \
        || log "  Clinica reported an error (details in $WORK/clinica.log): unfinished subjects will be retried"
    rm -rf "$WORK/wd" "$TMPDIR"

    # keep only the cropped image, which is the one used for training
    find "$CAPS/subjects" -name "*_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz" -delete 2>/dev/null
    "$CLINICA_ENV/bin/python" "$REPO/scripts/t1linear_to_2mm.py" "$CAPS" "$CAPS_2MM" \
        "$BIDS/participants_sessions_age.tsv" 2>&1 | tee -a "$LOG"

    PREV_LEFT=$LEFT
    LEFT=$(pending | wc -l)
    NOW=$(date +%s)
    NEW=$(( TOTAL - LEFT - DONE_AT_START ))
    if [ "$NEW" -gt 0 ]; then
        log "  Checkpoint: $(( TOTAL - LEFT )) / $TOTAL done | elapsed $(( (NOW - START) / 60 )) min | about $(( (NOW - START) * LEFT / NEW / 60 )) min left"
    fi
    if [ "$LEFT" -ge "$PREV_LEFT" ]; then
        log "No progress in this batch: stopping. Subjects still missing:"
        pending | tee -a "$LOG"
        exit 1
    fi
done

log "Finished: all $TOTAL subjects done in $(( ($(date +%s) - START) / 60 )) min."
log "Folder to upload to Drive: $CAPS_2MM ($(du -sh "$CAPS_2MM" | cut -f1))"
