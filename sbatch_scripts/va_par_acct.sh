#!/bin/bash
# VA parallel-batch ACCOUNTING. Runs LAST, afterany on the solve array and the
# merge job, and collects the memory numbers that cannot be collected from
# inside a running job.
#
# WHY THIS JOB EXISTS. seff and `sacct MaxRSS` read slurmdbd, which is not
# written until a step ENDS. va_slurm_mem() runs inside the solve/merge jobs, so
# every seff it captured said "State: RUNNING / Memory Utilized: 0.00 MB" and
# every MaxRSS column came out blank -- the 10- and 20-hub runs both lost their
# node-level memory this way, and the accounting rows age out of slurmdbd, so
# that loss is permanent. This job runs after those steps have ended, waits for
# the flush, and writes the real numbers.
#
# afterany, NOT afterok: a job that died on OOM or walltime is precisely the one
# whose memory figure matters most. An afterok dependency would skip it.
#
# NO VECTOR ENGINE, no python, no conda -- this is sacct and awk. It runs on a
# normal partition so it never queues behind the single VE node.
#SBATCH -p general
#SBATCH -q private
#SBATCH -c 1
#SBATCH --mem 2G
#SBATCH -t 0-00:20:00
#SBATCH -J va_par_acct
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail

PROJECT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT"
mkdir -p logs

RUN_NAME="${VA_RUN_NAME:-va_par}"
OUTDIR="${VA_OUTDIR:-results/va_parallel}"
RUN_DIR="$OUTDIR/$RUN_NAME"
TSV="$RUN_DIR/slurm_mem_va.tsv"
mkdir -p "$RUN_DIR"

# Job ids to account for, exported by va_par_launch.sh. Space separated; an
# array id expands to all of its tasks below.
JOBS="${VA_ACCT_JOBS:-}"
if [[ -z "$JOBS" ]]; then
  echo ">>> ABORT: VA_ACCT_JOBS is empty. Submit through va_par_launch.sh, or" >&2
  echo ">>>        rerun by hand:  VA_ACCT_JOBS='<solve_id> <merge_id>' \\" >&2
  echo ">>>                        VA_RUN_NAME=$RUN_NAME VA_OUTDIR=$OUTDIR \\" >&2
  echo ">>>                        sbatch sbatch_scripts/va_par_acct.sh" >&2
  exit 1
fi

echo ">>> run dir   $RUN_DIR"
echo ">>> accounting for jobs: $JOBS"

# slurmdbd writes on step completion but the batch step of a job is only
# finalised once the whole job is gone, and this job started the moment the
# dependency cleared. Give the flush a moment; then poll rather than guess.
sleep 30

if [[ ! -s "$TSV" ]]; then
  printf 'JobID\tLabel\tSource\tCgroupPeakMB\tMaxRSS\tMaxVMSize\tReqMem\tElapsed\tState\n' > "$TSV"
fi

# Expand an array job id into its individual task ids (61921976 -> 61921976_1
# ...). A non-array id yields just itself. -X gives one row per job, no steps.
expand_jobs() {
  local id rows out=""
  for id in $JOBS; do
    rows=$(sacct -j "$id" -X --format=JobID --noheader --parsable2 2>/dev/null \
           | awk 'NF' | grep -v '\.batch$' || true)
    if [[ -n "$rows" ]]; then out+=" $rows"; else out+=" $id"; fi
  done
  printf '%s' "$out"
}

ALL_JOBS=$(expand_jobs)
echo ">>> expanded to:$ALL_JOBS"

# Poll for MaxRSS to appear. On a busy slurmdbd the flush can lag well past the
# 30s above; 12 x 15s = 3 minutes is comfortably inside the 20 minute walltime.
for attempt in $(seq 1 12); do
  PENDING=0
  for jid in $ALL_JOBS; do
    RSS=$(sacct -j "$jid" --format=MaxRSS --units=M --noheader --parsable2 2>/dev/null \
          | tr -d ' ' | awk 'NF' | head -n1 || true)
    [[ -z "$RSS" ]] && PENDING=$((PENDING + 1))
  done
  if [[ "$PENDING" -eq 0 ]]; then
    echo ">>> accounting flushed after attempt $attempt"
    break
  fi
  echo ">>> attempt $attempt: $PENDING job(s) still without MaxRSS, waiting 15s ..."
  sleep 15
done

for jid in $ALL_JOBS; do
  # Label from the job name so the row is readable without cross-referencing:
  # va_par_solve task 3 -> batch3, va_par_merge -> merge.
  NAME=$(sacct -j "$jid" -X --format=JobName --noheader --parsable2 2>/dev/null | head -n1 || true)
  case "$NAME" in
    *solve*) LABEL="batch${jid##*_}" ;;
    *merge*) LABEL="merge" ;;
    *)       LABEL="${NAME:-job}" ;;
  esac

  if command -v seff >/dev/null 2>&1; then
    seff "$jid" > "$RUN_DIR/seff_${LABEL}_${jid}.txt" 2>&1 || true
  fi

  # Widest MaxRSS across the job's steps: the batch step holds the python
  # process, but a srun step would hold it instead, so take the max rather than
  # assuming which one. sort -h understands the K/M/G suffixes sacct emits.
  LINE=$(sacct -j "$jid" \
    --format=JobID,JobName,MaxRSS,MaxVMSize,ReqMem,Elapsed,State \
    --units=M --noheader --parsable2 2>/dev/null \
    | awk -F'|' '$3 != ""' | sort -t'|' -k3 -h | tail -n1 || true)
  MAXRSS=$(printf '%s' "$LINE" | awk -F'|' '{print $3}')
  MAXVM=$(printf  '%s' "$LINE" | awk -F'|' '{print $4}')
  REQMEM=$(printf '%s' "$LINE" | awk -F'|' '{print $5}')
  ELAPSED=$(printf '%s' "$LINE" | awk -F'|' '{print $6}')
  # State from the -X job row, not the step row: a step can read COMPLETED on a
  # job the scheduler killed.
  STATE=$(sacct -j "$jid" -X --format=State --noheader --parsable2 2>/dev/null | head -n1 || true)

  # CgroupPeakMB stays empty here by design -- the cgroup is gone once the step
  # ends. The in_job row already carries it; the two rows are complementary.
  printf '%s\t%s\tpost_hoc\t\t%s\t%s\t%s\t%s\t%s\n' \
    "$jid" "$LABEL" "$MAXRSS" "$MAXVM" "$REQMEM" "$ELAPSED" "$STATE" >> "$TSV"
  echo ">>> $LABEL ($jid): MaxRSS=${MAXRSS:-<none>} MaxVMSize=${MAXVM:-<none>}" \
       "ReqMem=${REQMEM:-<none>} Elapsed=${ELAPSED:-<none>} State=${STATE:-<none>}"

  if [[ -z "$MAXRSS" ]]; then
    echo ">>>   WARNING: still no MaxRSS for $jid. This cluster's jobacct_gather" >&2
    echo ">>>            plugin may not sample memory at all -- in that case the" >&2
    echo ">>>            in_job CgroupPeakMB column is the only node-level number" >&2
    echo ">>>            available, and it is the one to report." >&2
  fi
done

echo ""
echo ">>> $TSV"
column -t -s "$(printf '\t')" "$TSV" 2>/dev/null || cat "$TSV"
