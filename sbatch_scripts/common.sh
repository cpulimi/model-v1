#!/usr/bin/env bash
# Resolve repo root from sbatch_scripts/ no matter where the repo was cloned.
# Usage (inside any script in sbatch_scripts/):
#   source "$(dirname "$0")/common.sh"          # sbatch job scripts
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"  # launch scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
