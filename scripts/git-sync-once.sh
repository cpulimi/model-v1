#!/usr/bin/env bash
# Commit and push tracked changes (respects .gitignore). Safe to call repeatedly.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  exit 0
fi

if [[ -f "$ROOT/.auto-sync.lock" ]]; then
  lock_age=$(( $(date +%s) - $(stat -f %m "$ROOT/.auto-sync.lock" 2>/dev/null || echo 0) ))
  if (( lock_age < 45 )); then
    exit 0
  fi
fi

touch "$ROOT/.auto-sync.lock"
trap 'rm -f "$ROOT/.auto-sync.lock"' EXIT

if [[ -z "$(git status --porcelain)" ]]; then
  exit 0
fi

git add -A
if git diff --cached --quiet; then
  exit 0
fi

ts="$(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "Auto-sync: $ts"
git push origin HEAD
