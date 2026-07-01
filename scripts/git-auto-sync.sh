#!/usr/bin/env bash
# Background auto-sync: polls for local changes, commits, and pushes to GitHub.
# Usage:
#   ./scripts/git-auto-sync.sh start   # start daemon (default: every 60s)
#   ./scripts/git-auto-sync.sh stop    # stop daemon
#   ./scripts/git-auto-sync.sh status  # show whether daemon is running
#   ./scripts/git-auto-sync.sh once    # sync immediately
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$ROOT/.auto-sync.pid"
LOG_FILE="$ROOT/.auto-sync.log"
INTERVAL="${GIT_AUTO_SYNC_INTERVAL:-60}"

daemon_loop() {
  cd "$ROOT"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] auto-sync daemon started (interval=${INTERVAL}s)" >>"$LOG_FILE"
  while true; do
    if "$ROOT/scripts/git-sync-once.sh" >>"$LOG_FILE" 2>&1; then
      :
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] sync failed (see log)" >>"$LOG_FILE"
    fi
    sleep "$INTERVAL"
  done
}

start_daemon() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "auto-sync already running (pid $(cat "$PID_FILE"))"
    exit 0
  fi
  # Export paths/interval into the detached shell; declare -f alone does not
  # carry parent variables, which previously left INTERVAL empty and killed the loop.
  nohup env ROOT="$ROOT" LOG_FILE="$LOG_FILE" INTERVAL="$INTERVAL" bash -c '
    daemon_loop() {
      cd "$ROOT"
      echo "[$(date "+%Y-%m-%d %H:%M:%S")] auto-sync daemon started (interval=${INTERVAL}s)" >>"$LOG_FILE"
      while true; do
        if "$ROOT/scripts/git-sync-once.sh" >>"$LOG_FILE" 2>&1; then
          :
        else
          echo "[$(date "+%Y-%m-%d %H:%M:%S")] sync failed (see log)" >>"$LOG_FILE"
        fi
        sleep "$INTERVAL"
      done
    }
    daemon_loop
  ' >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "auto-sync started (pid $(cat "$PID_FILE"), interval=${INTERVAL}s, log=$LOG_FILE)"
}

stop_daemon() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "auto-sync not running"
    exit 0
  fi
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    echo "auto-sync stopped (pid $pid)"
  else
    echo "stale pid file removed"
  fi
  rm -f "$PID_FILE"
}

status_daemon() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "auto-sync running (pid $(cat "$PID_FILE"))"
  else
    echo "auto-sync not running"
    [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"
    exit 1
  fi
}

case "${1:-start}" in
  start) start_daemon ;;
  stop) stop_daemon ;;
  status) status_daemon ;;
  once) "$ROOT/scripts/git-sync-once.sh" ;;
  *) echo "usage: $0 {start|stop|status|once}"; exit 1 ;;
esac
