#!/usr/bin/env bash
# Sync source to the same absolute path on the named TPU slice workers.
# Run from worker zero. Set up SSH aliases and install requirements-tpu.txt
# plus this package on each worker first. Copy datasets separately.
set -euo pipefail
ALGEBRAIC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ALGEBRAIC_HOSTS=("$@")
if [ ${#ALGEBRAIC_HOSTS[@]} -eq 0 ]; then ALGEBRAIC_HOSTS=(w0 w1 w2 w3); fi
for worker in "${ALGEBRAIC_HOSTS[@]}"; do
  ssh "$worker" "mkdir -p '$ALGEBRAIC_ROOT'"
  # Worker zero already has the authoritative checkout.
  if [ "$(ssh "$worker" hostname)" = "$(hostname)" ]; then continue; fi
  rsync -a --delete --exclude .git --exclude .lake --exclude runs --exclude /data \
    --exclude .venv --exclude '*.egg-info' --exclude '__pycache__' --exclude .pytest_cache \
    "$ALGEBRAIC_ROOT/" "$worker:$ALGEBRAIC_ROOT/"
done
printf 'Workers synced.\n'
