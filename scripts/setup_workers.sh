#!/bin/bash
# Sync code and helper scripts from worker 0 to workers 1-3 of the slice.
# Assumes ~/.ssh/config defines hosts w1..w3 and ~/venv exists on each host
# (python3 -m virtualenv ~/venv; pip install -r requirements-tpu.txt).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p ~/bin
cat > ~/bin/chiprun <<'EOS'
#!/bin/bash
# Usage: chiprun <chip 0-3> <python args...>: one single-chip JAX job on this host.
chip=$1; shift
export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=$chip
export TPU_PROCESS_ADDRESSES=localhost:$((8476+chip)) TPU_PROCESS_PORT=$((8476+chip)) CLOUD_TPU_TASK_ID=0
export JAX_PLATFORMS=tpu
cd ~/sota-algebra && exec ~/venv/bin/python "$@"
EOS
cat > ~/bin/hostrun <<'EOS'
#!/bin/bash
# Usage: hostrun <python args...>: one JAX job on this host's four chips only.
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0,1,2,3
export TPU_PROCESS_ADDRESSES=localhost:8476 TPU_PROCESS_PORT=8476 CLOUD_TPU_TASK_ID=0
export JAX_PLATFORMS=tpu
cd ~/sota-algebra && exec ~/venv/bin/python "$@"
EOS
chmod +x ~/bin/chiprun ~/bin/hostrun
for w in w1 w2 w3; do
  ssh "$w" "mkdir -p ~/bin ~/sota-algebra"
  scp -q ~/bin/chiprun ~/bin/hostrun "$w":~/bin/
  rsync -a --delete --exclude .git --exclude runs --exclude data --exclude '__pycache__' --exclude prompt.md \
    "$ROOT"/ "$w":~/sota-algebra/
done
echo "workers synced"
