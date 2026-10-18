set -euo pipefail
ROOT=/home/ubuntu/work/golem_main_20261018_fast
for attempt in $(seq 1 300); do
  if [ -f "$ROOT/selection.json" ]; then break; fi
  sleep 2
done
test -f "$ROOT/selection.json"
/home/ubuntu/work/Isaac-GR00T/.venv/bin/python -u "$ROOT/scripts/package.py"
bash /home/ubuntu/work/so101_sim_pickplace/prepare_dataset.sh "$ROOT/dataset"
bash "$ROOT/scripts/run_train.sh"
cd /home/ubuntu/work/Isaac-GR00T
.venv/bin/python -u "$ROOT/scripts/infer.py"
