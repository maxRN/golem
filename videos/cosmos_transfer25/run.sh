#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/ubuntu/work/cosmos-transfer2.5
export COSMOS_EXPERIMENTAL_CHECKPOINTS=1
exec .venv/bin/python -u examples/inference.py \
  -i /home/ubuntu/work/so101_sim_pickplace/logs/cosmos_transfer25/specs/warm_diffuse.json \
     /home/ubuntu/work/so101_sim_pickplace/logs/cosmos_transfer25/specs/cool_diffuse.json \
     /home/ubuntu/work/so101_sim_pickplace/logs/cosmos_transfer25/specs/soft_side_light.json \
  -o /home/ubuntu/work/so101_sim_pickplace/logs/cosmos_transfer25/outputs \
  --model=edge/distilled --offload-guardrail-models --no-keep-going
