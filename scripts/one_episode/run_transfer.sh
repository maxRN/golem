set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /home/ubuntu/work/cosmos-transfer2.5
exec .venv/bin/python -u examples/inference.py \
  -i /home/ubuntu/work/golem_one_episode_20261018/specs/front.json \
     /home/ubuntu/work/golem_one_episode_20261018/specs/wrist.json \
  -o /home/ubuntu/work/golem_one_episode_20261018/outputs \
  --model=edge --offload-guardrail-models --no-keep-going
