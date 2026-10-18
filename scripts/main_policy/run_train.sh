set -euo pipefail
cd /home/ubuntu/work/Isaac-GR00T
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1 USE_WANDB=0 TOKENIZERS_PARALLELISM=false
export MAX_STEPS=20 SAVE_STEPS=20 GLOBAL_BATCH_SIZE=1
export DATALOADER_NUM_WORKERS=0 SHARD_SIZE=64 NUM_SHARDS_PER_EPOCH=10 EPISODE_SAMPLING_RATE=1
exec bash examples/finetune.sh \
  --base-model-path /home/ubuntu/work/Isaac-GR00T/checkpoints/GR00T-N1.7-3B \
  --dataset-path /home/ubuntu/work/golem_main_20261018_fast/dataset \
  --modality-config-path /home/ubuntu/work/so101_sim_pickplace/gr00t/so101_sim_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /home/ubuntu/work/golem_main_20261018_fast/policy \
  --save-only-model \
  -- --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model
