set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
python3 -m venv /home/ubuntu/work/cosmos3_reason_env
uv pip install --python /home/ubuntu/work/cosmos3_reason_env/bin/python \
  --torch-backend=cu128 'torch==2.9.0' 'torchvision==0.24.0' \
  'transformers==5.20.1' 'safetensors==0.8.0' accelerate 'av==16.0.1' pillow
/home/ubuntu/work/cosmos3_reason_env/bin/python -c \
  'from transformers import Cosmos3OmniForConditionalGeneration; import transformers; print(transformers.__version__)'
/home/ubuntu/work/Isaac-GR00T/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    'nvidia/Cosmos3-Nano',
    revision='e59a53c25979a090fa8706c9acc0c254a6e89b92',
    allow_patterns=['*.json', '*.txt', '*.model', 'transformer/*.safetensors', 'vision_encoder/*.safetensors'],
    ignore_patterns=['assets/*', 'sound_tokenizer/*', 'vae/*', 'text_tokenizer/*'],
)
print(path)
PY
