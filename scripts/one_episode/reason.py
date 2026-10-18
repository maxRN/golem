import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration


root = Path('/home/ubuntu/work/golem_one_episode_20261018')
model_id = 'nvidia/Cosmos3-Nano'
revision = 'e59a53c25979a090fa8706c9acc0c254a6e89b92'
processor = AutoProcessor.from_pretrained(model_id, revision=revision)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    model_id, revision=revision, dtype=torch.bfloat16, device_map='cuda:0',
    attn_implementation='sdpa',
)
model.eval()
prompt = (
    'These two synchronized videos show the front and wrist views of the same robot episode. '
    'The task is to pick up the blue cube and release it at rest on the green pad. '
    'Evaluate this episode for imitation-learning data curation. Assess task completion, '
    'motion continuity, object permanence, object shape stability, and gripper-object contact. '
    'Use visible evidence only. Occlusion by the gripper is not itself an anomaly. '
    'Do not reject solely for simulation-like textures or mild appearance differences. '
    'If evidence is insufficient, mark review rather than inventing an issue. '
    'Return a JSON object only, with keys verdict (accept, reject, or review), '
    'task_success (true, false, or null), issues (a list of objects with timestamp_seconds, '
    'description, and severity), and summary (a brief explanation).'
)
(root / 'review/reason_prompt.txt').write_text(prompt + '\n')
for name in sys.argv[1:] or ('source', 'augmented'):
    paths = [root / f'source/videos/chunk-000/observation.images.{camera}/episode_000000.mp4'
             if name == 'source' else root / f'outputs/{camera}.mp4'
             for camera in ('front', 'wrist')]
    content = [{'type': 'video', 'path': str(path)} for path in paths]
    content.append({'type': 'text', 'text': prompt})
    inputs = processor.apply_chat_template(
        [{'role': 'user', 'content': content}], fps=2, max_pixels=256 * 256,
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt',
    ).to(model.device, torch.bfloat16)
    started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=1536)
    text = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    (root / f'review/{name}_reason_raw.txt').write_text(text + '\n')
    answer = text.split('</think>')[-1]
    decision = json.loads(answer[answer.index('{'):answer.rindex('}') + 1])
    assert decision['verdict'] in ('accept', 'reject', 'review'), decision
    result = {'model': model_id, 'revision': revision, 'video_paths': list(map(str, paths)),
              'sampling_fps': 2, 'elapsed_seconds': time.monotonic() - started,
              'decision': decision}
    (root / f'review/{name}_reason.json').write_text(json.dumps(result, indent=2) + '\n')
    print(name, json.dumps(result), flush=True)
