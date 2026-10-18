import json
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

torch.set_num_threads(4)

root = Path('/home/ubuntu/work/golem_main_20261018_fast')
records = json.loads((root / 'candidates.json').read_text())
review = root / 'review'
review.mkdir(exist_ok=True)

def video_path(r, camera):
    if r['augmented']:
        return root / f"augmented/{r['candidate']:03d}_{camera}.mp4"
    return Path(r['source']) / f"videos/chunk-000/observation.images.{camera}/episode_{r['source_episode']:06d}.mp4"

def samples(path):
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        wanted = {int((stream.frames - 1) * f) for f in (0.05, 0.4, 0.7, 0.98)}
        return [frame.to_image().resize((192, 144)) for i, frame in enumerate(c.decode(video=0)) if i in wanted]

for start in range(0, 200, 10):
    if (review / f'batch_{start:03d}.jpg').exists():
        continue
    sheet = Image.new('RGB', (820, 1480), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, r in enumerate(records[start:start + 10]):
        draw.text((0, row * 148 + 60), str(r['candidate']), fill='black')
        for col, frame in enumerate(samples(video_path(r, 'front'))):
            sheet.paste(frame, (50 + col * 192, row * 148))
    sheet.save(review / f'batch_{start:03d}.jpg')
print('sheets_ready', flush=True)
model_id = 'nvidia/Cosmos3-Nano'
revision = 'e59a53c25979a090fa8706c9acc0c254a6e89b92'
processor = AutoProcessor.from_pretrained(model_id, revision=revision, local_files_only=True)
model = Cosmos3OmniForConditionalGeneration.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa', local_files_only=True)
model.eval()
processor.tokenizer.padding_side = 'left'
labels = ['G', 'B', 'U']
label_ids = [processor.tokenizer.encode(label, add_special_tokens=False) for label in labels]
assert all(len(ids) == 1 for ids in label_ids), label_ids
label_ids = [ids[0] for ids in label_ids]
results = []
for start in range(0, 200, 20):
    conversations = []
    for candidate in range(start, start + 20):
        sheet = Image.open(review / f'batch_{candidate // 10 * 10:03d}.jpg')
        row = candidate % 10
        path = review / f'episode_{candidate:03d}.jpg'
        sheet.crop((50, row * 148, 818, row * 148 + 144)).save(path)
        prompt = 'Four chronological frames show one robot episode. Task: pick up the cube and place it on the green pad. Cube color and lighting changes are intentional. Judge visible completion and contact. Answer with one letter: G for good, B for obvious failure, U for uncertain.'
        conversations.append([{'role': 'user', 'content': [{'type': 'image', 'path': str(path)}, {'type': 'text', 'text': prompt}]}])
    inputs = processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt', padding=True, processor_kwargs={'max_pixels': 128 * 1024}).to(model.device, torch.bfloat16)
    with torch.inference_mode():
        output = model(**inputs, logits_to_keep=1)
    probabilities = output.logits[:, -1, label_ids].float().softmax(-1).cpu().tolist()
    for candidate, probs in zip(range(start, start + 20), probabilities):
        label = labels[max(range(3), key=lambda i: probs[i])]
        results.append({'candidate': candidate, 'score': 10 * probs[0] + 5 * probs[2], 'label': label, 'conditional_probabilities': dict(zip(labels, probs))})
    del output, inputs
    (review / 'scores.json').write_text(json.dumps(results, indent=2))
    print('reviewed', start + 20, flush=True)
assert len(results) == 200
selected = sorted(results, key=lambda r: (-r['score'], r['candidate']))[:150]
(root / 'selection.json').write_text(json.dumps({'model': model_id, 'revision': revision, 'method': 'Cosmos forced-choice next-token likelihood ranking (G good, B bad, U uncertain), four chronological front-camera frames per episode; retain top 150 to satisfy minimum; low-ranked excluded, retained is not a success certification', 'selected': selected, 'excluded': [r for r in results if r not in selected]}, indent=2))
