import concurrent.futures
import json
import subprocess
from pathlib import Path

root = Path('/home/ubuntu/work/golem_main_20261018_fast')
base = Path('/home/ubuntu/work/so101_dr/datasets/local')
root.mkdir(exist_ok=True)
records = []
for name in ('so101_sim_dr', 'so101_sim_dr_cam2'):
    source = base / name
    episodes = [json.loads(x) for x in (source / 'meta/episodes.jsonl').read_text().splitlines()][:50]
    assert len(episodes) == 50
    for episode in episodes:
        index = len(records)
        records.append(dict(candidate=index, source=str(source), source_episode=episode['episode_index'], length=episode['length'], augmented=False))
records += [dict(r, candidate=r['candidate'] + 100, augmented=True) for r in records.copy()]
(root / 'candidates.json').write_text(json.dumps(records, indent=2))

def generate(record):
    if not record['augmented']:
        return
    i = record['candidate']
    for camera in ('front', 'wrist'):
        source = Path(record['source']) / f"videos/chunk-000/observation.images.{camera}/episode_{record['source_episode']:06d}.mp4"
        target = root / f'augmented/{i:03d}_{camera}.mp4'
        target.parent.mkdir(exist_ok=True)
        brightness = [-0.10, -0.05, 0.05, 0.10][i % 4]
        colors = ['1 0 -1 0', '-1 1 0 0', '0 -1 1 0', '1 -1 0 0']
        filters = f"selectivecolor=blues='{colors[i % 4]}',eq=brightness={brightness}:gamma={0.85 + (i % 7) * 0.05}"
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-threads', '1', '-i', str(source), '-vf', filters, '-filter_threads', '1', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-threads', '1', '-an', str(target)], check=True)
    print('generated', i, flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    list(pool.map(generate, records))
(root / 'augmentation_done.json').write_text(json.dumps({'augmented_runs': 100, 'generated_camera_videos': 200, 'method': 'FFmpeg selective blue color transform and brightness/gamma variation; original timing and actions preserved; not Cosmos Transfer'}))
