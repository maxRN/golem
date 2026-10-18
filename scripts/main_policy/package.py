import json
import os
import shutil
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

root = Path('/home/ubuntu/work/golem_main_20261018_fast')
records = json.loads((root / 'candidates.json').read_text())
selection = json.loads((root / 'selection.json').read_text())
chosen = sorted(r['candidate'] for r in selection['selected'])
assert len(chosen) >= 150
out = root / 'dataset'
(out / 'meta').mkdir(parents=True, exist_ok=True)
source = Path(records[0]['source'])
info = json.loads((source / 'meta/info.json').read_text())
episodes = []
manifest = []
offset = 0
for new_id, candidate in enumerate(chosen):
    r = records[candidate]
    old_id = r['source_episode']
    table = pq.read_table(Path(r['source']) / f'data/chunk-000/episode_{old_id:06d}.parquet')
    length = len(table)
    assert length == r['length']
    for name, values in [('episode_index', [new_id] * length), ('index', list(range(offset, offset + length))), ('task_index', [0] * length)]:
        field = table.schema.field(name)
        table = table.set_column(table.schema.get_field_index(name), field, pa.array(values, type=field.type))
    target = out / f'data/chunk-000/episode_{new_id:06d}.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)
    for camera in ('front', 'wrist'):
        path = root / f'augmented/{candidate:03d}_{camera}.mp4' if r['augmented'] else Path(r['source']) / f'videos/chunk-000/observation.images.{camera}/episode_{old_id:06d}.mp4'
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames,r_frame_rate,width,height', '-of', 'json', str(path)]))['streams'][0]
        assert int(probe['nb_frames']) == length, (path, probe, length)
        assert probe['r_frame_rate'] == '30/1' and (probe['width'], probe['height']) == (640, 480), probe
        target = out / f'videos/chunk-000/observation.images.{camera}/episode_{new_id:06d}.mp4'
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
    episodes.append({'episode_index': new_id, 'tasks': ['pick up the cube and place it on the green pad'], 'length': length})
    manifest.append(dict(r, dataset_episode=new_id, score=next(x['score'] for x in selection['selected'] if x['candidate'] == candidate)))
    offset += length
info.update(total_episodes=len(chosen), total_frames=offset, total_videos=2 * len(chosen), total_chunks=1, total_tasks=1, splits={'train': f'0:{len(chosen)}'})
(out / 'meta/info.json').write_text(json.dumps(info, indent=2))
(out / 'meta/tasks.jsonl').write_text(json.dumps({'task_index': 0, 'task': 'pick up the cube and place it on the green pad'}) + '\n')
(out / 'meta/episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in episodes))
result = {'raw_sim_episodes': sum(not r['augmented'] for r in manifest), 'augmented_episodes': sum(r['augmented'] for r in manifest), 'total_episodes': len(chosen), 'total_frames': offset, 'candidate_original_runs': 100, 'candidate_augmented_runs': 100, 'augmentation': json.loads((root / 'augmentation_done.json').read_text()), 'curation': selection['method'], 'episodes': manifest}
(root / 'manifest.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}), flush=True)
