import hashlib
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq


root = Path('/home/ubuntu/work/golem_one_episode_20261018')
source = Path('/home/ubuntu/work/so101_dr/datasets/local/so101_sim_dr_cam2')
snapshot = root / 'source'
for folder in ('source/meta', 'source/data/chunk-000', 'specs', 'outputs', 'review'):
    (root / folder).mkdir(parents=True, exist_ok=True)

info = json.loads((source / 'meta/info.json').read_text())
episode = json.loads((source / 'meta/episodes.jsonl').read_text().splitlines()[0])
parquet = source / 'data/chunk-000/episode_000000.parquet'
df = pq.read_table(parquet).to_pandas()
assert episode['episode_index'] == 0
assert len(df) == episode['length']
assert df['episode_index'].eq(0).all()
assert df['frame_index'].tolist() == list(range(len(df)))
shutil.copy2(parquet, snapshot / 'data/chunk-000' / parquet.name)
shutil.copy2(source / 'meta/tasks.jsonl', snapshot / 'meta/tasks.jsonl')
(snapshot / 'meta/episodes.jsonl').write_text(json.dumps(episode) + '\n')
info.update(total_episodes=1, total_frames=len(df), total_videos=2, total_chunks=1, splits={'train': '0:1'})
(snapshot / 'meta/info.json').write_text(json.dumps(info, indent=2) + '\n')

hashes = {}
for camera in ('front', 'wrist'):
    relative = Path(f'videos/chunk-000/observation.images.{camera}/episode_000000.mp4')
    target = snapshot / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / relative, target)
    hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    viewpoint = 'Fixed front camera' if camera == 'front' else 'Robot wrist-mounted camera'
    spec = {
        'name': camera,
        'video_path': str(target),
        'prompt': (
            f'{viewpoint} view of a compact yellow SO-101 robot arm picking up one blue cube '
            'and placing it on a fixed green square pad on a laboratory tabletop. '
            'Photorealistic matte gray tabletop, realistic plastic and metal robot materials, '
            'neutral diffuse indoor lighting and soft shadows. Preserve the original camera '
            'viewpoint and framing, the blue cube size and shape, green pad, table markings, '
            'robot geometry, object positions, all robot and object motion, contact events '
            'and timing exactly as in the input video. Consistent appearance throughout.'
        ),
        'num_steps': 12,
        'guidance': 3,
        'seed': 20261018,
        'resolution': '480',
        'keep_input_resolution': True,
        'edge': {'control_weight': 1.0},
    }
    (root / f'specs/{camera}.json').write_text(json.dumps(spec, indent=2) + '\n')

hashes['data/chunk-000/episode_000000.parquet'] = hashlib.sha256(parquet.read_bytes()).hexdigest()
manifest = {
    'source_dataset': str(source),
    'source_episode_index': 0,
    'source_snapshot': str(snapshot),
    'frames': len(df),
    'fps': info['fps'],
    'duration_seconds': len(df) / info['fps'],
    'grid_spot': 1,
    'task': episode['tasks'][0],
    'source_sha256': hashes,
    'transfer_model': 'nvidia/Cosmos-Transfer2.5-2B',
    'transfer_variant': 'edge',
    'frame_mapping': 'Generated frame i corresponds to source frame i; no temporal resampling.',
    'reason_curation': 'Skipped at user request; manual review required.',
}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
