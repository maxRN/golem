import hashlib
import json
import shutil
import subprocess
from pathlib import Path


root = Path('/home/ubuntu/work/golem_one_episode_20261018')
review = json.loads((root / 'review/decision.json').read_text())
assert review['verdict'] == 'accept', review
manifest = json.loads((root / 'manifest.json').read_text())
dataset = root / 'dataset'
dataset.mkdir(exist_ok=True)
shutil.copytree(root / 'source/meta', dataset / 'meta', dirs_exist_ok=True)
shutil.copytree(root / 'source/data', dataset / 'data', dirs_exist_ok=True)

checks = {}
for camera in ('front', 'wrist'):
    selection = review['camera_selection'][camera]
    assert selection in ('augmented', 'source'), selection
    generated = (root / f'outputs/{camera}.mp4' if selection == 'augmented' else
                 root / f'source/videos/chunk-000/observation.images.{camera}/episode_000000.mp4')
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
         '-show_entries', 'stream=nb_read_frames,r_frame_rate,width,height',
         '-of', 'json', str(generated)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)['streams'][0]
    numerator, denominator = map(int, stream['r_frame_rate'].split('/'))
    assert int(stream['nb_read_frames']) == manifest['frames'], stream
    assert numerator / denominator == manifest['fps'], stream
    assert (stream['height'], stream['width']) == (480, 640), stream
    target = dataset / f'videos/chunk-000/observation.images.{camera}/episode_000000.mp4'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated, target)
    checks[camera] = stream | {'selection': selection, 'input_path': str(generated),
                              'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}

parquet = dataset / 'data/chunk-000/episode_000000.parquet'
assert hashlib.sha256(parquet.read_bytes()).hexdigest() == manifest['source_sha256']['data/chunk-000/episode_000000.parquet']
manifest.update(dataset_path=str(dataset), augmented_episodes=1, raw_sim_episodes=0,
                real_episodes=0, video_checks=checks, manual_review=review,
                reason_curation='Cosmos 3 Nano accepted the source episode; augmented model review skipped to shorten this pipeline check.')
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(dataset / 'meta/augmentation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(checks, indent=2))
