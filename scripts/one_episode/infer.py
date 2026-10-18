import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy


root = Path('/home/ubuntu/work/golem_one_episode_20261018')
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, default=root / 'policy/checkpoint-2')
parser.add_argument('--output-dir', type=Path, default=root)
args = parser.parse_args()
checkpoint = args.checkpoint
torch.manual_seed(20261018)
policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                     model_path=str(checkpoint), device='cuda:0')
configs = policy.processor.get_modality_configs()[EmbodimentTag.NEW_EMBODIMENT.value]
loader = LeRobotEpisodeLoader(dataset_path=str(root / 'dataset'), modality_configs=configs)
trajectory = loader[0]
observation_configs = deepcopy(configs)
observation_configs.pop('action')
frame_index = len(trajectory) // 2
step = extract_step_data(trajectory, frame_index, observation_configs, EmbodimentTag.NEW_EMBODIMENT)
flat = {f'state.{key}': value for key, value in step.states.items()}
flat.update({f'video.{key}': np.array(value) for key, value in step.images.items()})
flat.update({key: step.text for key in configs['language'].modality_keys})
observation = parse_observation_gr00t(flat, configs)
started = time.monotonic()
actions, _ = policy.get_action(observation)
elapsed = time.monotonic() - started
expected = {'single_arm': (1, 16, 5), 'gripper': (1, 16, 1)}
assert set(actions) == set(expected), actions.keys()
for key, shape in expected.items():
    assert actions[key].shape == shape, (key, actions[key].shape)
    assert np.isfinite(actions[key]).all(), key
result = {
    'checkpoint': str(checkpoint),
    'dataset': str(root / 'dataset'),
    'episode_index': 0,
    'frame_index': frame_index,
    'timestamp_seconds': frame_index / 30,
    'task': step.text,
    'action_shapes': {key: list(value.shape) for key, value in actions.items()},
    'all_actions_finite': True,
    'action_units': 'LeRobot SO-101 normalized position units; arm [-100, 100], gripper [0, 100].',
    'inference_seconds': elapsed,
    'actions': {key: value.tolist() for key, value in actions.items()},
    'real_robot_evaluated': False,
}
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / 'inference.json').write_text(json.dumps(result, indent=2) + '\n')
np.savez(args.output_dir / 'inference_actions.npz', **actions)
print(json.dumps({key: value for key, value in result.items() if key != 'actions'}, indent=2))
