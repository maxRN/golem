import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('destination', type=Path)
args = parser.parse_args()
source = args.source
destination = args.destination
index = json.loads((source / 'model.safetensors.index.json').read_text())
destination.mkdir(parents=True, exist_ok=False)
for path in source.iterdir():
    if path.suffix != '.safetensors':
        target = destination / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)

total_size = 0
for shard in sorted(set(index['weight_map'].values())):
    tensors = load_file(source / shard)
    converted = {name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
                 for name, tensor in tensors.items()}
    with safe_open(source / shard, framework='pt') as original:
        save_file(converted, destination / shard, metadata=original.metadata())
    reloaded = load_file(destination / shard)
    assert reloaded.keys() == converted.keys()
    for name, tensor in converted.items():
        assert torch.equal(reloaded[name], tensor), name
        total_size += tensor.numel() * tensor.element_size()
    del tensors, converted, reloaded
    print(f'Converted and verified {shard}', flush=True)

index['metadata']['total_size'] = total_size
(destination / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
config_path = destination / 'config.json'
config = json.loads(config_path.read_text())
config['dtype'] = 'bfloat16'
config_path.write_text(json.dumps(config, indent=2) + '\n')
print(json.dumps({'destination': str(destination), 'tensor_bytes': total_size}))
