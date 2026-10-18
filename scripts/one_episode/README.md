# One-episode workflow

These scripts run on the existing Nebius VM at `ubuntu@81.85.2.207`.
All run artifacts are under `/home/ubuntu/work/golem_one_episode_20261018`.
The original dataset and existing simulation processes are left intact.

1. Run `prepare.py` with the GR00T environment's Python to snapshot episode 0 from
   `so101_sim_dr_cam2` and write the Transfer specifications.
2. Run `bash run_transfer.sh` to augment both complete camera streams with the
   base edge model, using 12 diffusion steps and 480-resolution generation.
3. Run `bash setup_reason.sh` to install Cosmos 3 Nano's separate environment and
   download its checkpoint. Run `reason.py source` with that environment's Python.
4. Inspect generated frames and save `review/decision.json` with an explicit
   `verdict` of `accept`, `reject`, or `review`, plus the decision's scope, reasons,
   and `camera_selection` mapping each camera to `augmented` or `source`.
   This run keeps the original wrist view because its augmentation changed the
   cube color and removed the green pad. Only the front view is augmented.
5. Run `package.py` with the GR00T environment's Python. It requires an accepted
   decision and verifies both videos have 299 frames at 30 fps and 640×480 pixels.
6. Run the existing dataset preparation command:

   ```bash
   bash /home/ubuntu/work/so101_sim_pickplace/prepare_dataset.sh \
     /home/ubuntu/work/golem_one_episode_20261018/dataset
   ```

7. Run `bash run_train.sh` for two optimizer steps with batch size one. The
   language and vision backbone stay frozen; the projector and action decoder train.
8. Reload the checkpoint and save offline action predictions:

   ```bash
   cd /home/ubuntu/work/Isaac-GR00T
   .venv/bin/python /home/ubuntu/work/golem_one_episode_20261018/scripts/infer.py
   ```

The checkpoint is `policy/checkpoint-2`. Offline verification writes
`inference.json` and `inference_actions.npz`. This run checks the pipeline only;
it does not evaluate policy quality or operate the real robot. Reason reviews
only the source episode to keep this first run short.

## Smaller checkpoint export

Export the full checkpoint in bfloat16 without requiring a separate base model:

```bash
cd /home/ubuntu/work/Isaac-GR00T
.venv/bin/python /home/ubuntu/work/golem_one_episode_20261018/scripts/export_bf16.py \
  /home/ubuntu/work/golem_one_episode_20261018/policy/checkpoint-2 \
  /home/ubuntu/work/golem_one_episode_20261018/policy/checkpoint-2-bf16
.venv/bin/python /home/ubuntu/work/golem_one_episode_20261018/scripts/infer.py \
  --checkpoint /home/ubuntu/work/golem_one_episode_20261018/policy/checkpoint-2-bf16 \
  --output-dir /home/ubuntu/work/golem_one_episode_20261018/export_bf16
```

The export destination must not already exist. The script converts and verifies
each tensor, preserves processor and training metadata, and updates the weight
index and model precision. Tensor storage falls from 12,576,064,000 to
6,288,032,000 bytes. The policy loader already uses bfloat16 for inference.
The exported checkpoint produced identical actions on the saved observation.
