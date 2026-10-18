# One-episode Cosmos-to-GR00T workflow

## Goal

Produce any fine-tuned GR00T policy from one Cosmos-augmented simulation episode,
then reload it and run offline inference. Policy quality and real robot evaluation
are outside this first task's scope.

## Completed steps

- [x] Snapshot one complete successful simulation episode with both camera views, synchronized states/actions, and task text.
- [x] Generate full front and wrist videos with Cosmos Transfer 2.5 and edge conditioning.
- [x] Run Cosmos 3 Nano locally from Hugging Face on the source episode and save curation JSON.
- [x] Inspect sampled generated frames and record a curation decision.
- [x] Package one training episode with the augmented front view, original wrist view, and unchanged states/actions.
- [x] Verify frame counts, frame rates, dimensions, and unchanged action data.
- [x] Load the packaged episode through GR00T's actual data loader.
- [x] Fine-tune pretrained GR00T N1.7 on Nebius for two optimizer steps with batch size one.
- [x] Save and reload the checkpoint.
- [x] Run one recorded observation through the policy and save finite action predictions with the expected shapes.

## Agreed shortcuts

The language and vision backbone stayed frozen. Training updated the projector
and action decoder only. Cosmos 3 Nano reviewed the source episode; a second
Reason pass was skipped. Manual inspection rejected the generated wrist view
because it changed the cube color and removed the target pad. The original wrist
view was retained. The augmented front view was accepted for pipeline validation
only, without certifying detailed visual geometry or cross-view appearance.

No performance threshold, further quality tuning, or real robot rollout was required.

## Result

Checkpoint on `ubuntu@81.85.2.207`:

```text
/home/ubuntu/work/golem_one_episode_20261018/policy/checkpoint-2
```

Reloaded successfully and produced a 16-step sequence of finite arm and gripper
actions from a recorded observation. Details and evidence are in
[artifacts/one_episode/RESULT.md](artifacts/one_episode/RESULT.md).

## Follow-up

Before generating 100 accepted augmented episodes, improve augmentation fidelity
and define the final dataset selection. Keep training data within the challenge's
300-episode cap and real teleoperation within 50 episodes. The full challenge also
requires an ablation without Cosmos and real robot evaluation.

## Scaling the next run

Augment selected episodes in batches, package the accepted dataset, then train
one policy on the whole dataset. Do not repeat setup and policy training for each
simulation episode.

The first run included environment setup, model downloads/loading, generation
of both camera views, source curation, and checkpoint saving. The two optimizer
steps took about 26 seconds including the training-time checkpoint save. Cosmos
generation dominated the end-to-end workflow. These timings do not predict the
cost of a full policy training run.

- Keep Cosmos Transfer loaded in one process across multiple episodes.
- Initially augment only the front camera and retain the synchronized original wrist videos. Confirm this camera combination is suitable before scaling; this first run did not evaluate policy quality.
- Target 50–100 accepted augmentations alongside selected raw simulations. Having 250 simulations does not require generating 250 augmentations.
- Benchmark the distilled Transfer model on short segments. Its documented limit is 93 sampled frames per clip; do not feed complete longer episodes through it without a validated segmentation strategy.
- Preserve the original frame/action mapping when splitting or joining segments. Check motion and appearance at chunk boundaries before adopting this shortcut.
- Keep Cosmos Reason loaded for batch curation. Use fewer sampled frames, smaller inputs, and short structured responses, then check whether this still catches relevant failures.
- Save generation settings, source IDs, curation decisions, and completion status per episode. Resume from completed outputs rather than regenerating them.
- Package accepted episodes together and train once. Run the required ablation separately.
- Select the final mixture within the 300-episode dataset cap and 50-episode real teleoperation cap.

