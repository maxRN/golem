# Submission checklist

Unchecked items still need completion or verification. Requirements are in
[TASK.md](TASK.md), subject to the correction below.

## Out of scope

Nebius Object Storage uploads are out of scope. The Object Storage submission
requirement in TASK.md and README.md is incorrect, per the team's clarification.

## Main policy

- [ ] Generate and accept the Cosmos Transfer training episodes with actions aligned to the generated videos.
- [ ] Fine-tune GR00T on Nebius using curated sim, Cosmos-augmented sim, and available real teleop data.
- [ ] Verify the saved checkpoint reloads and produces finite action predictions offline.
- [ ] Publish the checkpoint as a GitHub Release asset and link it from the README.

## Ablation policy

- [ ] Prepare the same curated sim and real data, excluding Cosmos Transfer augmentations.
- [ ] Fine-tune a second GR00T checkpoint on Nebius with comparable training settings.
- [ ] Verify the saved checkpoint reloads and produces finite action predictions offline.
- [ ] Publish the checkpoint as a GitHub Release asset and link it from the README.

Both policies may use Cosmos Reason curation. Document that this ablation measures
the effect of Transfer augmentation.

## Cosmos evidence and dataset manifest

- [ ] Run Cosmos 3 Reason on our sim and/or real episode videos and save its curation JSON.
- [ ] Document at least one episode-selection decision based on Reason's output.
- [ ] Save Transfer prompts, settings, source episode mappings, and original/generated video comparisons.
- [ ] Create a dataset manifest for each policy with episode counts per source and provenance.
- [ ] Verify compliance with the 300-episode dataset cap and 50-episode real teleop cap.
- [ ] Include training configurations, logs, and reproduction commands for both checkpoints.

## Robot evaluation

- [ ] Deploy and self-evaluate both checkpoints on the real SO-101; record success counts and starting grid spots.
- [ ] Save videos of the best real robot runs and identify the main checkpoint for judging.

Real robot evaluation is outside the one-episode smoke test in PLAN.md, but is
part of the full challenge submission workflow.

## Demo / pitch video

- [ ] Record a short explanation of the task and sim-to-real approach.
- [ ] Show the simulation, Cosmos before/after examples, and a Reason curation decision.
- [ ] Show available robot results and the main-versus-ablation comparison; state any evaluation limitations.
- [ ] Export the demo / pitch video and add its link to the README.

## Final submission

- [ ] Publish the source, documentation, and artifact links on GitHub without credentials.
- [ ] Complete the event portal submission and provide the GitHub and demo / pitch video links by 18:30 Berlin time on 18 October 2026, as recorded in README.md.
