# Fast main GR00T policy

Run artifacts live on `ubuntu@81.85.2.207` under `/home/ubuntu/work/golem_main_20261018_fast`.

The candidate set contains episodes 0–49 from each of `so101_sim_dr` and `so101_sim_dr_cam2`, plus one augmented copy of each. FFmpeg changes blue tones and brightness/gamma in both camera views, preserving frame counts, timestamps, states, and actions. These are fast image transforms, not Cosmos Transfer generations.

Cosmos 3 Nano classifies four chronological front-camera samples per episode, ten episodes per contact sheet. Cosmos scores each episode by the relative next-token probabilities of good, uncertain, and bad. The ranking score is 10 × P(good) + 5 × P(uncertain), normalized over those three choices. The top 150 candidates form the training dataset. Selection is relative ranking with a minimum count, not certification that every retained episode succeeds. The wrist views and motion between sampled frames are not reviewed by Cosmos. Every retained camera stream is checked for frame count, resolution, and frame rate before packaging.

The task text becomes “pick up the cube and place it on the green pad” to permit changed cube colors. The dataset is reindexed, GR00T statistics are regenerated, and the real GR00T loader checks it. GR00T N1.7 trains for 20 optimizer steps with batch size one, with the language and vision backbones frozen and the projector and action decoder trainable. Offline verification reloads `policy/checkpoint-20` and checks finite action predictions with the expected shapes. No real robot evaluation is performed.

Execution order on the VM:

```bash
/home/ubuntu/work/Isaac-GR00T/.venv/bin/python -u scripts/prepare.py
/home/ubuntu/work/cosmos3_reason_env/bin/python -u scripts/reason.py
bash scripts/finish.sh
```

`finish.sh` waits for selection, then packages the dataset, prepares statistics, trains, and verifies inference. The first two commands run from the copied scripts under the run directory. Run each once into a new output directory; dataset packaging uses hard links and deliberately fails if rerun into an existing dataset.
