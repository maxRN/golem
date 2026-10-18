# Nebius hackathon

The task is described in [TASK.md](TASK.md).
The simulation code is based on [so101-sim-pickplace-gr00t](https://github.com/ddiddi/so101-sim-pickplace-gr00t).

## Locations

| Item | Location |
| --- | --- |
| GPU VM | `ubuntu@81.85.2.207` |
| Simulation checkout on the VM | `/home/ubuntu/work/so101_sim_pickplace` |
| Isaac Lab Python on the VM | `/home/ubuntu/work/env_isaaclab/bin/python` |
| Local browser viewer | `http://127.0.0.1:5173/` |

Isaac Sim runs on the GPU VM. Your local computer runs the WebRTC viewer. The scene contains
a blue 30 mm cube, a fixed green pad, and nine starting spots on a 3×3 grid with
50 mm spacing. The scripted controller repeats pick-and-place attempts across the grid.

## Watch the simulation live

### 1. Check the simulator on the VM

Run from your local computer:

```bash
ssh ubuntu@81.85.2.207
```

Then run on the VM:

```bash
pgrep -af '[s]cripts/live_grid.py'
tail -n 20 ~/grid-live.log
```

If the live process already exists and the log shows new `[live] picking from spot`
messages, continue to the local viewer. Completed attempts print `success=True` or
`success=False`.

If no live process exists, start it on the VM:

```bash
cd ~/work/so101_sim_pickplace
source env.sh
export PUBLIC_IP=81.85.2.207
nohup "$ISAACLAB_PY" -u scripts/live_grid.py --livestream 1 --viz kit \
  --kit_args="--/exts/omni.services.livestream.session/quitOnSessionEnded=false --/app/window/width=1280 --/app/window/height=720" \
  > "$HOME/grid-live.log" 2>&1 < /dev/null &
echo $!
```

Keep the printed process ID. Allow several minutes for scene initialization and shader
compilation. Use `tail -f ~/grid-live.log` to watch progress. Ctrl+C stops following the
log while the simulator continues in the background. The simulator must be started again
after a VM reboot.

### 2. Start the viewer locally

Open a separate local terminal. Use Node.js 20.18.1 or later and npm 10.2.3 or later.
Replace `/path/to/so101-sim-pickplace-gr00t` with your simulation checkout:

```bash
cd /path/to/so101-sim-pickplace-gr00t/web-viewer
npm ci
npm run dev
```

Run `npm ci` for initial setup or after dependency changes. Keep `npm run dev` running
and open [the live viewer](http://127.0.0.1:5173/). Wait for **Live**.
If port 5173 is already occupied by the viewer, use that existing server.

Close other viewer tabs before reconnecting, then reload the remaining tab. Mouse and
keyboard input are disabled in the video to protect the running scene.

## Find the existing videos

The verified nine-spot recording completed **9/9 successful placements with zero retries**.
It includes front and wrist camera views and is about 1 minute 58 seconds long.

| File | Location |
| --- | --- |
| Verified video on the VM | `/home/ubuntu/work/so101_sim_pickplace/logs/grid_verified/grid_demo.mp4` |
| Per-spot results | `results.json` in the same `grid_verified` directory |
| Final images | `spot_01.png` through `spot_09.png` in the same directory |

The live WebRTC script writes its log to `/home/ubuntu/grid-live.log`. To save an MP4,
use the recording command below.

## Record a new grid video

First stop the live simulator to free the GPU. On the VM, find its PID:

```bash
pgrep -af '[s]cripts/live_grid.py'
```

Run `kill -TERM PID`, replacing `PID` with the simulator's process ID. Wait until
the process exits before recording.

Then run on the VM:

```bash
cd ~/work/so101_sim_pickplace
source env.sh
OUT="logs/grid_$(date +%Y%m%d_%H%M%S)"
"$ISAACLAB_PY" -u scripts/record_grid.py --output "$OUT"
echo "$PWD/$OUT"
```

Keep the SSH terminal open until recording finishes. The timestamp gives each run its
own directory. The recorder visits all nine spots and writes:

```text
logs/grid_YYYYMMDD_HHMMSS/
  grid_demo.mp4
  results.json
  spot_01.png
  ...
  spot_09.png
```

Expect `GRID OK: 9/9 successful spots` when all placements succeed. A failed placement
remains in the video, is marked in `results.json`, and causes the command to exit with an error.
Inspect the results before using the recording. To record selected spots, append
`--spots 1 5 9` to the recording command.

For the default `--output logs/grid`, the video is saved to
`/home/ubuntu/work/so101_sim_pickplace/logs/grid/grid_demo.mp4`.
Restart the live simulator with the earlier command when recording is complete.

## Copy recordings into this golem folder

Run from your local `golem` checkout to copy the verified recording and its results:

```bash
mkdir -p videos
scp -r ubuntu@81.85.2.207:/home/ubuntu/work/so101_sim_pickplace/logs/grid_verified \
  videos/
```

Open `videos/grid_verified/grid_demo.mp4` with your preferred video player.

For a new recording, replace `grid_verified` in the `scp` source with the timestamped
directory printed by the recording command. Videos remain on the VM until you copy them.

Training episodes collected with `collect.sh` have separate camera clips under
`/home/ubuntu/work/datasets/local/<DATASET_NAME>/videos/chunk-000/observation.images.front/`
and `observation.images.wrist/`, named `episode_000000.mp4` and onwards.
The current default dataset name is `so101_sim_pickplace`.

## Troubleshoot or stop

For a blank stream, inspect `~/grid-live.log` on the VM, wait for initialization, and reload
the viewer. If the log reports deleted geometry or an invalid physics view, stop that live
process and restart it. TCP 49100 and UDP 47998 must be reachable from your local computer.

Press Ctrl+C in the local viewer terminal to stop the viewer. Stop the simulator on the VM
with `kill -TERM PID` for its specific process. Stopping either process leaves the billable
Nebius VM running.

More WebRTC troubleshooting is in `docs/webrtc-stream.md` in the simulation checkout.

VM Git push verification completed on 18 October 2026.
