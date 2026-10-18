**NVIDIA × NEBIUS AI BUILDERS HACKATHON   BERLIN | 18 OCT 2026**

**Sim to Table**

*Physical AI challenge: train in sim, bridge with Cosmos, teleop on real SO-101 arms, evaluate in the real world*

The simulator in the repo makes robot demos instantly, but they look nothing like the real table. Your real arm makes demos that look right, but you can only record a few dozen in an afternoon. Today you combine both: generate demos in Isaac Sim, use NVIDIA Cosmos to make them look real and variant, add a small set of real teleop demos, train one ACT policy.

 

| Task | Robot | Scored on |
| :---- | :---- | :---- |
| Pick the 30 mm cube, place it on the pad | SO-101 arm | 3 autonomous trials on your arm run by a judge |

# **01 \- The challenge pipeline**

| Stage | Where | What you do | Output |
| :---- | :---- | :---- | :---- |
| 1\. Simulate | Isaac Sim (Nebius VM) | Run the scripted expert to collect pick-and-place episodes at scale. | 100s of sim episodes |
| 2\. Transfer | Cosmos Transfer 2.5 \+ Cosmos 3 Reasoner | Re-render sim clips toward real-world appearance (lighting, table surface, textures) with structure-preserving control, actions unchanged. Use Cosmos 3 Reasoner to score and curate sim and real episodes. | Cosmos-augmented episodes \+ curation JSON |
| 3\. Teleop | Your SO-101 leader/follower pair | Record real demonstrations at your station with the leader arm. | Few real episodes |
| 4\. Train | Nebius GPU VM | Train one ACT or GROOT policy on the mixed sim \+ Cosmos \+ real dataset, plus an ablation without Cosmos. | Two checkpoints |
| 5\. Evaluate | Real SO-101 | Self-evaluate both checkpoints on your arm. Judges then run the main checkpoint on the same arm. | Real-world success count |

 

## **Why this shape**

Sim alone fails on hardware because it has one light, one table, and one renderer. Real data alone is starved: few demos is not much to generalise from. Cosmos is the bridge: it turns sim data into data that looks like the real world, so the sim episodes actually help the real policy. The rules below are set so that this bridge, not raw teleop volume, decides the leaderboard.

## **Data rules**

•     **Real teleop is capped at 50 episodes;** the cap keeps real data scarce so sim and Cosmos have to carry weight.

•     **The main checkpoint must include sim-derived data** (raw sim and/or Cosmos-augmented). The dataset manifest must list episode counts per source.

•     **Total dataset cap: 300 episodes.**

•     **Policy: ACT** or GROOT

# 

# 

# 

# 

# 

# **02 \- Judging (100 points)**

Every team gets a station set up and calibrated: SO-101 leader/follower pair, camera on a standard mount, lamp, 30 mm cube, green pad, and a 9-spot cube grid taped to the table. You teleop here, self-evaluate here, and judges score you here. Nothing about the room changes between practice and judging.

**The one thing judges control is cube placement.** They place the cube on grid spots in a fixed sequence, identical for every team and revealed only at eval time. A policy that only learned one spot will fail most of them.

| Criterion | Pts | How it is scored |
| :---- | :---- | :---- |
| Accuracy real-world success | 30 | 3 judge-run trials on a real SO-101, 2 min cap each. 10 pts per success. Success \= cube released and at rest on the green pad. Leaderboard shows the raw count (x / 3). |
| NVIDIA Cosmos meaningful use | 20 | **10 for Reason:** called on your own sim and/or real episode videos, JSON saved, one stated curation decision made from it. **10 for Transfer:** structure-preserving augmentation of sim data toward the real look. |
| Nebius  | 20 | Usage for Nebius Cloud for training |
| Technical competence | 15 | Be able to deploy on the robot IRL |
| Creativity | 15 | Ensemble of models used |

 

 **03 \- How Cosmos and Nebius should be used**

## **NVIDIA Cosmos**

**Cosmos Transfer 2.5 is the bridge:** Re-render sim episodes so the lighting, table and textures look like the real world while geometry and motion stay fixed, so the recorded actions remain valid. Use edge conditioning: it preserves structure, which is exactly what keeps actions aligned. Target the look of your real station.

**Cosmos 3 Reason is the curator:** Point it at sim and real episode MP4s to score them (did the grasp succeed, was the motion clean) and drop bad episodes before training. 

## **Nebius**

•     **Object Storage** is the submission channel (https\://storage.\<region\>.nebius.cloud). Your folder lands there by the cutoff or it doesn't exist.

# **04 \- References**

* [Starter repo:  **so101-sim-pickplace-gr00t**](https://github.com/ddiddi/so101-sim-pickplace-gr00t)   
* [**Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac**](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)  
* [**Calibrating the SO-101**](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/07-calibrating-so101.html)  
* [**Nebius Token Factory quickstart**](https://docs.tokenfactory.nebius.com/quickstart)    
* Skills: **[https\://github.com/NVIDIA/skills](https://github.com/NVIDIA/skills)**


