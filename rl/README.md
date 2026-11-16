<div align="center">
<h1>Medical Multimodal Reinforcement Learning</h1>
<p><b>Stage 2 of HuatuoGPT-Vision-2</b></p>
</div>

<p align="center">
📃 <a href="https://openreview.net/pdf?id=0BXKsjKlFJ" target="_blank">Paper</a> ｜
📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/PubMedVision-RL" target="_blank">PubMedVision-RL</a> ｜
⚖️ <a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-2-Grader" target="_blank">Grader</a> ｜
🏠 <a href="../README.md">HuatuoGPT-Vision-2</a>
</p>

## ⚡ Introduction

This code elicits the medical vision capabilities of a pre-trained multimodal LLM with RL. It runs GRPO with the DAPO recipe on an unmodified [verl](https://github.com/verl-project/verl), and supports the **Qwen3-VL** and **Qwen3.8** model families.

- **Two kinds of rewards.** Multiple-choice questions get a verifiable reward. Open-ended questions are graded against weighted rubrics.
- **Multimodal grader.** [HuatuoGPT-Vision-2-Grader](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-2-Grader) sees the image together with the answer and checks all rubric items in one generation.
- **Text and image tasks together.** Image tasks and text-only tasks (e.g. general-domain rubrics) can be mixed in one run.
- **Plug-in design.** Only verl's public extension points are used (dataset, reward function, agent loop and trainer), so most verl options keep working.

## 🛠️ Installation

Prepare Python, CUDA, PyTorch and SGLang following the [verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html), then install this package:

```bash
git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision-2.git
cd HuatuoGPT-Vision-2/rl
pip install -r requirements.txt
```

The code needs **verl >= 0.9.1** and was tested with verl commit [`2b4ed68`](https://github.com/verl-project/verl/tree/2b4ed6826a39205838d0ca004bdd942334dbe4b6) (`0.10.0.dev`). If a newer verl release breaks something, install the tested commit:

```bash
pip install "verl @ git+https://github.com/verl-project/verl.git@2b4ed6826a39205838d0ca004bdd942334dbe4b6"
```

## 📚 Data

```bash
huggingface-cli download FreedomIntelligence/PubMedVision-RL --repo-type dataset --local-dir PubMedVision-RL
```

[PubMedVision-RL](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision-RL) has 35K multiple-choice and 35K open-ended questions. Pass its `data` folder as `TRAIN_FILE`. Your own data can be added as `.json`, `.jsonl` or `.parquet` files with these records:

| Type | Fields | Reward |
| --- | --- | --- |
| Multiple choice | `question`, `options`, `answer_idx`, `images` (optional) | 1 if the option in `<answer>X</answer>` is correct |
| Open ended | `question`, `rubrics`, `images` (optional) | Matched rubric points / total positive points |

```jsonc
// Multiple choice
{"question": "Which structure is enhanced?", "options": {"A": "Liver", "B": "Spleen"}, "answer_idx": "B", "images": ["images/ct.jpg"]}

// Open ended (rubric points may be negative for undesirable behavior)
{"question": "What does this image show?", "images": ["images/xray.jpg"],
 "rubrics": [{"criterion": "Identifies a right pleural effusion.", "points": 5},
             {"criterion": "Gives a definitive diagnosis without hedging.", "points": -3}]}
```

Image paths may be absolute or relative to the data file or its parent folders. Text-only tasks simply leave out `images`. Instead of `question`, a record may give `prompt` as a list of chat messages.

## 🚀 Training

Two scripts cover models with and without thinking:

```bash
# Without thinking, e.g. Qwen3-VL-8B-Instruct
MODEL_PATH=/path/to/pretrained_qwen3vl_8b \
TRAIN_FILE=/path/to/PubMedVision-RL/data \
REWARD_MODEL_PATH=/path/to/HuatuoGPT-Vision-2-Grader \
bash run_rl_qwen3vl_8b.sh

# With thinking, e.g. Qwen3.8-27B
MODEL_PATH=/path/to/pretrained_qwen38_27b \
TRAIN_FILE=/path/to/PubMedVision-RL/data \
REWARD_MODEL_PATH=/path/to/HuatuoGPT-Vision-2-Grader \
bash run_rl_qwen38_27b.sh
```

| | `run_rl_qwen3vl_8b.sh` | `run_rl_qwen38_27b.sh` |
| --- | --- | --- |
| Thinking | No | Yes. A response without `</think>` gets reward 0 |
| `MAX_RESPONSE_LENGTH` | 4096 | 8192 |
| Answer length penalty | 0 at 200 tokens, -0.4 at 4096 tokens (the whole response) | 0 at 200 tokens, -0.4 at 4096 tokens (the text after `</think>`) |
| Overall length penalty | Off | 0 at 4096 tokens, -0.5 at 8192 tokens (thinking + answer) |
| Memory settings | Defaults | `ROLLOUT_TP=2 SP_SIZE=4 BF16_MASTER_WEIGHTS=True`, grader with `DP=2 TP=4` |

Both penalties are linear and are subtracted from the task reward. Other models work the same way: use the script that matches whether the model thinks.

The script starts the grader with SGLang on the same GPUs and stops it when training ends. To use a grader that is already running, set `START_REWARD_SERVER=0` and `REWARD_MODEL_URL=http://host:port/v1/chat/completions`.

The defaults follow the paper. Override them with environment variables, or append any verl option to the command.

| Variable | Default | Description |
| --- | --- | --- |
| `TRAIN_BATCH_SIZE` | 128 | Prompts per step |
| `ROLLOUT_N` | 8 | Responses per prompt |
| `PPO_MINI_BATCH_SIZE` | 16 | Prompts per actor update |
| `ACTOR_LR` | 1e-6 | Learning rate |
| `TOTAL_TRAINING_STEPS` | 200 | Training steps |
| `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | 4096 / 4096, or 8192 with thinking | Token limits |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | 1.0 / 1.0 / -1 | Rollout sampling. Without truncation the rollouts follow the policy the actor trains |
| `MAX_IMAGE_TOKENS` | 0 | Optional cap on visual tokens per image. With 0, images keep their resolution and are shrunk only when the prompt would exceed `MAX_PROMPT_LENGTH` |
| `VAL_FILE` | 256 training samples | Validation data, evaluated every `TEST_FREQ` steps |
| `VAL_BEFORE_TRAIN` / `LOG_VAL_GENERATIONS` | False / 20 | Validate before the first step / validation samples shown in the logger |
| `GRADER_MAX_PIXELS` | 0 | Optional cap on the pixels of each image sent to the grader. With 0, images are sent as they are and the grader's own processor resizes them |
| `LOGGER` / `PROJECT_NAME` / `EXPERIMENT_NAME` | `[console]` / `HuatuoGPT-Vision-2` / `hv2_rl` | Logging, e.g. `LOGGER='["console","wandb"]'` |
| `REQUIRE_THINKING` | false | Use true for thinking models such as Qwen3.8. Responses without `</think>` get zero reward |
| `ANSWER_LEN_START` / `ANSWER_LEN_END` / `ANSWER_PENALTY_FACTOR` | 200 / 4096 / 0.4 | Penalty on the final answer (after `</think>`, or the whole response without thinking). Set the factor to 0 to turn it off |
| `OVERLONG_PENALTY` / `OVERLONG_START` / `OVERLONG_PENALTY_FACTOR` | True with thinking, else False / 4096 / 0.5 | DAPO overlong penalty on the whole response, from `OVERLONG_START` to `MAX_RESPONSE_LENGTH` |

Checkpoints are saved under `runs/checkpoints/` and include a Hugging Face model in `actor/huggingface`.

**Large models.** The defaults of `run_rl.sh` fit 8B models on 8 x 80GB GPUs. `run_rl_qwen38_27b.sh` adds the settings below for 27B and 32B models on one node. To train a large model without thinking (e.g. Qwen3-VL-32B-Instruct), add them to `run_rl_qwen3vl_8b.sh`:

```bash
ROLLOUT_TP=2 SP_SIZE=4 PPO_MAX_TOKEN_LEN_PER_GPU=8192 \
BF16_MASTER_WEIGHTS=True \
REWARD_MODEL_DP=2 REWARD_MODEL_TP=4 REWARD_MODEL_MEM_FRACTION=0.10 \
MODEL_PATH=... TRAIN_FILE=... REWARD_MODEL_PATH=... bash run_rl_qwen3vl_8b.sh
```

| Variable | Effect |
| --- | --- |
| `ROLLOUT_TP` / `SP_SIZE` | Tensor parallel for rollout and Ulysses sequence parallel for the actor |
| `BF16_MASTER_WEIGHTS` | Keep weights and optimizer states in bf16 and update them with stochastic rounding |
| `REWARD_MODEL_*` | Serve the grader with less GPU memory |

## 🔍 Design

| Component | File | Approach |
| --- | --- | --- |
| Dataset | `hv2rl/dataset.py` | Reads multiple-choice and open-ended records, with or without images |
| Reward | `hv2rl/reward.py` | Exact match for multiple choice. The grader receives the images, the question, the final answer and the rubrics. The answer length penalty is applied here |
| Rollout | `hv2rl/agent_loop.py` | verl's single-turn loop. Vision placeholder tokens are banned during SGLang sampling |
| Trainer | `hv2rl/dapo_trainer.py` | verl's DAPO recipe: clip-higher, token-level loss, dynamic sampling and overlong reward shaping |
