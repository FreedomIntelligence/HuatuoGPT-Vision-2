#!/usr/bin/env bash
# Thinking RL, e.g. Qwen3.8-27B after pre-training, on 8 x 80GB GPUs.
# Usage: MODEL_PATH=... TRAIN_FILE=... REWARD_MODEL_PATH=... bash run_rl_qwen38_27b.sh [extra hydra overrides]
# Every setting below can be overridden with an environment variable, see README.md.
set -euo pipefail

export REQUIRE_THINKING=true   # responses without </think> get reward 0
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-hv2_rl_qwen38_27b}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"

# length penalty on the final answer after </think>: 0 at 200 tokens -> -0.4 at 4096 tokens
export ANSWER_LEN_START="${ANSWER_LEN_START:-200}" ANSWER_LEN_END="${ANSWER_LEN_END:-4096}" ANSWER_PENALTY_FACTOR="${ANSWER_PENALTY_FACTOR:-0.4}"
# length penalty on the whole response (thinking + answer): 0 at 4096 tokens -> -0.5 at MAX_RESPONSE_LENGTH
export OVERLONG_PENALTY="${OVERLONG_PENALTY:-True}" OVERLONG_START="${OVERLONG_START:-4096}" OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-0.5}"

# memory settings for a 27B policy and the grader on one node
export ROLLOUT_TP="${ROLLOUT_TP:-2}" SP_SIZE="${SP_SIZE:-4}" PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
export BF16_MASTER_WEIGHTS="${BF16_MASTER_WEIGHTS:-True}"
export REWARD_MODEL_DP="${REWARD_MODEL_DP:-2}" REWARD_MODEL_TP="${REWARD_MODEL_TP:-4}" REWARD_MODEL_MEM_FRACTION="${REWARD_MODEL_MEM_FRACTION:-0.10}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_rl.sh" "$@"
