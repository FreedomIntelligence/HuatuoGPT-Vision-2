#!/usr/bin/env bash
# Non-thinking RL, e.g. Qwen3-VL-8B-Instruct after pre-training, on 8 x 80GB GPUs.
# Usage: MODEL_PATH=... TRAIN_FILE=... REWARD_MODEL_PATH=... bash run_rl_qwen3vl_8b.sh [extra hydra overrides]
# Every setting below can be overridden with an environment variable, see README.md.
set -euo pipefail

export REQUIRE_THINKING=false
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-hv2_rl_qwen3vl_8b}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"

# length penalty on the answer (the whole response here): 0 at 200 tokens -> -0.4 at 4096 tokens
export ANSWER_LEN_START="${ANSWER_LEN_START:-200}" ANSWER_LEN_END="${ANSWER_LEN_END:-4096}" ANSWER_PENALTY_FACTOR="${ANSWER_PENALTY_FACTOR:-0.4}"
# no extra penalty on the whole response
export OVERLONG_PENALTY="${OVERLONG_PENALTY:-False}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_rl.sh" "$@"
