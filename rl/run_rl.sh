#!/usr/bin/env bash
# HuatuoGPT-Vision-2 RL (GRPO with the DAPO recipe). See README.md for setup.
# Usage: MODEL_PATH=... TRAIN_FILE=... REWARD_MODEL_PATH=... bash run_rl.sh [extra hydra overrides]
# Ready-made settings: run_rl_qwen3vl_8b.sh (no thinking) and run_rl_qwen38_27b.sh (thinking).

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-hv2_rl}"
LOGGER="${LOGGER:-[console]}"
PROJECT_NAME="${PROJECT_NAME:-HuatuoGPT-Vision-2}"

# ----------------------------------------------------------------------------- inputs
MODEL_PATH="${MODEL_PATH:-}"            # policy, e.g. the pre-trained Qwen3-VL-8B
TRAIN_FILE="${TRAIN_FILE:-}"            # file or folder, e.g. /path/to/PubMedVision-RL/data
VAL_FILE="${VAL_FILE:-}"                # optional; defaults to 256 samples of TRAIN_FILE
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-}"  # HuatuoGPT-Vision-2-Grader

# ----------------------------------------------------------------------------- paper settings
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
REQUIRE_THINKING="${REQUIRE_THINKING:-false}"   # true for thinking models such as Qwen3.8 (no </think> = reward 0)
THINKING=$([[ "${REQUIRE_THINKING}" == "true" ]] && echo 1 || echo 0)
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$((THINKING ? 8192 : 4096))}"
MAX_IMAGE_TOKENS="${MAX_IMAGE_TOKENS:-0}"          # 0: no per-image cap, images are shrunk only to fit MAX_PROMPT_LENGTH
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
MAX_NUM_GEN_BATCHES="${MAX_NUM_GEN_BATCHES:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}" TOP_P="${TOP_P:-1.0}" TOP_K="${TOP_K:--1}"   # untruncated sampling keeps rollouts on-policy
# Length penalties (both linear, subtracted from the reward):
# 1. final answer (text after </think>, or the whole response without thinking): 0 at ANSWER_LEN_START tokens,
#    -ANSWER_PENALTY_FACTOR at ANSWER_LEN_END tokens and beyond
export ANSWER_LEN_START="${ANSWER_LEN_START:-200}" ANSWER_LEN_END="${ANSWER_LEN_END:-4096}"
export ANSWER_PENALTY_FACTOR="${ANSWER_PENALTY_FACTOR:-0.4}"
# 2. whole response including thinking (DAPO overlong shaping, on by default only with thinking):
#    0 at OVERLONG_START tokens, -OVERLONG_PENALTY_FACTOR at MAX_RESPONSE_LENGTH
OVERLONG_PENALTY="${OVERLONG_PENALTY:-$([[ ${THINKING} == 1 ]] && echo True || echo False)}"
OVERLONG_START="${OVERLONG_START:-4096}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-0.5}"
OVERLONG_BUFFER_LEN=$((MAX_RESPONSE_LENGTH - OVERLONG_START))
if [[ "${OVERLONG_PENALTY}" == "True" ]] && (( OVERLONG_BUFFER_LEN <= 0 )); then
  echo "OVERLONG_START (${OVERLONG_START}) must be below MAX_RESPONSE_LENGTH (${MAX_RESPONSE_LENGTH})" >&2; exit 1
fi
(( OVERLONG_BUFFER_LEN > 0 )) || OVERLONG_BUFFER_LEN=1   # unused when OVERLONG_PENALTY=False

# ----------------------------------------------------------------------------- GPUs
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"            # 2 for 27B / 32B
SP_SIZE="${SP_SIZE:-1}"                  # Ulysses sequence parallel for actor / reference
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
OFFLOAD="${OFFLOAD:-True}"
BF16_MASTER_WEIGHTS="${BF16_MASTER_WEIGHTS:-False}"  # True: bf16 weights + AdamW with stochastic rounding, saves memory
DEFER_GRAD_SYNC="${DEFER_GRAD_SYNC:-False}"  # True: reduce gradients once per mini-batch (faster, keeps full gradients, 8B only)
TEST_FREQ="${TEST_FREQ:-30}"
SAVE_FREQ="${SAVE_FREQ:-50}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-20}"   # validation samples shown in the logger

# ----------------------------------------------------------------------------- grader server
START_REWARD_SERVER="${START_REWARD_SERVER:-1}"   # 0: use an existing server at REWARD_MODEL_URL
REWARD_MODEL_PORT="${REWARD_MODEL_PORT:-30001}"
REWARD_MODEL_DP="${REWARD_MODEL_DP:-4}"
REWARD_MODEL_TP="${REWARD_MODEL_TP:-2}"
REWARD_MODEL_MEM_FRACTION="${REWARD_MODEL_MEM_FRACTION:-0.15}"
export REWARD_MODEL_URL="${REWARD_MODEL_URL:-http://127.0.0.1:${REWARD_MODEL_PORT}/v1/chat/completions}"
export REWARD_MODEL_NAME="${REWARD_MODEL_NAME:-default}"
export REQUIRE_THINKING POLICY_TOKENIZER="${MODEL_PATH}"   # the reward counts answer tokens with it

CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/${EXPERIMENT_NAME}"
LOG_DIR="${OUTPUT_DIR}/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RAY_TMP="/tmp/hv2rl-${UID}-$$"   # short path: Ray uses AF_UNIX sockets

if [[ "${DRY_RUN}" != "1" ]]; then
  : "${MODEL_PATH:?set MODEL_PATH}"
  : "${TRAIN_FILE:?set TRAIN_FILE}"
  if [[ "${START_REWARD_SERVER}" == "1" ]]; then : "${REWARD_MODEL_PATH:?set REWARD_MODEL_PATH or START_REWARD_SERVER=0}"; fi
  mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" && mkdir -m 700 -p "${RAY_TMP}"
  exec > >(tee -a "${LOG_DIR}/train_${EXPERIMENT_NAME}_${RUN_ID}.log") 2>&1
fi
VAL_ARGS=("data.val_files=${VAL_FILE:-${TRAIN_FILE}}")
BF16_ARGS=()
[[ "${BF16_MASTER_WEIGHTS}" == "True" ]] && BF16_ARGS=(
  "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16" "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16"
  "actor_rollout_ref.actor.optim.optimizer_impl=torchao.optim" "actor_rollout_ref.actor.optim.optimizer=_AdamW"
  "actor_rollout_ref.actor.optim.override_optimizer_config={bf16_stochastic_round:true}")
[[ -z "${VAL_FILE}" ]] && VAL_ARGS+=("data.val_max_samples=256")

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=true RAY_TMPDIR="${RAY_TMP}" VLLM_LOGGING_LEVEL=WARN NCCL_DEBUG=WARN

REWARD_SERVER_PID=""
stop_reward_server() { [[ -n "${REWARD_SERVER_PID}" ]] && kill -TERM -- "-${REWARD_SERVER_PID}" 2>/dev/null || true; }
start_reward_server() {
  echo "Starting grader ${REWARD_MODEL_PATH} on port ${REWARD_MODEL_PORT}"
  setsid "${PYTHON}" -m sglang.launch_server --model-path "${REWARD_MODEL_PATH}" --port "${REWARD_MODEL_PORT}" \
    --data-parallel-size "${REWARD_MODEL_DP}" --tensor-parallel-size "${REWARD_MODEL_TP}" \
    --mem-fraction-static "${REWARD_MODEL_MEM_FRACTION}" --served-model-name "${REWARD_MODEL_NAME}" \
    > "${LOG_DIR}/grader_${EXPERIMENT_NAME}_${RUN_ID}.log" 2>&1 &
  REWARD_SERVER_PID=$!
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${REWARD_MODEL_PORT}/health" >/dev/null && { echo "Grader ready"; return 0; }
    kill -0 "${REWARD_SERVER_PID}" 2>/dev/null || { echo "Grader failed, see ${LOG_DIR}" >&2; return 1; }
    sleep 5
  done
  echo "Grader did not start in time" >&2; return 1
}
trap stop_reward_server EXIT

ARGS=(
  # data
  "data.train_files=${TRAIN_FILE}" "${VAL_ARGS[@]}"
  "data.custom_cls.path=${PROJECT_DIR}/hv2rl/dataset.py" "data.custom_cls.name=MedicalRLDataset"
  "+data.max_image_tokens=${MAX_IMAGE_TOKENS}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" "data.gen_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}" "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True" "data.truncation=error" "data.return_raw_chat=True"
  # GRPO + DAPO (clip-higher, token-level loss, dynamic sampling, overlong shaping)
  "algorithm.adv_estimator=grpo" "algorithm.use_kl_in_reward=False"
  "algorithm.filter_groups.enable=True" "algorithm.filter_groups.metric=acc"
  "algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES}"
  "actor_rollout_ref.actor.clip_ratio_low=0.2" "actor_rollout_ref.actor.clip_ratio_high=0.28"
  "actor_rollout_ref.actor.clip_ratio_c=10.0" "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.use_kl_loss=False" "actor_rollout_ref.actor.entropy_coeff=0"
  # model and actor
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True" "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.actor.strategy=fsdp2" "actor_rollout_ref.ref.strategy=fsdp2"
  "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}" "actor_rollout_ref.actor.optim.lr_warmup_steps=1"
  "actor_rollout_ref.actor.optim.weight_decay=0.1" "actor_rollout_ref.actor.grad_clip=1.0"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.use_dynamic_bsz=True" "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE}"
  "actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE}"
  "actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD}"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD}" "${BF16_ARGS[@]}"
  "actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=${DEFER_GRAD_SYNC}"
  # rollout
  "actor_rollout_ref.rollout.name=sglang" "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.temperature=${TEMPERATURE}" "actor_rollout_ref.rollout.top_p=${TOP_P}" "actor_rollout_ref.rollout.top_k=${TOP_K}"
  "actor_rollout_ref.rollout.calculate_log_probs=False"
  "+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_piecewise_cuda_graph=True"
  "actor_rollout_ref.rollout.agent.default_agent_loop=medical_single_turn"
  "actor_rollout_ref.rollout.agent.agent_loop_config_path=${PROJECT_DIR}/configs/agent_loop.yaml"
  "actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE}" "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.val_kwargs.n=1"
  # reward
  "reward.num_workers=32" "reward.reward_manager.name=dapo"
  "reward.custom_reward_function.path=${PROJECT_DIR}/hv2rl/reward.py" "reward.custom_reward_function.name=compute_score"
  "+reward.reward_kwargs.overlong_buffer_cfg.enable=${OVERLONG_PENALTY}"
  "+reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}"
  "+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}"
  "+reward.reward_kwargs.overlong_buffer_cfg.log=False"
  "+reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}"
  # trainer
  "+ray_kwargs.ray_init._temp_dir=${RAY_TMP}" "trainer.use_v1=False" "trainer.balance_batch=True"
  "trainer.logger=${LOGGER}" "trainer.project_name=${PROJECT_NAME}" "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.n_gpus_per_node=${NGPUS_PER_NODE}" "trainer.nnodes=${NNODES}"
  "trainer.total_epochs=100" "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}" "trainer.test_freq=${TEST_FREQ}" "trainer.log_val_generations=${LOG_VAL_GENERATIONS}"
  "trainer.save_freq=${SAVE_FREQ}" "trainer.default_local_dir=${CHECKPOINT_DIR}" "trainer.resume_mode=auto"
  "actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model']"
  "hydra.run.dir=${OUTPUT_DIR}/hydra/${EXPERIMENT_NAME}"
)

cd "${PROJECT_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${ARGS[@]}" "$@"
  exit 0
fi
[[ "${START_REWARD_SERVER}" == "1" ]] && start_reward_server
"${PYTHON}" -u -m hv2rl.train "${ARGS[@]}" "$@"
