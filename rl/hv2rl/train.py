"""Entry point: python -m hv2rl.train <hydra overrides>. See run_rl.sh."""

import os

import hydra
from omegaconf import DictConfig
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from hv2rl.trainer import TaskRunner

# settings read by the reward function inside Ray workers
WORKER_ENV = ("REWARD_MODEL_URL", "REWARD_MODEL_NAME", "REWARD_MODEL_TIMEOUT", "REWARD_MODEL_MAX_TOKENS",
              "GRADER_MAX_PIXELS", "REQUIRE_THINKING", "POLICY_TOKENIZER", "ANSWER_LEN_START", "ANSWER_LEN_END", "ANSWER_PENALTY_FACTOR",
              "PYTHONPATH", "TMPDIR", "RAY_TMPDIR", "SWANLAB_LOG_DIR")


class _RunnerWithEnv:
    """Pass the reward settings to every worker, also when attaching to an existing Ray cluster."""

    def __init__(self):
        self.env = {name: os.environ[name] for name in WORKER_ENV if name in os.environ}

    def remote(self):
        return self.options().remote()

    def options(self, **options):
        runtime_env = options.pop("runtime_env", {})
        runtime_env["env_vars"] = {**self.env, **runtime_env.get("env_vars", {})}
        return TaskRunner.options(runtime_env=runtime_env, **options)


@hydra.main(config_path="pkg://verl.trainer.config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    run_ppo(config, task_runner_class=_RunnerWithEnv())


if __name__ == "__main__":
    main()
