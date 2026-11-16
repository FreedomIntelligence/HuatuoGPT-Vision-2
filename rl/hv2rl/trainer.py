"""Task runner: verl's native workers + the DAPO dynamic-sampling trainer."""

from __future__ import annotations

from pprint import pprint

import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo_v0 import BaseTaskRunner
from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler, need_critic, need_reference_policy
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.config import HFModelConfig
from verl.workers.engine_workers import ActorRolloutRefWorker

from hv2rl.dapo_trainer import RayDAPOTrainer


@ray.remote
class TaskRunner(BaseTaskRunner):
    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0) or config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"
        return ActorRolloutRefWorker, RayWorkerGroup

    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))

        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        tokenizer, processor = model_config.tokenizer, model_config.processor
        datasets = [
            create_rl_dataset(files, config.data, tokenizer, processor, is_train=is_train,
                              max_samples=config.data.get(f"{split}_max_samples", -1))
            for files, is_train, split in [(config.data.train_files, True, "train"), (config.data.val_files, False, "val")]
        ]
        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=self.init_resource_pool_mgr(config),
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=datasets[0],
            val_dataset=datasets[1],
            collate_fn=collate_fn,
            train_sampler=create_rl_sampler(config.data, datasets[0]),
        )
        trainer.init_workers()
        trainer.fit()
