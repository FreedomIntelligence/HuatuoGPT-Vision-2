"""Single-turn rollout for multimodal medical RL on an unmodified verl."""

from typing import Any

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.workers.rollout.utils import get_vision_placeholder_token_ids


class MedicalAgentLoop(SingleTurnAgentLoop):
    """Two small additions to verl's single-turn loop.

    1. Qwen3.5 / 3.8 checkpoints with a vision processor build their prompt with the Qwen3-VL builder.
    2. With SGLang, vision placeholder tokens are banned during sampling. A generated <|image_pad|>
       without a real image breaks the rope index and the image embedding merge in training
       (verl does the same for vLLM already).
    """

    def __init__(self, trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config,
                 hf_model_type=None, **kwargs):
        model_type = hf_model_type.strip().lower() if isinstance(hf_model_type, str) else None
        if model_type in {"qwen3_5", "qwen3_5_moe"} and getattr(processor, "image_processor", None) is not None:
            hf_model_type = "qwen3_vl"  # only selects the token builder, the model itself is unchanged
        super().__init__(trainer_config=trainer_config, server_manager=server_manager, tokenizer=tokenizer,
                         processor=processor, dataset_cls=dataset_cls, data_config=data_config,
                         hf_model_type=hf_model_type, **kwargs)
        banned = get_vision_placeholder_token_ids(processor) if self.rollout_config.name == "sglang" else []
        self.logit_bias = {str(token_id): -1e4 for token_id in banned}

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        if self.logit_bias:
            sampling_params = {**sampling_params, "logit_bias": {**sampling_params.get("logit_bias", {}), **self.logit_bias}}
        return await super().run(sampling_params, **kwargs)
