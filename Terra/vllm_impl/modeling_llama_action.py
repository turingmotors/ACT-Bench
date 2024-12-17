"""Code Reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py"""

from array import array
from typing import Iterable, List, Optional, Tuple, Union, Mapping

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.distributed import get_pp_group
from vllm.inputs import InputContext, INPUT_REGISTRY
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs, MultiModalPlugin
from vllm.sequence import IntermediateTensors, SequenceData

from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer


class ActionsPlugin(MultiModalPlugin):
    def get_data_key(self) -> str:
        return "actions"

    def _default_input_mapper(self, ctx: InputContext, data: object | List[object], **mm_processor_kwargs) -> MultiModalInputs:
        return MultiModalInputs({"actions": data})

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        return 4096


MULTIMODAL_REGISTRY.register_plugin(ActionsPlugin())


class LearnableFactorizedSpatioTemporalPositionalEmbedding(nn.Module):
    def __init__(self, num_spatio_embeddings: int, num_temporal_embeddings: int, embedding_dim: int):
        super().__init__()
        self.spatio_embeddings = nn.Embedding(num_spatio_embeddings, embedding_dim)
        self.temporal_embeddings = nn.Embedding(num_temporal_embeddings, embedding_dim)
        self.num_spatio_embeddings = num_spatio_embeddings
        self.num_temporal_embeddings = num_temporal_embeddings

    def forward(self, positions: torch.Tensor):
        spatio_indices = positions % self.num_spatio_embeddings
        temporal_indices = positions // self.num_spatio_embeddings
        return self.spatio_embeddings(spatio_indices) + self.temporal_embeddings(temporal_indices)


HF_LlamaActionConfig = AutoConfig.from_pretrained("turing-motors/Terra", subfolder="world_model", trust_remote_code=True)
LlamaActionConfig = HF_LlamaActionConfig.__class__


def get_max_action_tokens(ctx: InputContext):
    hf_config = ctx.get_hf_config(LlamaActionConfig)
    num_action_tokens = hf_config.num_action_embeddings
    num_frames = hf_config.num_temporal_embeddings - 1
    return num_action_tokens * num_frames


def create_dummy_data(ctx: InputContext, seq_len: int, mm_counts: Mapping[str, int]):
    hf_config = ctx.get_hf_config(LlamaActionConfig)

    num_frames = hf_config.num_temporal_embeddings
    vocab_size = hf_config.vocab_size
    num_action_tokens = hf_config.num_action_embeddings
    num_image_tokens = hf_config.num_image_patches
    dummy_seq = []
    np.random.seed(0)
    for i in range(num_frames - 1):
        dummy_image_tokens = np.random.randint(0, vocab_size, num_image_tokens).tolist()
        dummy_seq.extend(dummy_image_tokens)
        dummy_action_tokens = [-3] * num_action_tokens
        dummy_seq.extend(dummy_action_tokens)
    seq_data = SequenceData(array("l", dummy_seq))

    action = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.5],
        [0.0, 4.0, 1.0],
        [0.0, 6.0, 1.5],
        [0.0, 8.0, 2.0],
        [0.0, 10.0, 2.5],
        [0.0, 12.0, 3.0],
        [0.0, 14.0, 3.5],
        [0.0, 16.0, 4.0],
    ])
    actions = []
    for _ in range(num_frames - 1):
        actions.append(action[:num_action_tokens])
    actions = torch.cat(actions, dim=0)
    mm_data = {"actions": actions}
    return seq_data, mm_data


@MULTIMODAL_REGISTRY.register_input_mapper(data_type_key="actions")
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens("actions", get_max_action_tokens)
@INPUT_REGISTRY.register_dummy_data(create_dummy_data)
class LlamaActionForCausalLM(nn.Module, SupportsMultiModal):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm"
    }

    def __init__(
        self,
        config,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.multimodal_config = multimodal_config

        self.num_spatio_embeddings = config.num_spatio_embeddings
        self.num_temporal_embeddings = config.num_temporal_embeddings
        self.num_image_patches = config.num_image_patches
        self.num_action_embeddings = config.num_action_embeddings

        self.pos_embedding_spatio_temporal = LearnableFactorizedSpatioTemporalPositionalEmbedding(
            num_spatio_embeddings=self.num_spatio_embeddings,
            num_temporal_embeddings=self.num_temporal_embeddings,
            embedding_dim=config.hidden_size,
        )

        self.action_projection = nn.Linear(config.action_dim, config.hidden_size)

        self.model = LlamaModel(config,
                                cache_config,
                                quant_config,
                                lora_config=None,
                                prefix="model")
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                quant_config=quant_config,
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
            self.sampler = Sampler()
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Forward pass for the model.
        input_ids already accounts for the positions of the to-be-inserted action embeddings.
    
        action tokens are represetnted by -3.
        example: [1287, 3342, ..., 6571, -3, ..., -3]
        """
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        else:
            action_token_indices = (input_ids == -3).nonzero(as_tuple=True)[0]
            image_token_indices = (input_ids > 0).nonzero(as_tuple=True)[0]

            image_tokens = input_ids[image_token_indices]
            image_token_embeddings = self.model.get_input_embeddings(image_tokens)

            inputs_embeds = torch.zeros(
                (input_ids.size(0), image_token_embeddings.size(1)), 
                device=input_ids.device, dtype=image_token_embeddings.dtype
            )
            inputs_embeds[image_token_indices] = image_token_embeddings

            actions = kwargs.pop("actions", None)
            if actions is not None:
                assert len(action_token_indices) == actions.size(0) * actions.size(1), "actions must have the same length as the number of action tokens"
                actions = actions.to(dtype=self.action_projection.weight.dtype)
                action_embeddings = self.action_projection(actions)
                inputs_embeds[action_token_indices] = action_embeddings.view(-1, action_embeddings.size(-1))
            input_ids = None
            inputs_embeds += self.pos_embedding_spatio_temporal(positions)
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:
        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        if "wk" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)
        elif "wq" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        for item in modules:
            if item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights)
