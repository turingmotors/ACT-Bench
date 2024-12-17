import torch
from vllm import LLM, ModelRegistry

from vllm_impl.modeling_llama_action import LlamaActionForCausalLM


if __name__ == "__main__":
    ModelRegistry.register_model("LlamaActionForCausalLM", LlamaActionForCausalLM)
    device = torch.device("cuda:0")

    model = LLM(
        model="/data/models/Terra-v1",
        skip_tokenizer_init=True,
        enforce_eager=True,
        trust_remote_code=True,
        max_num_seqs=5,
        device=device,
        gpu_memory_utilization=0.5,
    )