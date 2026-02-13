import torch
from sglang.srt.layers.get_selected_logprob import get_selected_logprob
from sglang.srt.managers.router.model_runner import ForwardMode, InputMetadata
from sglang.srt.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from torch import nn


class LogitsProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward(self, input_ids, hidden_states, weight, input_metadata):
        # hidden_states is guaranteed to be 2D: [tokens/batch, hidden_dim]
        if not input_metadata.return_normalized_logprob:
            if input_metadata.forward_mode == ForwardMode.DECODE:
                # DECODE mode: hidden_states is [batch, hidden_dim]
                last_hidden = hidden_states
            else:
                # EXTEND mode: hidden_states is [total_tokens, hidden_dim]
                # Extract the last token of each sequence
                last_index = (
                    torch.cumsum(
                        input_metadata.seq_lens - input_metadata.prefix_lens,
                        dim=0,
                        dtype=torch.long,
                    )
                    - 1
                )
                # Clamp last_index to prevent out of bounds
                last_index = torch.clamp(last_index, 0, hidden_states.shape[0] - 1)
                last_hidden = hidden_states[last_index]
                hidden_states = None

            last_logits = torch.matmul(last_hidden, weight.T)
            if self.tp_size > 1:
                last_logits = tensor_model_parallel_all_gather(last_logits)
            last_logits = last_logits[:, : self.config.vocab_size]
            return last_logits, None
        else:
            assert input_metadata.forward_mode != ForwardMode.DECODE
            # hidden_states is [total_tokens, hidden_dim]
            last_index = (
                torch.cumsum(
                    input_metadata.seq_lens - input_metadata.prefix_lens,
                    dim=0,
                    dtype=torch.long,
                )
                - 1
            )

            logits = torch.matmul(hidden_states, weight.T)
            if self.tp_size > 1:
                logits = tensor_model_parallel_all_gather(logits)
            logits = logits[:, : self.config.vocab_size]
            all_logprobs = torch.log(torch.softmax(logits.float(), dim=-1) + 1e-6)

            normalized_logprobs = compute_normalized_logprobs(
                all_logprobs,
                input_metadata.seq_lens - input_metadata.prefix_lens,
                input_ids,
            )

            # Clamp last_index to prevent out of bounds
            last_index = torch.clamp(last_index, 0, logits.shape[0] - 1)
            last_logits = logits[last_index]
            return last_logits, normalized_logprobs


def compute_normalized_logprobs(all_logprobs, len_add_1, input_ids):
    # assert all_logprobs.shape[0] == torch.sum(len_add_1) == input_ids.shape[0]
    logprobs = torch.zeros(
        (all_logprobs.shape[0] - len_add_1.shape[0]), dtype=torch.float32, device="cuda"
    )
    get_selected_logprob(all_logprobs, len_add_1, input_ids, logprobs)
    cumsum = torch.cumsum(logprobs, dim=0, dtype=torch.float32)
    end = torch.cumsum(len_add_1.sub_(1), dim=0)
    start = torch.cat((torch.tensor([0], device="cuda"), end[:-1]), 0)
    end.sub_(1)
    sum_logp = cumsum[end] - cumsum[start] + logprobs[start]
    res = sum_logp / len_add_1
    return res
