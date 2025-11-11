# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "RL_REPLACEMENTS"
]

import torch
import inspect
import os
import numpy as np
from typing import Union, Callable, Optional, List, Dict
from .device_type import DEVICE_TYPE
from .temporary_patches.common import torch_compile_options
RL_REPLACEMENTS = dict()

# https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L1674
@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def selective_log_softmax(logits, index):
    logits = logits.to(torch.float32)
    selected_logits = torch.gather(logits, dim = -1, index = index.unsqueeze(-1)).squeeze(-1)
    # loop to reduce peak mem consumption
    # logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
    logsumexp_values = torch.logsumexp(logits, dim = -1)
    per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    return per_token_logps
pass

# More memory efficient by chunking on (bsz+qlen) dimension
# Exactly equivalent to the above
@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def chunked_hidden_states_selective_log_softmax(
    hidden_states, 
    lm_head, 
    index, 
    chunks=4,
    logit_scale_multiply=0.0, 
    logit_scale_divide=0.0, 
    logit_softcapping=0.0, 
    temperature=1.0
):
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1]) 
    flat_index = index.reshape(-1)                                    

    chunked_hidden_states = torch.chunk(flat_hidden_states, chunks=chunks, dim=0)
    chunked_index = torch.chunk(flat_index, chunks=chunks, dim=0)
    
    all_per_token_logps = []
    
    for chunk_hidden_states, chunk_index in zip(chunked_hidden_states, chunked_index):
        chunk_logits = chunk_hidden_states.to(lm_head.dtype) @ lm_head.t()

        if logit_scale_multiply != 0.0:
            chunk_logits = chunk_logits * logit_scale_multiply
        if logit_scale_divide != 0.0:
            chunk_logits = chunk_logits / logit_scale_divide
        if logit_softcapping != 0.0:
            chunk_logits = chunk_logits * torch.tanh(chunk_logits / logit_softcapping)

        chunk_logits = chunk_logits.to(torch.float32)

        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature

        selected_logits = torch.gather(chunk_logits, dim=-1, index=chunk_index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(chunk_logits, dim=-1)
        per_token_logps = selected_logits - logsumexp_values
        all_per_token_logps.append(per_token_logps)
    
    all_per_token_logps = torch.concat(all_per_token_logps)
    
    all_per_token_logps = all_per_token_logps.reshape((hidden_states.shape[0], hidden_states.shape[1]))
    return all_per_token_logps

RL_REPLACEMENTS["selective_log_softmax"] = chunked_hidden_states_selective_log_softmax



def calculate_pad_tokens_in_prompt(
    input_ids: torch.Tensor,
    logits_to_keep: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Given prompt tensor, it returns all the left padded tokens in that sequence. so [pad, pad, pad, cat] = 3 tokens 
    """
    if logits_to_keep >= input_ids.shape[1]:
        raise ValueError("logits_to_keep must be smaller than the sequence length.")

    prompt_section = input_ids[:, :-logits_to_keep]

    padding_mask = (prompt_section == pad_token_id)

    pad_token_counts = padding_mask.sum(dim=1)

    return pad_token_counts
pass
RL_REPLACEMENTS["calculate_pad_tokens_in_prompt"] = calculate_pad_tokens_in_prompt


def create_completion_attention_mask(
    completion_input_ids: torch.Tensor,
    left_pad_tokens_per_prompt: torch.Tensor,
    max_left_pad: int,
    pad_token_id: int
) -> torch.Tensor:
    """
    Given that we have a sequence, [p,p,p,c,c,c,pad,pad,pad]

    Where p are extra prompt tokens we got from slicing the torch tensor, c is completion tokens
    and pad are pad tokens, this function would make a completion mask that would 0 out the pad
    and p tokens. so in this example [0,0,0,1,1,1,0,0,0]
    """
    batch_size, completion_len = completion_input_ids.shape
    device = completion_input_ids.device

    num_tokens_to_mask = max_left_pad - left_pad_tokens_per_prompt

    indices = torch.arange(completion_len, device=device).unsqueeze(0)
    shift_mask = indices >= num_tokens_to_mask.unsqueeze(1)

    non_padding_mask = (completion_input_ids != pad_token_id)

    final_mask = shift_mask & non_padding_mask

    return final_mask
pass
RL_REPLACEMENTS["create_completion_attention_mask"] = create_completion_attention_mask


def left_pack_padding(tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    Moves all padding tokens in each sequence of a batch to the right.
    """
    mask = (tensor != pad_id)
    # Must do stable=True since binary mark is unordered
    sorted_indices = torch.argsort(mask, dim=1, descending=True, stable=True)
    packed_tensor = torch.gather(tensor, 1, sorted_indices)
    return packed_tensor
pass
RL_REPLACEMENTS["left_pack_padding"] = left_pack_padding

import torch

def align_logprobs_with_mask(
    logprob_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Aligns a log probability tensor with a given attention mask.
    """

    device = logprob_tensor.device
    batch_size, logprob_seq_len = logprob_tensor.shape
    mask_seq_len = attention_mask.shape[1]

    padded_logprobs = torch.full(
        attention_mask.shape,
        fill_value=pad_value,
        dtype=logprob_tensor.dtype,
        device=device
    )

    left_pad_counts = torch.argmax(attention_mask, dim=1)

    cols = torch.arange(logprob_seq_len, device=device)


    dest_indices = left_pad_counts.unsqueeze(1) + cols

    # Create destination row indices
    # Shape: [batch_size, logprob_seq_len]
    row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(dest_indices)

    # --- 4. Filter out-of-bounds indices and perform assignment ---
    # Create a mask to identify only the indices that are within the bounds
    # of the target tensor's sequence length.
    valid_mask = dest_indices < mask_seq_len

    # Use this mask to select only the valid row indices, column indices,
    # and the corresponding values from the logprob tensor.
    # This flattens the selected elements into 1D tensors.
    valid_rows = row_indices[valid_mask]
    valid_cols = dest_indices[valid_mask]
    valid_vals = logprob_tensor[valid_mask]

    # Place the valid values into their correct positions in the padded tensor
    # using a single, efficient advanced indexing operation.
    padded_logprobs[valid_rows, valid_cols] = valid_vals

    return padded_logprobs

RL_REPLACEMENTS["align_logprobs_with_mask"] = align_logprobs_with_mask


# Custom compiled GRPO loss - creates 3 Triton kernels
def grpo_compute_loss(
    ref,
    new,
    old,
    sampling_per_token_logps,
    input_ids,
    mask,
    beta,
    advantages,
    **kwargs
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Set defaults for optional arguments
    loss_type = kwargs.get("loss_type", "grpo")
    epsilon_low = kwargs.get("epsilon_low", 0.2)
    epsilon_high = kwargs.get("epsilon_high", 0.2)
    max_completion_length = kwargs.get("max_completion_length", 8192)
    delta = kwargs.get("delta", None)
    temperature = kwargs.get("temperature", 1.0)
    logit_scale_multiply = kwargs.get("logit_scale_multiply", 0.0)
    logit_scale_divide   = kwargs.get("logit_scale_divide", 0.0)
    logit_softcapping    = kwargs.get("logit_softcapping", 0.0)
    importance_sampling_level = kwargs.get("importance_sampling_level", "token")
    num_items_in_batch = kwargs.get("num_items_in_batch", None)
    current_gradient_accumulation_steps = kwargs.get("current_gradient_accumulation_steps", 1)
    num_processes = kwargs.get("num_processes", 1)
    use_vllm = kwargs.get("use_vllm", False)
    vllm_importance_sampling_cap = kwargs.get("vllm_importance_sampling_cap", 2.0)
    input_ids = input_ids.unsqueeze(-1)

    with torch.no_grad():
        if use_vllm and sampling_per_token_logps is not None:
            #must filter out extra prompt tokens in begining after making input_ids left padded
            importance_sampling_ratio = torch.exp((old * mask) - sampling_per_token_logps)
            importance_sampling_ratio = torch.clamp(
                importance_sampling_ratio, max=vllm_importance_sampling_cap
            )
        pass
    # pass
    
    # Reverse KL
    # Note that this is a low variance low bias estimator for the KL divergence as used in GRPO paper
    if beta != 0.0:
        kl_i = torch.exp(ref - new) - (ref - new) - 1.0

    else:
        # set kl_i to a tensor of zeros with the correct shape
        if importance_sampling_level == "sequence":
            kl_i = new.new_zeros(new.size(0), 1)
        else:
            kl_i = torch.zeros_like(new)
    # Full correct reverse KL divergence?? Missing term maybe?
    # kl_i = torch.exp(new) * kl_i

    # Below is forward KL (normal KL)
    # kl_i = torch.exp(old) * (old - new)
    if old is not None: 
        log_ratio = new - old
    else:
        log_ratio = new - new.detach()

    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
    else:
        raise ValueError(
            f"Unknown importance sampling level: {importance_sampling_level}. Possible values are 'token' "
            "and 'sequence'."
        )

    coef_1 =  torch.exp(log_importance_weights)

    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

    if delta is not None:
        loss_1 = torch.clamp(coef_1, max=delta) * advantages.unsqueeze(1)
    else:
        loss_1 = coef_1 * advantages.unsqueeze(1)
    pass

    # Must detach - otherwise gradients are not propagated correctly!
    # exp(x - x) == 1
    # loss_i = torch.exp(new - new.detach()) * advantages.unsqueeze(1)

    loss_2 = coef_2 * advantages.unsqueeze(1)
    loss_i = -torch.min(loss_1, loss_2)

    if use_vllm and sampling_per_token_logps is not None:
        loss_i = loss_i * importance_sampling_ratio     
        #delta for metric
        with torch.no_grad():
            delta = torch.abs(old - sampling_per_token_logps)
            delta = delta * mask
            flat_is_ratio = importance_sampling_ratio * mask
    else:
        delta = torch.tensor([]).detach()
        flat_is_ratio = torch.tensor([]).detach()
    if beta != 0.0:
        loss_i = loss_i + beta * kl_i
    
    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)

    # https://github.com/huggingface/trl/blob/e8b8499f1f8d76838155b515e414ee98f757d6d5/trl/trainer/grpo_trainer.py#L1624
    if loss_type == "grpo":
        loss = ((loss_i * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "bnpo":
        loss = (loss_i * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "dr_grpo":
        loss = (loss_i * mask).sum() / (loss_i.size(0) * max_completion_length)
        loss = loss / current_gradient_accumulation_steps
    elif loss_type == "dapo":
        normalizer = num_items_in_batch/ num_processes
        loss = (loss_i * mask).sum() / normalizer
    else: 
        raise ValueError(f"Unknown loss type: {loss_type}")

    # loss = (loss_i * mask).sum() / mask.sum()

    # Get metrics as well which are folded
    def masked_batch_mean(x):
        with torch.inference_mode():
            completion_length = n_mask_per_reward.mean()
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return completion_length, x.mean()
            else:
                mean_kl_per_reward = (x * mask).sum(1) / n_mask_per_reward
                mean_kl = mean_kl_per_reward.mean()
                return completion_length, mean_kl
    completion_length, mean_kl = masked_batch_mean(kl_i)
    return loss, completion_length, mean_kl, delta, flat_is_ratio
pass
RL_REPLACEMENTS["grpo_compute_loss"]      = grpo_compute_loss
RL_REPLACEMENTS["grpo_compute_loss_slow"] = \
    f"@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options)\n"\
    f"{inspect.getsource(grpo_compute_loss)}"
RL_REPLACEMENTS["grpo_compute_loss_slow"] = \
    RL_REPLACEMENTS["grpo_compute_loss_slow"].replace(
        "def grpo_compute_loss",
        "def grpo_compute_loss_slow",
)

import torch

# Assume grpo_compute_loss, torch_compile_options, and RL_REPLACEMENTS are defined elsewhere

# Unsloth's memory efficient GRPO implementation
class UnslothEfficientGRPO(torch.autograd.Function):
    # All Unsloth Zoo code licensed under LGPLv3
    @staticmethod
    def forward(
        ctx,
        _old_hidden_states,
        _sampling_per_token_logps,
        lm_head,
        _full_mask,             # ! ADDED
        _input_ids,
        _completion_input_ids,  # ! ADDED
        _mask,
        _advantages,
        model,
        autocaster,
        beta,
        logits_to_keep,
        scaler = None,
        max_left_pad = None,
        n_chunks = 1,
        ref_model = None,
        extra_kwargs = None, 
    ):
        if extra_kwargs is None:
            extra_kwargs = {}
        
        # This function is fine. 'input_ids' and 'mask' here will correctly
        # refer to completion_input_ids and completion_mask.
        def compute_loss(new, old, ref, sampling_per_token_logps, input_ids, mask, advantages, scaling):
            # unsloth_zoo/rl_replacements.py
            loss, completion_length, mean_kl, delta, flat_is_ratio = grpo_compute_loss(
                ref,
                new,
                old,
                sampling_per_token_logps,
                input_ids, #! This is the completion_input_ids
                mask,      #! This is the completion_mask
                beta,
                advantages,
                **extra_kwargs,
            )

            # Scale loss if needed for mixed precision training
            scaled_loss = loss * scaling
            # Must add .loss.detach otherwise autograd uses 2x VRAM
            return scaled_loss, (loss.detach(), completion_length, mean_kl, delta, flat_is_ratio)
        pass

        device =_old_hidden_states.device
        grad_inputs = torch.empty_like(_old_hidden_states)
        accumulated_loss                = torch.zeros(1, device = device)
        accumulated_completion_length = torch.zeros(1, device = device)
        accumulated_mean_kl           = torch.zeros(1, device = device)
        accumulated_delta             = []
        accumulated_flat_is_ratio     = []
        
        # This function signature is fine. The *caller* will pass
        # completion_input_ids_j as the 'input_ids_j' argument.
        def accumulate_chunk(
            new_hidden_states_j,
            old_hidden_states_j,
            ref_hidden_states_j,
            sampling_per_token_logps_j,
            input_ids_j, #! This will be completion_input_ids_j
            mask_j,      #! This will be completion_mask_j
            advantages_j,
            scaling,
            grad_inputs_j,
        ):
            (chunk_grad_input,), (chunk_loss, (unscaled_loss, chunk_completion_length, chunk_mean_kl, chunk_delta, chunk_flat_is_ratio)) = torch.func.grad_and_value(
                compute_loss,
                argnums = (0,),
                has_aux = True,
            )(new_hidden_states_j, old_hidden_states_j, ref_hidden_states_j, sampling_per_token_logps_j, input_ids_j, mask_j, advantages_j, scaling)
            accumulated_loss                .add_(unscaled_loss)
            accumulated_completion_length.add_(chunk_completion_length)
            accumulated_mean_kl           .add_(chunk_mean_kl)
            accumulated_delta             .append(chunk_delta)
            accumulated_flat_is_ratio     .append(chunk_flat_is_ratio)
            grad_inputs_j[:] = chunk_grad_input
        pass

        accumulate_chunk = torch.compile(
            accumulate_chunk,
            fullgraph = True,
            # [TODO] Dynamic marking causes torch.compile errors if sequence length is long
            dynamic = True,
            options = torch_compile_options,
        )
        def _get_model_logits(
            model,
            input_ids,
            full_mask,
            pixel_values,
            image_grid_thw,
            pixel_attention_mask,
            image_sizes,
            _are_pixel_values_present, # The boolean check
            logits_to_keep,
            max_left_pad,
        ):
            """
            Helper function to run a single model forward pass,
            encapsulating the if/else logic for pixel values.
            """
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
            if not _are_pixel_values_present:
                logits = model(
                    input_ids=input_ids,
                    attention_mask=full_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    pixel_attention_mask=pixel_attention_mask,
                    image_sizes=image_sizes,
                ).logits
                # Slice after, as per the original 'if' block
                logits = logits[:, -(logits_to_keep + max_left_pad + 1) :, :]
            else:
                # Pass logits_to_keep directly, as per the original 'else' block
                logits = model(
                    input_ids=input_ids,
                    attention_mask=full_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    pixel_attention_mask=pixel_attention_mask,
                    image_sizes=image_sizes,
                    logits_to_keep=logits_to_keep + 1,
                ).logits
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
            logits = logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred 
            return logits

        # def _get_model_logits(
        #     model,
        #     input_ids_j,
        #     full_mask_j,
        #     pixel_values_j,
        #     image_grid_thw_j,
        #     pixel_attention_mask_j,
        #     image_sizes_j,
        #     _are_pixel_values_present, # The boolean check
        #     logits_to_keep,
        #     max_left_pad,
        # ):
        #     """
        #     Helper function to run a single model forward pass,
        #     encapsulating the if/else logic for pixel values.
        #     """
        #     os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        #     if not _are_pixel_values_present:
        #         logits = model(
        #             input_ids=input_ids_j,
        #             attention_mask=full_mask_j,
        #             pixel_values=pixel_values_j,
        #             image_grid_thw=image_grid_thw_j,
        #             pixel_attention_mask=pixel_attention_mask_j,
        #             image_sizes=image_sizes_j,
        #         ).logits
        #         # Slice after, as per the original 'if' block
        #         logits = logits[:, -(logits_to_keep + max_left_pad + 1) :, :]
        #     else:
        #         # Pass logits_to_keep directly, as per the original 'else' block
        #         logits = model(
        #             input_ids=input_ids_j,
        #             attention_mask=full_mask_j,
        #             pixel_values=pixel_values_j,
        #             image_grid_thw=image_grid_thw_j,
        #             pixel_attention_mask=pixel_attention_mask_j,
        #             image_sizes=image_sizes_j,
        #             logits_to_keep=logits_to_keep + 1,
        #         ).logits
        #     os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
        #     logits = logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred 
        #     return logits

        # --- Main Fused Function ---

        def get_policy_and_reference_logprobs(
            # --- Model and Inputs ---
            model,
            input_ids_j,
            full_mask_j,
            pixel_values_j,
            image_grid_thw_j,
            pixel_attention_mask_j,
            image_sizes_j,
            completion_input_ids, # From the logprob calculation part
            lm_head,              # From the logprob calculation part
            
            # --- Control Parameters ---
            _are_pixel_values_present,
            logits_to_keep,
            max_left_pad,
            beta,
            
            # --- Logprob Parameters ---
            logit_scale_multiply,
            logit_scale_divide,
            logit_softcapping,
            temperature,
            inner_logprob_chunks=64
        ):
            """
            Calculates and returns the log probabilities for both the
            policy model (new) and the reference model (ref).
            """
            
            # 1. Calculate policy hidden states
            new_hidden_states_j = _get_model_logits(
                model=model,
                input_ids=input_ids_j,
                full_mask=full_mask_j,
                pixel_values=pixel_values_j,
                image_grid_thw=image_grid_thw_j,
                pixel_attention_mask=pixel_attention_mask_j,
                image_sizes=image_sizes_j,
                _are_pixel_values_present=_are_pixel_values_present,
                logits_to_keep=logits_to_keep,
                max_left_pad=max_left_pad,
            )
            # 3. Calculate logprobs for policy
            new_logprobs = chunked_hidden_states_selective_log_softmax(
                hidden_states=new_hidden_states_j,
                index=completion_input_ids,
                lm_head=lm_head,
                logit_scale_multiply=logit_scale_multiply,
                logit_scale_divide=logit_scale_divide,
                logit_softcapping=logit_softcapping,
                temperature=temperature,
                chunks=inner_logprob_chunks
            )

            # 2. Calculate reference hidden states
            ref_hidden_states_j = None
            if beta != 0.0:
                with torch.no_grad():
                    with model.disable_adapter():
                        ref_hidden_states_j = _get_model_logits(
                            model=model,
                            input_ids=input_ids_j,
                            full_mask=full_mask_j,
                            pixel_values= pixel_values_j,
                            image_grid_thw=image_grid_thw_j,
                            pixel_attention_mask=pixel_attention_mask_j,
                            image_sizes=image_sizes_j,
                            _are_pixel_values_present=_are_pixel_values_present,
                            logits_to_keep=logits_to_keep,
                            max_left_pad=max_left_pad,
                        )
                    # 4. Calculate logprobs for reference
                    ref_logprobs = chunked_hidden_states_selective_log_softmax(
                        hidden_states=ref_hidden_states_j, # Handles None if beta == 0.0
                        index=completion_input_ids,
                        lm_head=lm_head,
                        logit_scale_multiply=logit_scale_multiply,
                        logit_scale_divide=logit_scale_divide,
                        logit_softcapping=logit_softcapping,
                        temperature=temperature,
                        chunks=inner_logprob_chunks
                    )

            return new_logprobs, ref_logprobs
        pixel_values = extra_kwargs.get('pixel_values',None)
        image_grid_thw = extra_kwargs.get('image_grid_thw',None)
        pixel_attention_mask = extra_kwargs.get('pixel_attention_mask',None)
        image_sizes = extra_kwargs.get('image_sizes',None)
        logit_scale_multiply = extra_kwargs.get('logit_scale_multiply',0.0)
        logit_scale_divide = extra_kwargs.get('logit_scale_divide',0.0)
        logit_softcapping = extra_kwargs.get('logit_softcapping',0.0)
        temperature = extra_kwargs.get('temperature', 1.0)
        # Check if vision inputs are present *before* chunking
        _are_pixel_values_present = (pixel_values is not None)
        
        grad_inputs_chunks = torch.chunk(grad_inputs,       chunks = n_chunks, dim = 0)
        
        # Chunk optional vision inputs
        if pixel_values is not None:
            pixel_values= torch.chunk(pixel_values, chunks=n_chunks, dim=0)
        else:
            pixel_values = [None] * n_chunks

        if image_grid_thw is not None:
            image_grid_thws = torch.chunk(image_grid_thw, chunks=n_chunks, dim=0)
        else:
            image_grid_thws = [None] * n_chunks

        if pixel_attention_mask is not None:
            pixel_attention_masks= torch.chunk(pixel_attention_mask, chunks=n_chunks, dim=0)
        else:
            pixel_attention_masks = [None] * n_chunks

        if image_sizes is not None:
            image_sizes_chunks = torch.chunk(image_sizes, chunks=n_chunks, dim=0)
        else:
            image_sizes_chunks = [None] * n_chunks
        
        # Chunk other optional inputs
        #TODO CHANGE THIS TO LOGPROBS
        if _old_hidden_states is not None: 
            old_hidden_states  = torch.chunk(_old_hidden_states, chunks = n_chunks, dim = 0)
        else: 
            old_hidden_states = [None] * n_chunks

        if _sampling_per_token_logps is not None: 
            sampling_per_token_logps  = torch.chunk(_sampling_per_token_logps, chunks = n_chunks, dim = 0)
        else:
            sampling_per_token_logps = [None] * n_chunks
        
        # Chunk non-optional inputs
        full_mask            = torch.chunk(_full_mask,             chunks = n_chunks, dim = 0) #! ADDED
        input_ids            = torch.chunk(_input_ids,             chunks = n_chunks, dim = 0)
        completion_input_ids = torch.chunk(_completion_input_ids,  chunks = n_chunks, dim = 0) #! ADDED
        mask                 = torch.chunk(_mask,                  chunks = n_chunks, dim = 0)
        advantages           = torch.chunk(_advantages,            chunks = n_chunks, dim = 0)

        # Get mixed precision scaling if seen
        scaling = scaler.get_scale() if scaler is not None else 1.0

        # Force torch.compile to use dynamic shapes for seqlen dim
        # mark_dynamic = lambda x: torch._dynamo.mark_dynamic(x, 1)
        # breakpoint() # Left user's breakpoint
        
        #! CHANGED: Added full_mask_j, completion_input_ids_j to loop
        for (
                grad_inputs_j, old_hidden_states_j, sampling_per_token_logps_j, 
                full_mask_j, input_ids_j, completion_input_ids_j, mask_j, advantages_j, 
                pixel_values_j, image_grid_thw_j, 
                pixel_attention_mask_j, image_sizes_j
            ) in \
            zip(
                grad_inputs_chunks, old_hidden_states, sampling_per_token_logps, 
                full_mask, input_ids, completion_input_ids, mask, advantages, #! CHANGED
                pixel_values, image_grid_thws, 
                pixel_attention_masks, image_sizes_chunks
            ):

            with autocaster:
                new_logprobs, ref_logprobs = get_policy_and_reference_logprobs(
                                # --- Model and Inputs ---
                    model,
                    input_ids_j,
                    full_mask_j,
                    pixel_values_j,
                    image_grid_thw_j,
                    pixel_attention_mask_j,
                    image_sizes_j,
                    completion_input_ids_j, # From the logprob calculation part
                    lm_head,              # From the logprob calculation part
                    
                    # --- Control Parameters ---
                    _are_pixel_values_present,
                    logits_to_keep,
                    max_left_pad,
                    beta,
                    
                    # --- Logprob Parameters ---
                    logit_scale_multiply,
                    logit_scale_divide,
                    logit_softcapping,
                    temperature,
                    inner_logprob_chunks=64
                )

                # [TODO] Dynamic marking
                #! CHANGED: Pass completion_input_ids_j as the input_ids_j argument
                accumulate_chunk(
                    new_logprobs,
                    old_hidden_states_j,
                    ref_logprobs,
                    sampling_per_token_logps_j,
                    completion_input_ids_j, #! Pass completion_ids for loss
                    mask_j,                 #! Pass completion_mask for loss
                    advantages_j,
                    scaling,
                    grad_inputs_j,
                )
        pass

        grad_inputs                   .div_(n_chunks)
        accumulated_loss              .div_(n_chunks)
        accumulated_completion_length .div_(n_chunks)
        accumulated_mean_kl           .div_(n_chunks)

        if _sampling_per_token_logps is not None:
            accumulated_delta = torch.cat(accumulated_delta, dim=0)
            accumulated_flat_is_ratio = torch.cat(accumulated_flat_is_ratio, dim=0)
        else:
            accumulated_delta = None
            accumulated_flat_is_ratio = None
        
        ctx.save_for_backward(grad_inputs)
        return (
            accumulated_loss,
            accumulated_completion_length,
            accumulated_mean_kl,
            accumulated_delta,
            accumulated_flat_is_ratio
        )
    pass

    @staticmethod
    def backward(ctx, grad_output, dcompletion_length, dmean_kl, ddelta, ddflat_is_ratio):
        (grad_input,) = ctx.saved_tensors
        
        #! CHANGED: Matched None tuple to new 17 forward inputs
        # (grad_input is for _old_hidden_states)
        return (
            grad_input, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None,
        )
    pass
pass

# Assuming RL_REPLACEMENTS is a dict
# RL_REPLACEMENTS["UnslothEfficientGRPO"] = UnslothEfficientGRPO
RL_REPLACEMENTS["UnslothEfficientGRPO"] = UnslothEfficientGRPO


def grpo_accumulated_loss(
    trainer,
    input_ids,
    attention_mask,
    logits_to_keep,
    completion_mask,
    advantages,
    old_logprobs,
    n_chunks = -1,
    **kwargs,
):
    # All Unsloth Zoo code licensed under LGPLv3
    bsz, qlen = input_ids.shape

    pixel_values = kwargs.get('pixel_values',None)
    image_grid_thw = kwargs.get('image_grid_thw',None)
    pixel_attention_mask = kwargs.get('pixel_attention_mask',None)
    image_sizes = kwargs.get('image_sizes',None)
    sampling_per_token_logps = kwargs.get("sampling_per_token_logps", None)
    #delete this from kwargs so less issues 
    del kwargs["sampling_per_token_logps"]
    kwargs["vllm_importance_sampling_cap"] = trainer.vllm_importance_sampling_cap if sampling_per_token_logps is not None else None
    kwargs["use_vllm"] = trainer.use_vllm
    # Find closest multiple
    factors = [i for i in range(1, bsz + 1) if bsz % i == 0]
    if n_chunks == -1: n_chunks = bsz
    n_chunks = factors[min(np.searchsorted(factors, n_chunks), len(factors)-1)]

    if not hasattr(trainer, '_autocast_dtype'):
        trainer._autocast_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
        if os.environ.get('UNSLOTH_FORCE_FLOAT32', '0') == '1': trainer._autocast_dtype = None
    pass
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    lm_head = trainer.model.get_output_embeddings().weight

    if pixel_values is None:
        left_pad_tokens_per_prompt = calculate_pad_tokens_in_prompt(input_ids, logits_to_keep, trainer.processing_class.pad_token_id)

        max_left_pad = max(left_pad_tokens_per_prompt).item()

        input_ids = left_pack_padding(input_ids, trainer.processing_class.pad_token_id)

        completion_input_ids = input_ids[:, -(logits_to_keep +max_left_pad):]

        completion_mask = create_completion_attention_mask(completion_input_ids, left_pad_tokens_per_prompt, max_left_pad, trainer.processing_class.pad_token_id).to(attention_mask.dtype)
        #TODO given the completion mask here we need to, handle the left pad tokens so the sizes of completion
        #token or old logprobs are compatible with the importance sampling logprobs
        if trainer.use_vllm and sampling_per_token_logps is not None:
            sampling_per_token_logps = align_logprobs_with_mask(sampling_per_token_logps, completion_mask)
        attention_mask =  input_ids != trainer.processing_class.pad_token_id
        attention_mask = attention_mask.to(attention_mask.dtype)
    else: 
        completion_input_ids = input_ids[:, -logits_to_keep:]
        max_left_pad = None
    
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper = False)

    # Do not move hidden_states from device 1 to device 0:
    for module in unwrapped_model.modules():
        if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "io_same_decice"):
            module._hf_hook.io_same_decice = False
    pass


    # Get autocaster
    if trainer._autocast_dtype is None:
        autocaster = nullcontext()
    else:
        autocaster = torch.amp.autocast(device_type = trainer.model.device.type, dtype = trainer._autocast_dtype)
    if pixel_values is None and old_hidden_states is not None: 
            old_hidden_states = old_hidden_states[:, -(logits_to_keep +max_left_pad+1): , :]
    loss, completion_length, mean_kl, delta, flat_is_ratio = UnslothEfficientGRPO.apply(
        old_logprobs,
        sampling_per_token_logps,
        lm_head,
        attention_mask,
        input_ids,
        completion_input_ids,
        completion_mask,
        advantages,
        unwrapped_model, 
        autocaster,
        trainer.beta,
        logits_to_keep,
        trainer.accelerator.scaler,
        max_left_pad,
        n_chunks,
        trainer.ref_model,
        kwargs # pass kwargs as a dict
    )

    
    # Must force not returning hidden states but logits otherwise gibberish
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"

    return loss, completion_length, mean_kl, delta, flat_is_ratio
    # Old non efficient code path
    new_logits = torch.matmul(new_hidden_states, lm_head.t())
    new_logits = new_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
    old_logits = torch.matmul(old_hidden_states, lm_head.t())
    old_logits = old_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
    loss, completion_length, mean_kl = grpo_compute_loss(
        old_logits,
        new_logits,
        completion_input_ids,
        completion_mask,
        trainer.beta,
        advantages,
    )
    return loss, completion_length, mean_kl
    pass
pass
RL_REPLACEMENTS["grpo_accumulated_loss"] = grpo_accumulated_loss

from .dataset_utils import sft_prepare_dataset
RL_REPLACEMENTS["sft_prepare_dataset"] = sft_prepare_dataset

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
