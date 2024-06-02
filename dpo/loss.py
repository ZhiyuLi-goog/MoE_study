"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import datetime
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from accelerate import Accelerator
from torch.profiler import profile, record_function, ProfilerActivity


# 1. reference implementation from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L45
def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


# 2. huggingface implementation from https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py#L977C1-L1086C56
def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    device,
    loss_type: str = "sigmoid",
    beta: float = 1.,
    label_smoothing: float = 0.,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    # if self.reference_free:
    if False:
        ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
    else:
        ref_logratios = reference_chosen_logps - reference_rejected_logps

    pi_logratios = pi_logratios.to(device)
    ref_logratios = ref_logratios.to(device)
    logits = pi_logratios - ref_logratios

    # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
    # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
    # calculates a conservative DPO loss.
    if loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(beta * logits) * label_smoothing
        )
#     elif loss_type == "robust":
#         losses = (
#             -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
#             + F.logsigmoid(-self.beta * logits) * self.label_smoothing
#         ) / (1 - 2 * self.label_smoothing)
#     elif loss_type == "hinge":
#         losses = torch.relu(1 - self.beta * logits)
#     elif loss_type == "ipo":
#         # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
#         losses = (logits - 1 / (2 * self.beta)) ** 2
#     elif loss_type == "kto_pair":
#         # eqn (7) of the HALOs paper
#         chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
#         rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)
# 
#         chosen_logratios = policy_chosen_logps - reference_chosen_logps
#         rejected_logratios = policy_rejected_logps - reference_rejected_logps
#         # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
#         losses = torch.cat(
#             (
#                 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
#                 1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
#             ),
#             0,
#         )
#     elif loss_type == "bco_pair":
#         chosen_logratios = policy_chosen_logps - reference_chosen_logps
#         rejected_logratios = policy_rejected_logps - reference_rejected_logps
# 
#         chosen_rewards = beta * chosen_logratios
#         rejected_rewards = beta * rejected_logratios
#         rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
#         self.running.update(rewards)
#         delta = self.running.mean
# 
#         losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
#             -(self.beta * rejected_logratios - delta)
#         )
#     elif self.loss_type == "sppo_hard":
#         # In the paper (https://arxiv.org/pdf/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
#         a = policy_chosen_logps - reference_chosen_logps
#         b = policy_rejected_logps - reference_rejected_logps
# 
#         losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
#     elif self.loss_type == "nca_pair":
#         chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
#         rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
#         losses = (
#             -F.logsigmoid(chosen_rewards)
#             - 0.5 * F.logsigmoid(-chosen_rewards)
#             - 0.5 * F.logsigmoid(-rejected_rewards)
#         )
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'bco_pair', 'sppo_hard', 'nca_pair', 'robust']"
        )

    chosen_rewards = (
        beta
        * (
            policy_chosen_logps.to(device) - reference_chosen_logps.to(device)
        ).detach()
    )
    rejected_rewards = (
        beta
        * (
            policy_rejected_logps.to(device)
            - reference_rejected_logps.to(device)
        ).detach()
    )

    return losses, chosen_rewards, rejected_rewards


# 3.NeMo-aligner: https://github.com/NVIDIA/NeMo-Aligner/blob/5b898b67ab08ed68e2b2b89bf898db233e71eb68/nemo_aligner/models/nlp/gpt/megatron_gpt_dpo_model.py#L193
def get_reduced_masked_logps(logps, labels, average_log_probs=False):
    assert logps.shape == labels.shape, "logps and labels shape mismatch"

    loss_mask = (labels > -1).float()

    if average_log_probs:
        # need to guard against divide by zero in case labels are all -100
        return (logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
    else:
        return (logps * loss_mask).sum(-1)

def loss_func(pi_logprobs, ref_logprobs, labels, average_log_probs=False, ref_policy_kl_penalty: float = 0.):
    rewards = get_reduced_masked_logps(
        pi_logprobs - ref_logprobs, labels, average_log_probs=average_log_probs
    )
    chosen_rewards, reject_rewards = split_output_tensor(ref_policy_kl_penalty * rewards)

    loss = -torch.nn.functional.logsigmoid(chosen_rewards - reject_rewards).mean(0)

    with torch.no_grad():
        comp = chosen_rewards > reject_rewards
        acc_chosen = comp.float().mean()

    return loss, acc_chosen


def split_output_tensor(output_tensor):
    chosen_logps, reject_logps = torch.split(output_tensor.float(), len(output_tensor) // 2, dim=0)
    return chosen_logps, reject_logps


def nemo_dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    ):
    pi_logprobs = torch.cat(
        (policy_chosen_logps, policy_rejected_logps), dim=0
    )
    ref_logprobs = torch.cat(
        (reference_chosen_logps, reference_rejected_logps), dim=0
    )
    return loss_func(pi_logprobs, ref_logprobs, torch.zeros_like(pi_logprobs))

if __name__ == "__main__":
    accelerator = Accelerator()
    device = accelerator.device
    batch_size = 1
    seq_length = 65536
    TP = 2
    vocab_size = 32000 // TP

    print(f"{device=}")
    batch = {
        "policy_chosen_logps": torch.randn(batch_size, seq_length, vocab_size, dtype=torch.float32).to(device),
        "policy_rejected_logps": torch.randn(batch_size, seq_length, vocab_size, dtype=torch.float32).to(device),
        "reference_chosen_logps": torch.randn(batch_size, seq_length, vocab_size, dtype=torch.float32).to(device),
        "reference_rejected_logps": torch.randn(batch_size, seq_length, vocab_size, dtype=torch.float32).to(device),
    }
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("reference_dpo"):
            preference_loss(batch["policy_chosen_logps"], batch["policy_rejected_logps"], batch["reference_chosen_logps"], batch["reference_rejected_logps"], 1.)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("hf_dpo"):
            dpo_loss(batch["policy_chosen_logps"], batch["policy_rejected_logps"], batch["reference_chosen_logps"], batch["reference_rejected_logps"], accelerator.device)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("nemo_dpo_loss"):
            nemo_dpo_loss(batch["policy_chosen_logps"], batch["policy_rejected_logps"], batch["reference_chosen_logps"], batch["reference_rejected_logps"])

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))