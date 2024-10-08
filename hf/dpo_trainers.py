import torch
import torch.nn as nn
from typing import Tuple, Union, List, Dict, Literal
from omegaconf import DictConfig
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
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
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios

    # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
    # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
    # calculates a conservative DPO loss.
    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing)
        - F.logsigmoid(-beta * logits) * label_smoothing
    )

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = (
        beta * (policy_rejected_logps - reference_rejected_logps).detach()
    )

    return losses, chosen_rewards, rejected_rewards


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = -100,
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "Logits (batch and sequence length dim) and labels must have the same shape."
        )

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels = labels.masked_fill(labels == label_pad_token_id, 0)

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)


def cross_entropy_loss(logits, labels, pad_token_id=0):
    # Flatten the tokens
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = loss_fct(logits, labels)
    return loss


def create_concatenated_batch(batch: Dict[str, Union[List, torch.LongTensor]]):
    # all items in batch are the same in length
    concatenated_batch = {}
    concatenated_batch["concatenated_input_ids"] = torch.cat(
        (batch["chosen_input_ids"], batch["rejected_input_ids"]), dim=0
    )
    concatenated_batch["concatenated_attention_mask"] = torch.cat(
        (batch["chosen_attention_mask"], batch["rejected_attention_mask"]), dim=0
    )
    concatenated_batch["concatenated_labels"] = torch.cat(
        (batch["chosen_labels"], batch["rejected_labels"]), dim=0
    )
    return concatenated_batch


def concatenated_forward(
    model: nn.Module,
    batch: Dict[str, Union[List, torch.LongTensor]],
    label_pad_token_id: int = -100,
    pad_token_id: int = 0,
) -> Tuple[
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
    torch.FloatTensor,
]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    concatenated_batch = create_concatenated_batch(batch)
    all_logits = model(
        concatenated_batch["concatenated_input_ids"],
        attention_mask=concatenated_batch["concatenated_attention_mask"],
        use_cache=False,
    ).logits.to(torch.float32)

    all_logps, size_completion = get_batch_logps(
        all_logits,
        concatenated_batch["concatenated_labels"],
        label_pad_token_id=label_pad_token_id,
    )

    # use torch.chunk to avoid copy, each chunk is a view of input tensor
    chosen_logits, rejected_logits = all_logits.chunk(2, axis=0)
    chosen_logps, rejected_logps = all_logps.chunk(2, axis=0)

    # from https://arxiv.org/pdf/2404.19733
    # Note that the NLL term is normalized by the total response length
    nll_loss = cross_entropy_loss(
        chosen_logits, batch["chosen_labels"], pad_token_id
    )
    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)


def get_batch_loss_metrics(
    model,
    ref_model,
    batch: Dict[str, Union[List, torch.LongTensor]],
    train_eval: Literal["train", "eval"] = "train",
    label_pad_token_id: int = -100,
    pad_token_id: int = 0,
    beta: float = 0.1,
    config: DictConfig = None,
):
    """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
    metrics = {}

    (
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
        policy_chosen_nll_loss,
    ) = concatenated_forward(model, batch, label_pad_token_id, pad_token_id)

    with torch.no_grad():
        (
            reference_chosen_logps,
            reference_rejected_logps,
            _,
            _,
            _,
        ) = concatenated_forward(ref_model, batch, label_pad_token_id, pad_token_id)

    losses, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )
    reward_accuracies = (chosen_rewards > rejected_rewards).float()

    prefix = "eval/" if train_eval == "eval" else "train/"
    num_examples = batch["chosen_input_ids"].shape[0]
    # TODO
    # mute metrics now since it trigger recompile in pytorch xla
    metrics[f"{prefix}rewards/chosen_per_example"] = chosen_rewards.mean()
    metrics[f"{prefix}rewards/rejected_per_example"] = rejected_rewards.mean()
    metrics[f"{prefix}rewards/accuracies_per_example"] = reward_accuracies.mean()
    metrics[f"{prefix}rewards/margins_per_example"] = (
        chosen_rewards - rejected_rewards
    ).mean()
    metrics[f"{prefix}logps/rejected_per_example"] = (
        policy_rejected_logps.detach().mean()
    )
    metrics[f"{prefix}logps/chosen_per_example"] = policy_chosen_logps.detach().mean()
    metrics[f"{prefix}logits/rejected_per_example"] = (
        policy_rejected_logits.detach().mean()
    )
    metrics[f"{prefix}logits/chosen_per_example"] = policy_chosen_logits.detach().mean()
    metrics[f"{prefix}losses_per_example"] = losses.detach().mean()
    metrics[f"{prefix}num_examples"] = num_examples
    metrics[f"{prefix}ppl_per_token"] = torch.exp(policy_chosen_nll_loss.detach())

    if config.aux_nll_loss_coef > 0:
        losses += config.aux_nll_loss_coef * policy_chosen_nll_loss

    return losses.mean(), metrics
