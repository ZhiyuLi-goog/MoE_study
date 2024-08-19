import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import default_data_collator

import os
import hydra
from omegaconf import OmegaConf, DictConfig
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

import numpy as np
import torch.nn.functional as F

import gc
from transformers import logging

from typing import Dict, Union, List, Tuple, Literal
import torch

from datetime import datetime
import os
import getpass
from transformers import set_seed

from accelerate import Accelerator
from accelerate import FullyShardedDataParallelPlugin as FSDPPlugin
from accelerate.state import AcceleratorState
from accelerate.utils import FP8RecipeKwargs, set_seed
from accelerate.utils.transformer_engine import convert_model

from utils_gpu import get_data_device_iterator

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dir: get_local_run_dir(exp_name, local_dir))
logger = logging.get_logger(__name__)

WORLD_SIZE = torch.cuda.device_count()

def setup(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=WORLD_SIZE)

def cleanup():
    dist.destroy_process_group()

def get_local_dir(prefix: str) -> str:
    """Return the path to the cache directory for this user."""
    if os.path.exists(prefix):
        return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"
    
def dpo_loss(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
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
        if loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing)
                - F.logsigmoid(-beta * logits) * label_smoothing
            )
        elif loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * beta)) ** 2

        chosen_rewards = (
            beta
            * (
                policy_chosen_logps - reference_chosen_logps
            ).detach()
        )
        rejected_rewards = (
            beta
            * (
                policy_rejected_logps
                - reference_rejected_logps
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

def get_local_run_dir(exp_name: str, local_dir: str) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dir)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask

def report_metrics(step, loss, metrics):
    logger.info(f'{step=}, {loss=}, {metrics=}')

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
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels = labels.masked_fill(labels == label_pad_token_id, 0)

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)


def cross_entropy_loss(logits, labels):
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss()
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = loss_fct(logits, labels)
    return loss


def forward(
        model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], label_pad_token_id: int = -100,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    chosen_logits = model(
        batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
        use_cache=False,
    ).logits
    rejected_logits = model(
        batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
        use_cache=False,
    ).logits
    chosen_labels = batch["chosen_labels"].clone()
    rejected_labels = batch["rejected_labels"].clone()

    chosen_logps, size_completion = get_batch_logps(chosen_logits, chosen_labels, label_pad_token_id=label_pad_token_id)
    rejected_logps, size_completion = get_batch_logps(rejected_logits, rejected_labels, label_pad_token_id=label_pad_token_id)
    nll_loss = cross_entropy_loss(chosen_logits, chosen_labels)
    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)

def create_concatenated_batch(batch: Dict[str, Union[List, torch.LongTensor]], device: torch.device = None):
    # all items in batch are the same in length
    concatenated_batch = {}
    concatenated_batch["concatenated_input_ids"] = torch.cat((batch["chosen_input_ids"], batch["rejected_input_ids"]), dim=0).to(device=device)
    concatenated_batch["concatenated_attention_mask"] = torch.cat((batch["chosen_attention_mask"], batch["rejected_attention_mask"]), dim=0).to(device=device)
    concatenated_batch["concatenated_labels"] = torch.cat((batch["chosen_labels"], batch["rejected_labels"]), dim=0).to(device=device)
    return concatenated_batch

def concatenated_forward(
        model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], label_pad_token_id: int = -100, device: torch.device = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    """
    concatenated_batch = create_concatenated_batch(batch, device)
    len_chosen = concatenated_batch["concatenated_input_ids"].shape[0] // 2
    all_logits = model(
        concatenated_batch["concatenated_input_ids"],
        attention_mask=concatenated_batch["concatenated_attention_mask"],
        use_cache=False,
    ).logits

    all_logps, size_completion = get_batch_logps(
        all_logits,
        concatenated_batch["concatenated_labels"],
        label_pad_token_id=label_pad_token_id,
    )

    labels = concatenated_batch["concatenated_labels"].clone()
    nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

    chosen_logps = all_logps[:len_chosen]
    rejected_logps = all_logps[len_chosen:]

    chosen_logits = all_logits[:len_chosen]
    rejected_logits = all_logits[len_chosen:]

    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)


def get_batch_loss_metrics(
        model,
        ref_model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
        label_pad_token_id: int = -100,
        beta: float = 0.1,
        config: DictConfig = None,
        device: torch.device = None,
    ):
    """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
    metrics = {}

    if config.concatenated_forward:
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = concatenated_forward(model, batch, label_pad_token_id, device)
    else:
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = forward(model, batch, label_pad_token_id)

    with torch.no_grad():
        if config.concatenated_forward:
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
                _,
            ) = concatenated_forward(ref_model, batch, label_pad_token_id, device)
        else:
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
                _,
            ) = forward(ref_model, batch, label_pad_token_id)

    losses, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )
    reward_accuracies = (chosen_rewards > rejected_rewards).float()

    prefix = "eval_" if train_eval == "eval" else ""
    # TODO
    # mute metrics now since it trigger recompile in pytorch xla
    metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean()
    metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean()
    metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean()
    metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
    metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean()
    metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean()
    metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean()
    metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean()

    return losses.mean(), metrics


def prepare_model(model, config):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
    )
    model = FSDP(
        model,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        auto_wrap_policy=size_based_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True)
    )
    
    return model


def clip_gradient(model, config):
    """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #device = torch.device("cuda", local_rank)

    accelerator = Accelerator()
    device = accelerator.device
    
    #torch.cuda.set_device(device)

    OmegaConf.resolve(config)
    set_seed(config.seed)

    logger.info("\n\n************** Experiment configuration ***********")
    logger.info(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    model_torch_dtype = getattr(torch, config.model.torch_dtype)

    logger.info("loading model")
    if config.model.config_path:
        model_config = AutoConfig.from_pretrained(config.model.config_path)
        model_config.static = True
        model_config.flash_attention = True
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(model_config).to_empty(device=device).to(model_torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=config.cache_local_dir, low_cpu_mem_usage=True, torch_dtype=model_torch_dtype)
    
    logger.info("loading tokenzier")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    logger.info("tokenzier loaded")

    logger.info("model loaded")
    model = prepare_model(model, config)
    logger.info("model prepared")

    gc.collect()

    logger.info("loading ref_model")
    if config.model.config_path:
        with torch.device("meta"):
            ref_model = AutoModelForCausalLM.from_config(model_config).to_empty(device=device).to(model_torch_dtype)
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=config.cache_local_dir, low_cpu_mem_usage=True, torch_dtype=model_torch_dtype)
    logger.info("ref_model loaded")
    ref_model.eval()
    ref_model = prepare_model(ref_model, config)

    logger.info("ref_model prepared")
    gc.collect()

    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (config.warmup_steps + 1)))
    
    ds = get_data_device_iterator(config, tokenizer)
    train_sampler = DistributedSampler(ds['train'])
    train_device_loader = DataLoader(ds['train'], drop_last=True, collate_fn=default_data_collator, sampler=train_sampler)

    #model, optimizer, train_device_loader = accelerator.prepare(model, optimizer, train_device_loader)

    start_step = 0
    for step in np.arange(start_step, config.max_steps):
        _, batch = next(enumerate(train_device_loader))

        loss, metrics = get_batch_loss_metrics(model, ref_model, batch, "train", beta=config.beta, config=config, device=device)

        loss.backward()
        #accelerator.backward(loss)
        # grad_norm = clip_gradient(model, config)
        # metrics['grad_norm'] = grad_norm
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if local_rank == 0 and step > start_step and step % config.report_metrics_freq == 0:
            report_metrics(step, loss, metrics)

    cleanup()


if __name__ == '__main__':
    main()