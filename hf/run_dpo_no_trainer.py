import torch
import torch.nn as nn
import transformers
import os
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import json
from typing import Optional, Set
from transformers import AutoTokenizer

import numpy as np
import torch.nn.functional as F

import functools
import gc
from transformers import logging

from typing import Dict, Union, List, Tuple, Literal
import torch

from datetime import datetime
import os
import getpass
from transformers import set_seed
from utils import get_synthetic_data_device_iterator, get_data_device_iterator, get_cpu_memory, print_batch
import torch_xla.debug.metrics as met
from model_utils_tpu import setup_xla, setup_model_optimizer, get_global_batch_size
from input_pipeline_tpu import get_input_pipeline
from accelerate import Accelerator
from dpo_trainers import get_batch_loss_metrics
from file_utils import get_file

OmegaConf.register_new_resolver("path_join", lambda output_dir, exp_name: os.path.join(output_dir, exp_name))
OmegaConf.register_new_resolver("get_global_batch_size", lambda per_device_batch_size: get_global_batch_size(per_device_batch_size))


logger = logging.get_logger(__name__)

def clip_gradient(model, config):
    """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

def train_step(model, ref_model, train_device_loader, config, step, optimizer, scheduler, start_step, tokenizer):
    batch = next(train_device_loader)
    if step == start_step:
        print_batch(batch, tokenizer)
    optimizer.zero_grad()
    model.train()
    loss, metrics = get_batch_loss_metrics(model, ref_model, batch, "train", beta=config.beta, config=config)

    loss.backward()
    if config.max_grad_norm > 0.:
        grad_norm = clip_gradient(model, config)
        metrics['grad_norm'] = grad_norm
    optimizer.step()
    scheduler.step()
    return metrics


def eval_fn(model, ref_model, eval_device_loader, config, step):
    prefix = 'eval_'
    group_eval_metrics = {
        f"{prefix}rewards/chosen": [],
        f"{prefix}rewards/rejected": [],
        f"{prefix}rewards/accuracies": [],
        f"{prefix}rewards/margins": [],
        f"{prefix}logps/rejected": [],
        f"{prefix}logps/chosen": [],
        f"{prefix}logits/rejected": [],
        f"{prefix}logits/chosen": [],
        f"{prefix}losses": [],
        f"{prefix}num_samples": [],
        f"{prefix}ppl": [],
    }

    for eval_batch in eval_device_loader:
        model.eval()
        with torch.no_grad():
            _, eval_metrics = get_batch_loss_metrics(model, ref_model, eval_batch, "eval", beta=config.beta, config=config)
        for k in group_eval_metrics:
            group_eval_metrics[k].append(eval_metrics[k])

    for k, v in group_eval_metrics.items():
        # ppl is per token metrics which was averged
        if k == f"{prefix}ppl":
            group_eval_metrics[k] = sum(v) / len(v)
        else:
            group_eval_metrics[k] = sum(v)

    for k, v in group_eval_metrics.items():
        if k not in (f"{prefix}num_samples", f"{prefix}ppl"):
            group_eval_metrics[k] /= group_eval_metrics[f'{prefix}num_samples']
    group_eval_metrics['trained_examples'] = step * 
    return group_eval_metrics 

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    set_seed(config.seed)
    logger.info("\n\n************** Experiment configuration ***********")
    logger.info(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.output_path, 'config.yaml')
    with get_file(config_path) as f:
        OmegaConf.save(config, f)
    
    accelerator = Accelerator(log_with="tensorboard", project_dir=config.output_path)
    accelerator.init_trackers(config.run_name, config=config)
    setup_xla(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    model, ref_model, optimizer = setup_model_optimizer(config)

    if tokenizer.vocab_size != model.config.vocab_size:
        logger.warning(
            f"Found mismatch between {tokenizer.vocab_size=} and {model.config.vocab_size}"
        )
    train_device_loader, eval_device_loader = get_input_pipeline(config, tokenizer)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (config.warmup_steps + 1)))
    start_step = 0

    logger.info(f"cpu memory usage: {get_cpu_memory()}")
    step = start_step
    for step in np.arange(start_step, config.max_steps):
        if step == start_step and config.do_first_eval or step > start_step and step % config.eval_frequency == 0:
            eval_metrics = eval_fn(model, ref_model, eval_device_loader, config, step)
            accelerator.log({"eval": eval_metrics}, step=step)
        try:
            train_metrics = train_step(model, ref_model, train_device_loader, config, step, optimizer, scheduler, start_step, tokenizer)
            accelerator.log({"train": train_metrics}, step=step)
        except StopIteration:
            break


if __name__ == '__main__':
    main()