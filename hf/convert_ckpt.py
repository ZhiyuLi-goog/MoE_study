import torch
import torch.nn as nn
import transformers
import os
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import json
from typing import Optional, Set
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import copy

import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

import numpy as np
import torch.nn.functional as F

import torch_xla

import functools
import gc
from transformers import logging

from typing import Dict, Union, List, Tuple, Literal
import torch

from datetime import datetime
import os
import getpass
from transformers import set_seed

from utils import get_synthetic_data_device_iterator, get_data_device_iterator, get_cpu_memory, verify_model, compare_tensors
import torch_xla.debug.metrics as met
from torch_xla.experimental.distributed_checkpoint import CheckpointManager
import jax

from run_dpo_no_trainer import prepare_model
logger = logging.get_logger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    set_seed(config.seed)

    logger.info("\n\n************** Experiment configuration ***********")
    logger.info(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, axis_names=("fsdp", "tensor") )
    xs.set_global_mesh(mesh)

    policy_dtype = getattr(torch, config.model.policy_dtype)

    logger.info(f"cpu memory usage: {get_cpu_memory()}")
    logger.info("loading model")
    if config.model.config_path:
        model_config = AutoConfig.from_pretrained(config.model.config_path)
        model_config.static = True
        model_config.flash_attention = config.flash_attention
        model_config.gmm = False
        model_config.gmm_stack = False
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(model_config).to_empty(device=xm.xla_device()).to(policy_dtype)
        model.apply(model._init_weights)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=config.cache_local_dir, torch_dtype=policy_dtype)
        model.config.static = True
        model.config.flash_attention = config.flash_attention
        model.config.gmm = False
        model.config.gmm_stack = False
    
    logger.info("model loaded")
    logger.info("cpu:")
    for k, v in model.state_dict().items():
        logger.info(f"{k}: {v.dtype} {v.mean()}")

    model = prepare_model(model, config)
    logger.info("FSDP model prepared tpu:")
    for k, v in model.state_dict().items():
        logger.info(f"{k}: {v.dtype} {v.mean()}")
    gc.collect()
    xm.mark_step()

    if config.model.name_or_path == "mistralai/Mixtral-8x22B-v0.1":
        # sentencepiece mismatch in a recent commit https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/discussions/9
        # https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/discussions/10
        tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path, revision="refs/pr/10", padding_side="right")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    
    verify_model(model, tokenizer, config, mesh)
    torch.distributed.init_process_group('gloo', init_method='xla://')
    if config.checkpoint_manager_path:
        ckpt_manager = CheckpointManager(
            path=config.checkpoint_manager_path,
            save_interval=1,
            max_to_keep=1,
        )

        state_dict = {
            'model': model.state_dict(),
        }
        logger.info("saved model.state_dict:")
        for k, v in state_dict['model'].items():
            logger.info(f"{k}: {v.dtype} {v.mean()}")

        ckpt_manager.save(0, state_dict)
    else:
        raise ValueError("need valid {config.checkpoint_manager_path=}")

    logger.info("checkpoing saving finished.")


if __name__ == '__main__':
    torch_xla._XLAC._xla_set_use_full_mat_mul_precision(use_full_mat_mul_precision=True)
    # jax.config.update("jax_default_matmul_precision", "highest")
    main()