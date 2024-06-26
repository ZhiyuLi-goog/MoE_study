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
from utils import get_synthetic_data_device_iterator, get_data_device_iterator, get_cpu_memory
import torch_xla.debug.metrics as met
from torch_xla.experimental.distributed_checkpoint import CheckpointManager

from run_dpo_no_trainer import prepare_model, logger

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

    model_torch_dtype = getattr(torch, config.model.torch_dtype)

    logger.info(f"cpu memory usage: {get_cpu_memory()}")
    logger.info("loading model")
    if config.model.config_path:
        model_config = AutoConfig.from_pretrained(config.model.config_path)
        model_config.static = True
        model_config.flash_attention = True
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(model_config).to_empty(device=xm.xla_device()).to(model_torch_dtype)
        model.init_weights()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=config.cache_local_dir, low_cpu_mem_usage=True, torch_dtype=model_torch_dtype)
        model = model.to_empty(device='cpu').to(model_torch_dtype)
    
    logger.info("model loaded")
    model = prepare_model(model, config)

    logger.info("model prepared")
    gc.collect()
    xm.mark_step()

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

        ckpt_manager.save(0, state_dict)
    else:
        raise ValueError("need valid {config.checkpoint_manager_path=}")
    
    logger.info("checkpoing saving finished.")


if __name__ == '__main__':
    main()