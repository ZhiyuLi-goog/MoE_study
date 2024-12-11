import torch
import os
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
from transformers import AutoTokenizer, TrainingArguments, default_data_collator

import numpy as np

from transformers import set_seed
from file_utils import get_file
from mlperf_logging_utils import MLPerfCallback
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
from torch.utils.data import DataLoader

from clm_datasets import get_datasets, process_datasets
import torch_xla
import torch_xla.distributed.parallel_loader as pl

torch.set_printoptions(threshold=50000)


OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)

# from transformers import Trainer, logging
from transformers import logging
from trainer_utils_tpu import Trainer

from model_utils_tpu import (
    setup_xla,
    setup_model_optimizer,
    get_global_batch_size,
    TensorBoardCallback,
)
from accelerate import Accelerator

OmegaConf.register_new_resolver(
    "get_global_batch_size",
    lambda per_device_batch_size: get_global_batch_size(per_device_batch_size),
)
OmegaConf.register_new_resolver(
    "path_join", lambda output_dir, exp_name: os.path.join(output_dir, exp_name)
)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    logger = logging.get_logger(__name__)

    OmegaConf.resolve(config)
    set_seed(config.seed)
    logger.info("\n\n************** Experiment configuration ***********")
    logger.info(OmegaConf.to_yaml(config))

    if config.eval_frequency == -1:
        config.eval_frequency = int(
            np.ceil(24576 * 2048 / config.max_length / config.global_train_batch_size)
        )
    logger.info(f"{config.eval_frequency=}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path, add_eos_token=False, add_bos_token=False
    )

    config_path = os.path.join(config.run_dir, "config.yaml")
    with get_file(config_path, "w") as f:
        OmegaConf.save(config, f)

    logger.info(f"log tensorboard to {os.path.join(config.run_dir, 'tensorboard')}")
    accelerator = Accelerator(log_with="tensorboard", project_dir=config.run_dir)
    setup_xla(config)

    def print_tensor(key, tensor, dim=None):
        if dim is None:
            return f"{key}: dtype={tensor.dtype}, shape={tensor.shape}, mean={tensor.mean()}, min={tensor.min()}, max={tensor.max()}, std={tensor.std()}"
        else:
            return ( 
                    f"{key} dtype={tensor.dtype}, shape={tensor.shape}\n"
                    f"{key} mean={tensor.mean(dim)}\n"
                    f"{key} min={tensor.min(dim)[0]}\n"
                    f"{key} max={tensor.max(dim)[0]}\n"
                    f"{key} std={tensor.std(dim)}"
                   )

    model, optimizer, scheduler = setup_model_optimizer(config)
    for k, v in model.state_dict().items():
        logger.info(f"{print_tensor(k, v)}")

    if tokenizer.vocab_size != model.config.vocab_size:
        logger.warning(
            f"Found mismatch between {tokenizer.vocab_size=} and {model.config.vocab_size}"
        )
    raw_datasets = get_datasets(config)
    datasets = process_datasets(raw_datasets, tokenizer, config)
    logger.info(f"{datasets=}")
    train_dataset, eval_dataset = datasets["train"], datasets["validation"]

    mesh = xs.get_global_mesh()
    eval_dataloader = pl.MpDeviceLoader(
            DataLoader(
                eval_dataset,
                collate_fn=default_data_collator,
                batch_size=1,
            ),
            torch_xla.device(),
            input_sharding=xs.ShardingSpec(mesh, ("fsdp", None)),
        )

    for batch in eval_dataloader:
        with torch.no_grad():
            logger.info(f"{batch=}")
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            logger.info(f"{print_tensor('logits', logits, dim=-1)}")

            for i, layer_output in enumerate(outputs.hidden_states):
                logger.info(f"{print_tensor(f'layer_output_{i}', layer_output, dim=-1)}")
        xm.mark_step()
        break


if __name__ == "__main__":
    main()
