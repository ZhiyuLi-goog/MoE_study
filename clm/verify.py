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
from torch.utils.data import DataLoader

from clm_datasets import get_datasets, process_datasets
import torch_xla
import torch_xla.distributed.parallel_loader as pl


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

    model, optimizer, scheduler = setup_model_optimizer(config)
    for k, v in model.state_dict().items():
        logger.info(f"{k}: {v.dtype} {v.mean()=} {v.min()=} {v.max()=} {v.std()=}")

    if tokenizer.vocab_size != model.config.vocab_size:
        logger.warning(
            f"Found mismatch between {tokenizer.vocab_size=} and {model.config.vocab_size}"
        )
    raw_datasets = get_datasets(config)
    datasets = process_datasets(raw_datasets, tokenizer, config)
    logger.info(f"{datasets=}")
    train_dataset, eval_dataset = datasets["train"], datasets["validation"]

    train_dataloader = pl.MpDeviceLoader(
            DataLoader(
                train_dataset,
                collate_fn=default_data_collator,
                batch_size=1,
            ),
            torch_xla.device(),
        )

    for batch in train_dataloader:
        with torch.no_grad():
            logger.info(f"{batch=}")
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            logger.info(f"{logits.mean(-1)=}")
            logger.info(f"{logits.min(-1)=}")
            logger.info(f"{logits.max(-1)=}")
            logger.info(f"{logits.std(-1)=}")

            for i, layer_output in enumerate(outputs.hidden_states):
                logger.info(f"layer_idx={i}")
                logger.info(f"{layer_output.mean(-1)=}")
                logger.info(f"{layer_output.min(-1)=}")
                logger.info(f"{layer_output.max(-1)=}")
                logger.info(f"{layer_output.std(-1)=}")
        break


if __name__ == "__main__":
    main()
