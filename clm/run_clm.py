import torch
import os
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
from transformers import AutoTokenizer, TrainingArguments, default_data_collator

import numpy as np

from transformers import set_seed
from file_utils import get_file
from mlperf_logging_utils import MLPerfCallback

from clm_datasets import get_datasets, process_datasets

USE_CUDA = torch.cuda.is_available()  # os.environ.get('USE_CUDA', False)

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)

if not USE_CUDA:
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
else:
    import torch.multiprocessing as mp
    from nemo.core.config import hydra_runner
    from nemo_aligner.utils.train_script_utils import (
        CustomLoggerWrapper,
        add_custom_checkpoint_callback,
        init_using_ptl,
        extract_optimizer_scheduler_from_ptl_model,
    )
    from model_utils_gpu import (
        setup_distributed,
        setup_model_optimizer,
        Tracker,
    )
    from input_pipeline_gpu import (
        get_input_pipeline,
    )
    from trainer_utils_gpu import Trainer
    from nemo_aligner.utils.distributed import Timer

    OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)

    mp.set_start_method("spawn", force=True)


def hydra_decorator(config_path, config_name):
    def decorator(func):
        if not USE_CUDA:
            return hydra.main(
                version_base=None, config_path=config_path, config_name=config_name
            )(func)
        else:
            return hydra_runner(config_path=config_path, config_name=config_name)(func)

    return decorator


@hydra_decorator(config_path="config", config_name="config")
def main(config: DictConfig):
    if USE_CUDA:
        from nemo.utils import logging as logger
    else:
        logger = logging.get_logger(__name__)

    OmegaConf.resolve(config)
    set_seed(config.seed)
    logger.info("\n\n************** Experiment configuration ***********")
    logger.info(OmegaConf.to_yaml(config))
    if USE_CUDA:
        setup_distributed(config)

    if config.eval_frequency == -1:
        config.eval_frequency = int(
            np.ceil(24576 * 2048 / config.max_length / config.global_train_batch_size)
        )
    logger.info(f"{config.eval_frequency=}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path, add_eos_token=False, add_bos_token=False, use_fast=False,
    )

    if not USE_CUDA:
        config_path = os.path.join(config.run_dir, "config.yaml")
        with get_file(config_path, "w") as f:
            OmegaConf.save(config, f)

        logger.info(f"log tensorboard to {os.path.join(config.run_dir, 'tensorboard')}")
        accelerator = Accelerator(log_with="tensorboard", project_dir=config.run_dir)
        setup_xla(config)

        model, optimizer, scheduler = setup_model_optimizer(config)

        if tokenizer.vocab_size != model.config.vocab_size:
            logger.warning(
                f"Found mismatch between {tokenizer.vocab_size=} and {model.config.vocab_size}"
            )
    else:
        from nemo.utils.exp_manager import exp_manager
        from nemo_aligner.utils.train_script_utils import resolve_and_create_trainer

        megatron_trainer = resolve_and_create_trainer(config, "sft")
        exp_manager(megatron_trainer, config.exp_manager)

        with open_dict(config):
            config.model.precision = config.trainer.precision

        model, _, _ = setup_model_optimizer(config, megatron_trainer)

        with open_dict(config):
            # overwrite the model config with the config from the checkpoint
            config.model.encoder_seq_length = model.cfg.encoder_seq_length

    raw_datasets = get_datasets(config)
    datasets = process_datasets(raw_datasets, tokenizer, config)
    logger.info(f"{datasets=}")
    train_dataset, eval_dataset = datasets["train"], datasets["validation"]
    if USE_CUDA:
        train_dataset, eval_dataset, train_ds, _ = get_input_pipeline(
            config, train_dataset, eval_dataset, tokenizer, model.tokenizer
        )
        tokenizer = model.tokenizer
        # initialize optimizer states
        init_using_ptl(megatron_trainer, model, train_dataset, train_ds)
        optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(model)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        callbacks=[MLPerfCallback(config), TensorBoardCallback(config)],
    )
    if USE_CUDA:
        ckpt_callback = add_custom_checkpoint_callback(megatron_trainer, model)
        timer = Timer(config.exp_manager.get("max_time_per_run"))

        trainer.setup(
            cfg=config.trainer.sft,
            logger=CustomLoggerWrapper(megatron_trainer.loggers),
            ckpt_callback=ckpt_callback,
            run_timer=timer,
        )

    trainer.train()


if __name__ == "__main__":
    main()
