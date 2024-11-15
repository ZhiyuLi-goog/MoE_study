import os

import hydra
import numpy as np
import torch
from clm_datasets import get_datasets, process_datasets
from file_utils import get_file
from mlperf_logging_utils import MLPerfCallback
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    logging,
    set_seed,
)

USE_CUDA = torch.cuda.is_available()  # os.environ.get('USE_CUDA', False)

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)

if not USE_CUDA:
    from model_utils_tpu import (
        TensorBoardCallback,
        get_global_batch_size,
        setup_model_optimizer,
        setup_xla,
    )
    from trainer_utils_tpu import Trainer
    from transformers import logging

    OmegaConf.register_new_resolver(
        "get_global_batch_size",
        lambda per_device_batch_size: get_global_batch_size(per_device_batch_size),
    )
    OmegaConf.register_new_resolver(
        "path_join", lambda output_dir, exp_name: os.path.join(output_dir, exp_name)
    )
else:
    import torch.multiprocessing as mp
    from model_utils_gpu import setup_distributed, setup_model_and_trainer
    from nemo.collections import llm
    from trainer_utils_gpu import Trainer

    OmegaConf.register_new_resolver("int_div", lambda x, y: x // y, replace=True)
    OmegaConf.register_new_resolver(
        "path_join", lambda output_dir, exp_name: os.path.join(output_dir, exp_name)
    )
    OmegaConf.register_new_resolver(
        "get_global_batch_size",
        lambda per_device_batch_size: per_device_batch_size,
    )

    mp.set_start_method("spawn", force=True)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    if USE_CUDA:
        logger = logging.getLogger(__name__)
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
        config.model.name_or_path, add_eos_token=False, add_bos_token=False
    )

    clmlogger = ClmLogger(target_eval_loss=0)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)

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
        if "adam" in config.optimizer.lower():
            optimizer_name = "adam"
        else:
            raise ValueError("Unsupported optimizer for GPU run")

        data_parallel_size = torch.distributed.get_world_size() // (
            config.tensor_parallelism
            * config.pipeline_parallelism
            * config.context_parallelism
        )

        config.global_train_batch_size = int(
            config.per_device_train_batch_size
            * config.gradient_accumulation_steps
            * data_parallel_size
        )

        config.global_eval_batch_size = config.global_train_batch_size

        model, trainer, optimizer = setup_model_and_trainer(
            model_name_or_path=config.model.name_or_path,
            input_sequence_length=config.model.max_sequence_length,
            global_batch_size=config.global_train_batch_size,
            nodes=torch.distributed.get_world_size() // torch.cuda.device_count(),
            tp_size=config.tensor_parallelism,
            pp_size=config.pipeline_parallelism,
            cp_size=config.context_parallelism,
            learning_rate=config.lr,
            optimizer_name=optimizer_name,
            tokenizer_name_or_path=config.model.name_or_path,
            scheduler=config.sched,
            trainer_args=trainer_args,
            # callbacks=[MLPerfCallback(clmlogger, 100, 10)],
            callbacks=[],
        )
    raw_datasets = get_datasets(config)
    datasets = process_datasets(raw_datasets, tokenizer, config)
    logger.info(f"{datasets=}")
    train_dataset, eval_dataset = datasets["train"], datasets["validation"]

    if USE_CUDA:
        from clm_datasets import DatasetModule

        dataset = DatasetModule(
            train_dataset,
            eval_dataset,
            tokenizer,
            micro_batch_size=config.per_device_train_batch_size,
            global_batch_size=config.global_train_batch_size,
            seq_len=config.model.max_sequence_length,
        )

        llm.train(
            model=model,
            data=dataset,
            trainer=trainer,
            tokenizer="data",
            optim=optimizer,
        )
    else:

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=trainer_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=[optimizer, scheduler],
            # Data collator will default to DataCollatorWithPadding, so we change it.
            compute_metrics=None,
            data_collator=default_data_collator,
            # callbacks=[MLPerfCallback(clmlogger, len(train_dataset), len(eval_dataset))],
            callbacks=[MLPerfCallback(clmlogger, 100, 10)],
        )
        trainer.train()


if __name__ == "__main__":
    main()
