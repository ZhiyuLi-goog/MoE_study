import torch
import os
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
from transformers import AutoTokenizer, TrainingArguments, default_data_collator

import numpy as np

from pdb import set_trace

from transformers import set_seed
from file_utils import get_file
from mlperf_logging_utils import ClmLogger, MLPerfCallback

from clm_datasets import get_datasets, process_datasets
USE_CUDA = torch.cuda.is_available()  # os.environ.get('USE_CUDA', False)

if not USE_CUDA:
    from transformers import Trainer, logging

    from model_utils_tpu import (
        setup_xla,
        setup_model_optimizer,
        get_global_batch_size,
        Tracker,
    )
    from accelerate import Accelerator

    OmegaConf.register_new_resolver(
        "get_global_batch_size",
        lambda per_device_batch_size: get_global_batch_size(per_device_batch_size),
    )
    OmegaConf.register_new_resolver(
        "path_join", lambda output_dir, exp_name: os.path.join(output_dir, exp_name)
    )

    logger = logging.get_logger(__name__)
else:
    import torch.multiprocessing as mp
    from nemo.core.config import hydra_runner
    from nemo_aligner.utils.train_script_utils import (
        CustomLoggerWrapper,
        add_custom_checkpoint_callback,
        init_using_ptl,
        extract_optimizer_scheduler_from_ptl_model
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

    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
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

    OmegaConf.resolve(config)
    set_seed(config.seed)
    logger.info("\n\n************** Experiment configuration ***********")
    logger.info(OmegaConf.to_yaml(config))
    if USE_CUDA:
        setup_distributed(config)

    trainer_args = TrainingArguments(
        output_dir=config.run_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        num_train_epochs=1,
        evaluation_strategy="steps",
        save_strategy="no",
        max_steps=config.max_steps,
        eval_steps=config.eval_frequency,
        eval_delay=0,
        logging_strategy="no",
        logging_steps=config.report_metrics_freq,
        report_to="tensorboard",
        seed=config.seed,
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )

    if not USE_CUDA:
        config_path = os.path.join(config.run_dir, "config.yaml")
        with get_file(config_path, "w") as f:
            OmegaConf.save(config, f)

        logger.info(f"log tensorboard to {os.path.join(config.run_dir, 'tensorboard')}")
        accelerator = Accelerator(log_with="tensorboard", project_dir=config.run_dir)
        tracker = Tracker(config, accelerator, logger)
        setup_xla(config)

        tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
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
        tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)

        with open_dict(config):
            # overwrite the model config with the config from the checkpoint
            config.model.encoder_seq_length = model.cfg.encoder_seq_length
   
    raw_datasets = get_datasets(config)
    datasets = process_datasets(raw_datasets, tokenizer, config)
    logger.info(f"{datasets=}")
    train_dataset, eval_dataset = datasets['train'], datasets['validation']
    if USE_CUDA:
        train_dataset, eval_dataset, train_ds, _ = get_input_pipeline(config, train_dataset, eval_dataset, tokenizer, model.tokenizer)
        tokenizer = model.tokenizer
        # initialize optimizer states
        init_using_ptl(megatron_trainer, model, train_dataset, train_ds)
        optimizer, scheduler = extract_optimizer_scheduler_from_ptl_model(model)

    clmlogger = ClmLogger(target_eval_loss=0)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits


    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        # Flatten the tokens
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1)
        # Enable model parallelism
        labels = labels.to(logits.device)
        loss = loss_fct(logits, labels)
        return {
            'eval_loss': loss,
        }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=trainer_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=[optimizer, scheduler],
        # Data collator will default to DataCollatorWithPadding, so we change it.
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        callbacks=[MLPerfCallback(clmlogger, len(train_dataset), len(eval_dataset))],
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

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
