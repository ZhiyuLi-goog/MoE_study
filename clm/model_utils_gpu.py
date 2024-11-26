import os

import torch
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import (
    get_prompt_template_example,
)
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf, open_dict
from transformers import TrainingArguments


def setup_distributed(config):
    """Initialize torch.distributed."""
    # Get rank and world size.
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    logging.info(
        f"Initializing torch.distributed with local_rank: {local_rank}, rank: {rank}, world_size: {world_size}"
    )

    # Set the device id.
    device = rank % torch.cuda.device_count()
    if local_rank is not None:
        device = local_rank
    torch.cuda.set_device(device)

    # Call the init process.
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    import datetime

    DEFAULT_TIMEOUT = datetime.timedelta(minutes=60)
    init_method += master_ip + ":" + master_port
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=DEFAULT_TIMEOUT,
        world_size=world_size,
        rank=rank,
        init_method=init_method,
    )
    return local_rank, rank, world_size


def setup_model_and_trainer(
    model_name_or_path: str,
    input_sequence_length: int,
    global_batch_size: int,
    nodes: int,
    tp_size: int,
    pp_size: int,
    vpp_size: int,
    cp_size: int,
    learning_rate: float,
    optimizer_name: str,
    tokenizer_name_or_path: str,
    scheduler,
    trainer_args: TrainingArguments,
    *,
    logger,
    callbacks: list,
):
    logging.info("loading model")

    if "mixtral-8x7b" in model_name_or_path.lower():
        mixtral_config = llm.MixtralConfig8x7B(
            max_position_embeddings=input_sequence_length,
            seq_length=input_sequence_length,
        )
    elif "mixtral-8x22b" in model_name_or_path.lower():
        mixtral_config = llm.MixtralConfig8x22B(
            max_position_embeddings=input_sequence_length,
            seq_length=input_sequence_length,
        )
    else:
        raise ValueError(f"Unknown model specified: {model_name_or_path}")

    resume = nl.AutoResume(resume_from_path="/app/checkpoints/")
    tokenizer = AutoTokenizer(pretrained_model_name=tokenizer_name_or_path)
    model = llm.MixtralModel(mixtral_config, tokenizer=tokenizer)

    ## initialize the strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        virtual_pipeline_model_parallel_size=vpp_size,
        sequence_parallel=True,
        context_parallel_size=cp_size,
        pipeline_dtype=torch.bfloat16,
        ckpt_load_optimizer=False,
    )

    precision = nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )

    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer=optimizer_name,
        lr=learning_rate,
        bf16=True,
        params_dtype=torch.bfloat16,
    )

    if scheduler.name == "CosineAnnealing":
        opt_sched = nl.lr_scheduler.CosineAnnealingScheduler(
            max_steps=scheduler.max_steps,
            warmup_steps=scheduler.warmup_steps,
            min_lr=scheduler.min_lr,
        )
    elif scheduler.name == "WarmupHoldPolicy":
        opt_sched = nl.lr_scheduler.WarmupHoldPolicyScheduler(
            warmup_steps=scheduler.warmup_steps,
            hold_steps=scheduler.hold_steps,
            max_steps=scheduler.max_steps,
        )

    opt = nl.MegatronOptimizerModule(config=opt_config, lr_scheduler=opt_sched)

    trainer = nl.Trainer(
        devices=8,
        num_nodes=nodes,
        max_steps=trainer_args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=precision,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,
        val_check_interval=trainer_args.eval_steps,
        log_every_n_steps=trainer_args.logging_steps,
        gradient_clip_val=trainer_args.max_grad_norm,
    )

    logger.set_trainer(trainer)
    logger.log_hyperparams(None)

    return (
        model,
        trainer,
        opt,
        resume,
    )


def flatten(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


class Tracker:
    def __init__(self, config, logger):
        exp_config = {}
        for k, v in flatten(OmegaConf.to_container(config)).items():
            if isinstance(v, (str, int, float, str, bool, torch.Tensor)):
                exp_config[k] = v
            else:
                exp_config[k] = str(v)

        self.logger = logger
        self.config = config

    def record_train_step(self, metrics, num_examples):
        if torch.cuda.current_device() == 0:
            self.logger.log_metrics(
                metrics,
                step=int(num_examples / self.config.global_train_batch_size),
                prefix="train/",
            )

    def record_eval_step(self, metrics, num_examples):
        if torch.cuda.current_device() == 0:
            self.logger.log_metrics(metrics, step=int(num_examples), prefix="val/")
