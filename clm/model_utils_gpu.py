import os

import torch
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import (
    get_prompt_template_example,
)
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf, open_dict
from transformers import AutoTokenizer


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
    input_sequence_length: int,
    global_batch_size: int,
    max_training_tokens: int,
    nodes: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    learning_rate: float,
    tokenizer_name_or_path: str,
    *,
    callbacks: list,
):
    logging.info("loading model")
    mixtral_config = llm.MixtralConfig8x7B(
        max_position_embeddings=input_sequence_length,
        seq_length=input_sequence_length,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    model = llm.MixtralModel(mixtral_config, tokenizer=tokenizer)

    ## initialize the strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        sequence_parallel=True,
        context_parallel_size=cp_size,
        pipeline_dtype=torch.bfloat16,
    )

    precision = nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )

    tokens_per_batch = global_batch_size * input_sequence_length
    training_steps = (max_training_tokens + tokens_per_batch - 1) // tokens_per_batch

    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=learning_rate,
        bf16=True,
        params_dtype=torch.bfloat16,
    )

    opt_sched = nl.lr_scheduler.CosineAnnealingScheduler(
        max_steps=training_steps,
        warmup_steps=0,
        min_lr=learning_rate * 0.01,
    )

    opt = nl.MegatronOptimizerModule(config=opt_config, lr_scheduler=opt_sched)

    trainer = nl.Trainer(
        devices=8,
        num_nodes=nodes,
        max_steps=training_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=precision,
        callbacks=callbacks,
        logger=None,
        enable_progress_bar=False,
        val_check_interval=100,
    )

    return (
        model,
        trainer,
        opt,
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


def _modify_config(gpt_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(gpt_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(gpt_cfg):
        gpt_cfg.megatron_amp_O2 = cfg.model.get("megatron_amp_O2", False)
        gpt_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
        gpt_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
        gpt_cfg.sequence_parallel = cfg.model.get("sequence_parallel", False)
        gpt_cfg.activations_checkpoint_granularity = cfg.model.get(
            "activations_checkpoint_granularity", None
        )
        gpt_cfg.activations_checkpoint_num_layers = cfg.model.get(
            "activations_checkpoint_num_layers", None
        )
        gpt_cfg.activations_checkpoint_method = cfg.model.get(
            "activations_checkpoint_method", None
        )
        gpt_cfg.activations_checkpoint_layers_per_pipeline = cfg.model.get(
            "activations_checkpoint_layers_per_pipeline", None
        )
        gpt_cfg.peft = cfg.model.peft
        gpt_cfg.data = cfg.model.data
        gpt_cfg.optim = cfg.model.optim
        gpt_cfg.precision = cfg.trainer.precision
        gpt_cfg.answer_only_loss = cfg.model.answer_only_loss
        gpt_cfg.restore_from_path = cfg.model.restore_from_path
        gpt_cfg.resume_from_checkpoint = cfg.model.resume_from_checkpoint
        gpt_cfg.save_nemo_on_validation_end = cfg.model.save_nemo_on_validation_end
        gpt_cfg.gradient_as_bucket_view = cfg.model.gradient_as_bucket_view
        gpt_cfg.hidden_dropout = cfg.model.get("hidden_dropout", 0.0)
        gpt_cfg.attention_dropout = cfg.model.get("attention_dropout", 0.0)
        gpt_cfg.ffn_dropout = cfg.model.ffn_dropout
        gpt_cfg.use_flash_attention = cfg.model.get("use_flash_attention", False)
        # if TP/PP size is -1, use default TP/PP size as original model
        if cfg.model.get("tensor_model_parallel_size", 1) > 0:
            gpt_cfg.tensor_model_parallel_size = cfg.model.get(
                "tensor_model_parallel_size", 1
            )
        if cfg.model.get("pipeline_model_parallel_size", 1) > 0:
            gpt_cfg.pipeline_model_parallel_size = cfg.model.get(
                "pipeline_model_parallel_size", 1
            )
        gpt_cfg.pipeline_model_parallel_split_rank = cfg.model.get(
            "pipeline_model_parallel_split_rank", 0
        )

        if cfg.model.data.get("chat", False):
            # chat model, overwrite the prompt template
            prompt_template = get_prompt_template_example(
                cfg.model.data.chat_prompt_tokens
            )
            gpt_cfg.data.train_ds.prompt_template = prompt_template
            gpt_cfg.data.validation_ds.prompt_template = prompt_template

        sft_cls = GPTSFTModel
        gpt_cfg.target = f"{sft_cls.__module__}.{sft_cls.__name__}"

        if cfg.model.get("use_flash_attention", None) is not None:
            gpt_cfg.use_flash_attention = cfg.model.use_flash_attention

        if cfg.model.get("seq_len_interpolation_factor", None) is not None:
            gpt_cfg.seq_len_interpolation_factor = (
                cfg.model.seq_len_interpolation_factor
            )

        gpt_cfg.inference = cfg.model.get("inference", {})

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(gpt_cfg)
            gpt_cfg.cfg = gpt_cfg

    return gpt_cfg


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
