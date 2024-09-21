import torch
import os
import hydra
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer

import numpy as np

from transformers import logging

from transformers import set_seed
from utils import get_cpu_memory, print_batch
from dpo_trainers import get_batch_loss_metrics
from file_utils import get_file

USE_CUDA = torch.cuda.is_available()  # os.environ.get('USE_CUDA', False)
assert USE_CUDA == False, "CUDA not supported"
if not USE_CUDA:
    from model_utils_tpu import (
        setup_xla,
        setup_model_optimizer,
        get_global_batch_size,
        Tracker,
    )
    from input_pipeline_tpu import get_input_pipeline
    from accelerate import Accelerator

    OmegaConf.register_new_resolver(
        "get_global_batch_size",
        lambda per_device_batch_size: get_global_batch_size(per_device_batch_size),
    )
    OmegaConf.register_new_resolver(
        "path_join", lambda output_dir, exp_name: os.path.join(output_dir, exp_name)
    )


logger = logging.get_logger(__name__)


def clip_gradient(model, config):
    """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)


def train_step(
    model,
    ref_model,
    train_device_loader,
    config,
    step,
    optimizer,
    scheduler,
    start_step,
    tokenizer,
):
    batch = next(train_device_loader)
    if step == start_step:
        print_batch(batch, tokenizer)
    optimizer.zero_grad()
    model.train()
    loss, metrics = get_batch_loss_metrics(
        model, ref_model, batch, "train", beta=config.beta, config=config
    )

    loss.backward()
    if config.max_grad_norm > 0.0:
        grad_norm = clip_gradient(model, config)
        metrics["train/grad_norm"] = grad_norm
    metrics["train/lr"] = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()
    return metrics


def eval_fn(model, ref_model, eval_device_loader, config, step, tracker):
    prefix = "eval/"
    group_eval_metrics = {
        f"{prefix}rewards/chosen_per_example": [],
        f"{prefix}rewards/rejected_per_example": [],
        f"{prefix}rewards/accuracies_per_example": [],
        f"{prefix}rewards/margins_per_example": [],
        f"{prefix}logps/rejected_per_example": [],
        f"{prefix}logps/chosen_per_example": [],
        f"{prefix}logits/rejected_per_example": [],
        f"{prefix}logits/chosen_per_example": [],
        f"{prefix}losses_per_example": [],
        f"{prefix}num_examples": [],
        f"{prefix}ppl_per_token": [],
    }

    for eval_batch in eval_device_loader:
        model.eval()
        with torch.no_grad():
            _, eval_metrics = get_batch_loss_metrics(
                model, ref_model, eval_batch, "eval", beta=config.beta, config=config
            )
        for k in group_eval_metrics:
            group_eval_metrics[k].append(eval_metrics[k])

    for k, v in group_eval_metrics.items():
        if k == f"{prefix}num_examples":
            group_eval_metrics[k] = sum(v)
        else:
            group_eval_metrics[k] = sum(v) / len(v)

    group_eval_metrics[f"{prefix}trained_num_examples"] = (
        step * config.global_train_batch_size
    )
    tracker.record_eval_step(group_eval_metrics, step * config.global_train_batch_size)


def hydra_decorator(config_path, config_name):
    def decorator(func):
        if not USE_CUDA:
            return hydra.main(
                version_base=None, config_path=config_path, config_name=config_name
            )(func)

    return decorator


@hydra_decorator(config_path="config", config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    set_seed(config.seed)
    logger.info("\n\n************** Experiment configuration ***********")
    logger.info(OmegaConf.to_yaml(config))

    if not USE_CUDA:
        config_path = os.path.join(config.run_dir, "config.yaml")
        with get_file(config_path, "w") as f:
            OmegaConf.save(config, f)

        logger.info(f"log tensorboard to {os.path.join(config.run_dir, 'tensorboard')}")
        accelerator = Accelerator(log_with="tensorboard", project_dir=config.run_dir)
        tracker = Tracker(config, accelerator, logger)
        setup_xla(config)

        tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)
        model, ref_model, optimizer, scheduler = setup_model_optimizer(config)

    if tokenizer.vocab_size != model.config.vocab_size:
        logger.warning(
            f"Found mismatch between {tokenizer.vocab_size=} and {model.config.vocab_size}"
        )
    train_device_loader, eval_device_loader, train_ds, _ = get_input_pipeline(
        config, tokenizer
    )

    start_step = 0

    logger.info(f"cpu memory usage: {get_cpu_memory()}")
    step = start_step
    for step in np.arange(start_step, config.max_steps):
        if (
            step == start_step
            and config.do_first_eval
            or step > start_step
            and step % config.eval_frequency == 0
        ):
            eval_metrics = eval_fn(
                model, ref_model, eval_device_loader, config, step, tracker
            )
        try:
            train_metrics = train_step(
                model,
                ref_model,
                train_device_loader,
                config,
                step,
                optimizer,
                scheduler,
                start_step,
                tokenizer,
            )
            if step % config.report_metrics_freq == 0:
                tracker.record_train_step(
                    train_metrics, step * config.global_train_batch_size
                )
        except StopIteration:
            break

    if not USE_CUDA:
        tracker.accelerator.end_training()


if __name__ == "__main__":
    main()
