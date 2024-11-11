import os

import torch
import torch.distributed as dist
from mlperf_logging import mllog
from mlperf_logging.mllog import constants
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    is_torch_xla_available,
)

if is_torch_xla_available():
    import torch_xla.runtime as xr


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if is_torch_xla_available():
        return xr.global_ordinal()
    else:
        if not is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()


def barrier():
    if not is_dist_avail_and_initialized():
        return
    torch.distributed.barrier()


class ClmLogger:
    def __init__(self, config, filename=None, default_stack_offset=2):
        self.mllogger = mllog.get_mllogger()
        mllog.config(
            default_stack_offset=default_stack_offset,
            filename=(
                filename
                or os.getenv("COMPLIANCE_FILE")
                or os.path.join(config.run_dir, "mlperf_compliance.log")
            ),
        )
        self.target_eval_loss = config.target_eval_loss

    def event(self, key, value=None, metadata=None, sync=False, log_rank=None):
        if get_rank() == 0:
            self.mllogger.event(key=key, value=value, metadata=metadata)

    def start(self, key, value=None, metadata=None, sync=False, log_rank=None):
        if get_rank() == 0:
            self.mllogger.start(key=key, value=value, metadata=metadata)

    def end(self, key, value=None, metadata=None, sync=False, log_rank=None):
        if get_rank() == 0:
            self.mllogger.end(key=key, value=value, metadata=metadata)


class MLPerfCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def __init__(self, config):
        super().__init__()
        self.mllogger = ClmLogger(config)
        self.submission_info = {
            "submission_benchmark": "mixture-of-expert",  # TODO change task name
            "submission_division": "closed",
            "submission_org": "Google",
            "submission_platform": "reference",
            "submission_status": "reference",
        }
        self.mllogger.event(
            key=constants.CACHE_CLEAR,
            value="True",
        )
        self.mllogger.start(key=constants.INIT_START, value="")
        self.config = config
        self.global_batch_tokens = config.global_train_batch_size * config.max_length

    def on_train_begin(self, args, state, control, **kwargs):

        self.mllogger.event(
            key=constants.SUBMISSION_BENCHMARK,
            value=self.submission_info["submission_benchmark"],
        )
        self.mllogger.event(
            key=constants.SUBMISSION_DIVISION,
            value=self.submission_info["submission_division"],
        )
        self.mllogger.event(
            key=constants.SUBMISSION_ORG, value=self.submission_info["submission_org"]
        )
        self.mllogger.event(
            key=constants.SUBMISSION_PLATFORM,
            value=self.submission_info["submission_platform"],
        )
        self.mllogger.event(
            key=constants.SUBMISSION_STATUS,
            value=self.submission_info["submission_status"],
        )
        self.mllogger.event(
            key=constants.GLOBAL_BATCH_SIZE,
            value=self.config.global_train_batch_size,
        )
        self.mllogger.event(
            key=constants.EVAL_SAMPLES,
            value=12694503,
        )
        self.mllogger.event(key=constants.SEED, value=args.seed)
        self.mllogger.event(
            key=constants.OPT_LR_WARMUP_FACTOR, value=args.sched.warmup_ratio
        )
        self.mllogger.event(key=constants.OPT_LR_TRAINING_STEPS, value=args.max_steps)
        self.mllogger.event(
            key=constants.OPT_ADAMW_WEIGHT_DECAY, value=args.weight_decay
        )
        self.mllogger.event(
            key=constants.OPT_GRADIENT_CLIP_NORM, value=args.max_grad_norm
        )
        self.mllogger.event(key=constants.OPT_BASE_LR, value=args.lr)
        self.mllogger.event(
            key=constants.GRADIENT_ACCUMULATION_STEPS,
            value=args.gradient_accumulation_steps,
        )
        # device warmup should be done here
        self.mllogger.end(key=constants.INIT_STOP, value="")
        self.mllogger.start(constants.RUN_START, value="")
        self.mllogger.start(
            constants.BLOCK_START,
            value="",
            metadata={
                "samples_count": 0,
            },
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step.
        """
        if state.global_step % (state.eval_steps) == 0 and state.global_step > 0:
            self.mllogger.end(
                constants.BLOCK_STOP,
                value="",
                metadata={
                    "samples_count": state.global_step * self.global_batch_tokens,
                },
            )
            self.mllogger.event(
                constants.EVAL_ACCURACY,
                value=state.log_history[-1]["eval/loss"],
                metadata={
                    "samples_count": state.global_step * self.global_batch_tokens
                },
            )
            latest_eval_loss = float("nan")
            if state.log_history and "eval/loss" in state.log_history[-1]:
                latest_eval_loss = state.log_history[-1]["eval/loss"]
            if latest_eval_loss <= self.mllogger.target_eval_loss:
                control.should_training_stop = True
                self.mllogger.end(
                    constants.RUN_STOP,
                    value=latest_eval_loss,
                    metadata={
                        "samples_count": state.global_step * self.global_batch_tokens,
                        "status": "success",
                    },
                )
            if state.global_step >= state.max_steps:
                control.should_training_stop = True
                self.mllogger.end(
                    constants.RUN_STOP,
                    value=latest_eval_loss,
                    metadata={
                        "samples_count": state.global_step * self.global_batch_tokens,
                        "status": "fail",
                    },
                )

            if not control.should_training_stop:
                self.mllogger.start(
                    constants.BLOCK_START,
                    value="",
                    metadata={
                        "samples_count": state.global_step * self.global_batch_tokens
                    },
                )

        return control
