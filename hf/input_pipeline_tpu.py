import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.parallel_loader as pl

from preference_datasets import get_dataloader


class MultiHostDataLoadIterator:
    """fold get_next_batch_sharded into a iterator class"""

    def __init__(self, data_loader, mesh):
        self.device_loader = pl.MpDeviceLoader(
            data_loader,
            torch_xla.device(),
            # Shard the input's batch dimension along the `fsdp` axis, no sharding along other dimensions
            input_sharding=xs.ShardingSpec(mesh, ("fsdp", None)),
        )
        self.data_iterator = iter(self.device_loader)

    def reset(self):
        self.data_iterator = iter(self.device_loader)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        return next(self.data_iterator)


def get_synthetic_dataloader(config, tokenizer, global_batch_size):
    # Shard the input(data parallel).
    # Scale the batch size with num_devices since there will be only one
    # process that handles all runtime devices.
    assert xr.world_size() == 1

    def get_data(global_batch_size):
        data = {
            "chosen_input_ids": torch.randint(
                tokenizer.vocab_size,
                (global_batch_size, config.max_length),
                dtype=torch.int64,
            ),
            "chosen_attention_mask": torch.ones(
                global_batch_size, config.max_length, dtype=torch.int64
            ),
            "rejected_input_ids": torch.randint(
                tokenizer.vocab_size,
                (global_batch_size, config.max_length),
                dtype=torch.int64,
            ),
            "rejected_attention_mask": torch.ones(
                global_batch_size, config.max_length, dtype=torch.int64
            ),
        }
        data["chosen_labels"] = data["chosen_input_ids"]
        data["rejected_labels"] = data["rejected_input_ids"]
        data["chosen_labels"][:, : config.max_length // 2] = config.label_pad_token_id
        data["rejected_labels"][:, : config.max_length // 2] = config.label_pad_token_id
        return data

    train_loader = xu.SampleGenerator(
        data=get_data(config.global_train_batch_size),
        sample_count=100,
    )
    eval_loader = xu.SampleGenerator(
        data=get_data(config.global_eval_batch_size),
        sample_count=10,
    )
    return train_loader, eval_loader


def get_input_pipeline(config, tokenizer):
    """get input_pipeline."""
    mesh = xs.get_global_mesh()
    num_devices = xr.global_runtime_device_count()
    global_batch_size = int(config.per_device_train_batch_size * num_devices)
    if config.use_synthetic_data:
        train_loader, eval_loader = get_synthetic_dataloader(
            config, tokenizer, global_batch_size
        )
    else:
        train_loader, eval_loader = get_dataloader(
            config, tokenizer, load_from_cache_file=True
        )
    if config.num_proc > 1:
        raise ValueError(
            f"{config.num_proc=}, which is bigger than 1. HuggingFace treats SPMD as a single-device program."
        )
    return (
        MultiHostDataLoadIterator(train_loader, mesh),
        MultiHostDataLoadIterator(eval_loader, mesh),
        None,
        None,
    )
