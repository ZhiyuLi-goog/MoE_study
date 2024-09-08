from datasets import load_dataset
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from typing import Optional, Union, Dict, List, Any
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
import numpy as np
from functools import partial
from psutil import Process
from torch.utils.data import DataLoader
from transformers import default_data_collator
import torch
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
import os


from transformers import logging

logger = logging.get_logger(__name__)


def compare_tensors(t1, t2, name=str, atol=1e-6, rtol=1e-6):
    result = torch.allclose(t1, t2, atol=atol, rtol=rtol)
    if result:
        return True
    else:
        print(f"{name=} {t1.shape=}")
        np.testing.assert_allclose(
            t1.float().cpu().numpy(), t2.float().cpu().numpy(), atol=atol, rtol=rtol
        )
        return False


def strip_padding(tokens_list, padding_token_id):
    """
    Strips padding tokens from the beginning and end of each sequence in a list.

    Args:
    tokens_list (list of list of int): List containing sequences of token IDs.
    padding_token_id (int): The token ID used for padding.

    Returns:
    list of list of int: The list of sequences with padding tokens removed.
    """

    def strip_single_sequence(sequence):
        # Remove padding tokens from the start
        start = 0
        while start < len(sequence) and sequence[start] == padding_token_id:
            start += 1
        # Remove padding tokens from the end
        end = len(sequence)
        while end > start and sequence[end - 1] == padding_token_id:
            end -= 1
        return sequence[start:end]

    return [strip_single_sequence(seq) for seq in tokens_list]


def decode(input_ids, tokenizer):
    # Assuming `input_ids' is tensor of shape (batch_size, seq_length)
    input_ids = input_ids.cpu().numpy()
    input_ids = strip_padding(input_ids, padding_token_id=-100)
    # Decode each sequence in the batch separately
    decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in input_ids]
    return decoded


def print_batch(batch, tokenizer):
    chosens = decode(batch["chosen_input_ids"], tokenizer)
    chosen_onlys = decode(batch["chosen_labels"], tokenizer)
    rejecteds = decode(batch["rejected_input_ids"], tokenizer)
    rejected_onlys = decode(batch["rejected_labels"], tokenizer)

    # Log each pair of chosen and rejected sequences
    for (
        chosen,
        rejected,
        chosen_only,
        rejected_only,
    ) in zip(chosens, rejecteds, chosen_onlys, rejected_onlys):
        logger.debug(
            f"{chosen=}\n\n{rejected=}\n\n{chosen_only=}\n\n{rejected_only=}\n\n"
        )


def fmt_size(num_bytes: int) -> str:
    assert num_bytes > 0
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if num_bytes < 1024.0:
            break
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} {unit}"


def get_cpu_memory() -> str:
    """print out cpu/tpu memory."""
    cpu_bytes = Process().memory_info().rss
    return fmt_size(cpu_bytes)
