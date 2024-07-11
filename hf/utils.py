from datasets import load_dataset
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
import torch
import torch_xla
import torch_xla.utils.utils as xu
import torch_xla.distributed.spmd as xs
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data import DataLoader
from transformers import default_data_collator
import torch
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

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


class MultiHostDataLoadIterator:
    """fold get_next_batch_sharded into a iterator class"""
  
    def __init__(self, data_loader, mesh):
      self.device_loader = pl.MpDeviceLoader(
          data_loader,
          torch_xla.device(),
          # Shard the input's batch dimension along the `fsdp` axis, no sharding along other dimensions
          input_sharding=xs.ShardingSpec(mesh, ('fsdp', None)))
      self.data_iterator = iter(self.device_loader)
  
    def reset(self):
      self.data_iterator = iter(self.device_loader)
  
    def __iter__(self):
      self.reset()
      return self

    def __next__(self):
      return next(self.data_iterator)


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )

@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    max_length: int = -1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                to_pad = [torch.LongTensor(ex[k]) for ex in features]

                if (k.startswith("prompt")) and (k.endswith("input_ids")):
                    if self.pad_token_id is None:
                        raise ValueError(
                            "Padding is enabled, but the tokenizer is not configured with a padding token."
                            " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                            " before calling the trainer."
                        )
                    padding_value = self.pad_token_id
                elif k.endswith("_attention_mask"):
                    padding_value = 0
                elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                    padding_value = self.label_pad_token_id
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                # Convert to tensor and pad
                if self.max_length > 0:
                    padded_batch[k] = torch.stack([pad_to_length(ex, self.max_length, padding_value) for ex in to_pad], dim=0)
                else:
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch

def get_synthetic_data_device_iterator(config, tokenizer, mesh):
    # Shard the input(data parallel).
    # Scale the batch size with num_devices since there will be only one
    # process that handles all runtime devices.
    assert xr.world_size() == 1
    num_devices = xr.global_runtime_device_count()
    global_batch_size = int(config.per_device_train_batch_size * num_devices)
    data = {
        "chosen_input_ids": torch.randint(tokenizer.vocab_size, (global_batch_size, config.max_length), dtype=torch.int64),
        "chosen_attention_mask": torch.ones(global_batch_size, config.max_length, dtype=torch.int64),
        "rejected_input_ids": torch.randint(tokenizer.vocab_size, (global_batch_size, config.max_length), dtype=torch.int64),
        "rejected_attention_mask": torch.ones(global_batch_size, config.max_length, dtype=torch.int64),
    }
    data["chosen_labels"] = data["chosen_input_ids"]
    data["rejected_labels"] = data["rejected_input_ids"]
    # data["chosen_labels"][:, :config.max_length // 2] = config.label_pad_token_id
    # data["rejected_labels"][:, :config.max_length // 2] = config.label_pad_token_id
    train_loader = xu.SampleGenerator(
        data = data,
        sample_count=100,
    )
    eval_loader = xu.SampleGenerator(
        data = data,
        sample_count=10,
    )

    return MultiHostDataLoadIterator(train_loader, mesh), MultiHostDataLoadIterator(eval_loader, mesh)


def build_tokenized_answer(tokenizer, prompt, answer):
    """
    Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
    It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
    Reference:
        https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    """

    full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
    prompt_input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
    answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

    # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
    full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

    # Prepare input tokens for token by token comparison
    full_input_ids = np.array(full_tokenized["input_ids"])

    if len(full_input_ids) != len(full_concat_input_ids):
        raise ValueError("Prompt input ids and answer input ids should have the same length.")

    # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
    # can be merged together when tokenizing prompt+answer. This could result
    # on the last token from the prompt being different when tokenized on its own
    # vs when done as prompt+answer.
    response_token_ids_start_idx = len(prompt_input_ids)

    # If tokenized prompt is different than both prompt+answer, then it means the
    # last token has changed due to merging.
    if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
        response_token_ids_start_idx -= 1

    prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
    prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

    if len(prompt_input_ids) != len(prompt_attention_mask):
        raise ValueError("Prompt input ids and attention mask should have the same length.")

    answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
    answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

    return dict(
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
        input_ids=answer_input_ids,
        attention_mask=answer_attention_mask,
    )


def tokenize_row(feature, tokenizer=None, truncation_mode="keep_start", max_length=512, max_prompt_length=256) -> Dict:
    """Tokenize a single row from a DPO specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + chosen or prompt + rejected responses is/are too long. First
        we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
        the sum of the length of the prompt and the chosen/rejected response, with
        label_pad_token_id  for the prompt tokens.
    """
    label_pad_token_id = -100
    batch = {}
    prompt = feature["prompt"]
    chosen = feature["chosen"]
    rejected = feature["rejected"]

    # Check issues below for more details
    #  1. https://github.com/huggingface/trl/issues/907
    #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
    #  3. https://github.com/LianjiaTech/BELLE/issues/337

    if not isinstance(prompt, str):
        raise ValueError(f"prompt should be an str but got {type(prompt)}")
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

    if not isinstance(chosen, str):
        raise ValueError(f"chosen should be an str but got {type(chosen)}")
    chosen_tokens = build_tokenized_answer(tokenizer, prompt, chosen)

    if not isinstance(rejected, str):
        raise ValueError(f"rejected should be an str but got {type(rejected)}")
    rejected_tokens = build_tokenized_answer(tokenizer, prompt, rejected)

    # Last prompt token might get merged by tokenizer and
    # it should not be included for generation if that happens
    prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

    chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
    rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
    prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

    for k, v in prompt_tokens.items():
        prompt_tokens[k] = v[:prompt_len_input_ids]

    # Make sure prompts only have one different token at most an
    # and length only differs by 1 at most
    num_diff_tokens = sum(
        [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
    )
    num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
    if num_diff_tokens > 1 or num_diff_len > 1:
        raise ValueError(
            "Chosen and rejected prompt_input_ids might only differ on the "
            "last token due to tokenizer merge ops."
        )

    # add BOS token to head of prompt. Avoid adding if it's already there
    bos_token_id = tokenizer.bos_token_id
    if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
        prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
        prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
    if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
        chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
        chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
    if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
        rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
        rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

    # add EOS token to end of answer. Avoid adding if it's already there
    eos_token_id = tokenizer.eos_token_id
    if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
        chosen_tokens["input_ids"].append(eos_token_id)
        chosen_tokens["attention_mask"].append(1)
    if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
        rejected_tokens["input_ids"].append(eos_token_id)
        rejected_tokens["attention_mask"].append(1)

    longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

    # if combined sequence is too long, truncate the prompt
    for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
            if truncation_mode == "keep_start":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: max_prompt_length]
            elif truncation_mode == "keep_end":
                for k in ["prompt_input_ids", "prompt_attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][-max_prompt_length :]
            else:
                raise ValueError(f"Unknown truncation mode: {truncation_mode}")

    # if that's still too long, truncate the response
    for answer_tokens in [chosen_tokens, rejected_tokens]:
        if len(answer_tokens["prompt_input_ids"]) + longer_response_length > max_length:
            for k in ["input_ids", "attention_mask"]:
                answer_tokens[k] = answer_tokens[k][: max_length - max_prompt_length]

    # Create labels/torc
    chosen_sequence_tokens = {
        k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
    }
    rejected_sequence_tokens = {
        k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
    }
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
        label_pad_token_id
    ] * len(chosen_tokens["prompt_input_ids"])
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
        label_pad_token_id
    ] * len(rejected_tokens["prompt_input_ids"])

    for k, toks in {
        "chosen_": chosen_sequence_tokens,
        "rejected_": rejected_sequence_tokens,
        "": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}{type_key}"] = tokens

    return batch

def get_data_device_iterator(config, tokenizer, mesh, load_from_cache_file=True):

    ds = load_dataset(config.datasets)
    num_proc = config.num_proc
    if num_proc > 1:
        raise ValueError(f"{config.num_proc=}, which is bigger than 1. HuggingFace treats SPMD as a single-device program.")

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    # HuggingFace treats SPMD as a single-device program
    ds = ds.map(
        process,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc="apply_chat_template",
    )

    ds = ds.map(partial(tokenize_row, tokenizer=tokenizer, max_prompt_length=config.max_prompt_length, max_length=config.max_length), num_proc=num_proc, load_from_cache_file=load_from_cache_file, desc="tokenize_row")

    ds = ds.select_columns(
        ['chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels']
    )

    # def pad_sequence(row, max_length: int = None, pad_token_id: int = 0, label_pad_token_id: int = -100):
    #     if not max_length:
    #         max_length = max([len(v) for _, v in row.items()])
    #     for k in row.keys():
    #         if k.endswith("input_ids"):
    #             padding_value = pad_token_id
    #         elif k.endswith("_attention_mask"):
    #             padding_value = 0
    #         elif k.startswith(("chosen", "rejected")):
    #             padding_value = label_pad_token_id
    #         row[k] += [padding_value] * (max_length - len(row[k]))
    #     return row
    
    # ds = ds.map(partial(pad_sequence, max_length=config.max_length), num_proc=num_proc, load_from_cache_file=load_from_cache_file, desc="pad_sequence")

    data_collator = DPODataCollatorWithPadding(max_length=config.max_length)

    num_devices = xr.global_runtime_device_count()
    global_batch_size = int(config.per_device_train_batch_size * num_devices)
    train_loader, eval_loader = DataLoader(ds['train'], batch_size=global_batch_size, shuffle=True, drop_last=True, collate_fn=data_collator), DataLoader(ds['test'], batch_size=global_batch_size, collate_fn=data_collator)
    return MultiHostDataLoadIterator(train_loader, mesh), MultiHostDataLoadIterator(eval_loader, mesh)

