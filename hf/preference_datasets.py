from typing import Union, Dict, List, Any
import os
import torch
from datasets import concatenate_datasets, load_dataset, DatasetDict
from functools import partial
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from omegaconf import ListConfig


ASSISTANT_PREFIX = "\n\nAssistant:"
HUMAN_PREFIX = "\n\nHuman: "


def pad_to_length(
    tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1
) -> torch.Tensor:
    """pad up to a fix sequence length."""
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value
                * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


@dataclass
class DPODataCollatorWithPadding:
    """
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
            if k.endswith(
                ("_input_ids", "_attention_mask", "_labels", "_pixel_values")
            ):
                to_pad = [torch.LongTensor(ex[k]) for ex in features]

                if k.endswith("input_ids"):
                    if self.pad_token_id is None:
                        raise ValueError(
                            "Padding is enabled, but DPODataCollatorWithPadding is not configured with a pad_token_id."
                        )
                    padding_value = self.pad_token_id
                elif k.endswith("_attention_mask"):
                    padding_value = 0
                elif k.startswith(("chosen", "rejected", "completion")) or (
                    "decoder" in k
                ):
                    padding_value = self.label_pad_token_id
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                # Convert to tensor and pad
                if self.max_length > 0:
                    padded_batch[k] = torch.stack(
                        [
                            pad_to_length(ex, self.max_length, padding_value)
                            for ex in to_pad
                        ],
                        dim=0,
                    )
                else:
                    padded_batch[k] = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


def tokenize_row(
    feature,
    tokenizer=None,
    truncation_mode="keep_end",
    max_length=512,
    max_prompt_length=256,
    label_pad_token_id=-100,
) -> Dict:
    """Tokenize a single row from a DPO specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + chosen or prompt + rejected responses is/are too long. First
        we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
        the sum of the length of the prompt and the chosen/rejected response, with
        label_pad_token_id  for the prompt tokens.
    """
    batch = {}
    chosen = feature["chosen"]
    rejected = feature["rejected"]
    prompt = feature["prompt"]

    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    # add EOS token to end of answer. Avoid adding if it's already there
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id != chosen_tokens["input_ids"][-1]:
        chosen_tokens["input_ids"].append(eos_token_id)
    if eos_token_id != rejected_tokens["input_ids"][-1]:
        rejected_tokens["input_ids"].append(eos_token_id)
        rejected_tokens["attention_mask"].append(1)
    longer_response_length = max(
        len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
    )

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
        if truncation_mode == "keep_start":
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == "keep_end":
            prompt_tokens = {
                k: v[-max_prompt_length:] for k, v in prompt_tokens.items()
            }
        else:
            raise ValueError(f"Unknown truncation mode: {truncation_mode}")

    # if that's still too long, truncate the response
    if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
        chosen_tokens = {
            k: v[: max_length - max_prompt_length] for k, v in chosen_tokens.items()
        }
        rejected_tokens = {
            k: v[: max_length - max_prompt_length] for k, v in rejected_tokens.items()
        }

    # Create labels
    chosen_sequence_tokens = {
        k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens
    }
    rejected_sequence_tokens = {
        k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens
    }
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
        label_pad_token_id
    ] * len(prompt_tokens["input_ids"])
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [
        label_pad_token_id
    ] * len(prompt_tokens["input_ids"])

    batch = {}

    batch["prompt"] = prompt
    batch["chosen"] = prompt + chosen
    batch["rejected"] = prompt + rejected
    batch["chosen_response_only"] = chosen
    batch["rejected_response_only"] = rejected

    for k, toks in {
        "chosen": chosen_sequence_tokens,
        "rejected": rejected_sequence_tokens,
        "prompt": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}_{type_key}"] = np.array(tokens)
    return batch


def get_shp(num_proc: int = 1, load_from_cache_file: bool = True):
    """get and process stanfordnlp/SHP dataset."""
    ds = load_dataset("stanfordnlp/SHP")
    ds.pop("validation")

    def format_process(row):
        prompt = HUMAN_PREFIX + row["history"] + ASSISTANT_PREFIX
        # if it is 1 if A is preferred to B; 0 if B is preferred to A
        if row["labels"] == 1:
            chosen = row["human_ref_A"]
            rejected = row["human_ref_B"]
        else:
            chosen = row["human_ref_B"]
            rejected = row["human_ref_A"]
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    ds = ds.map(
        format_process,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc="get_shp/format_process",
    )
    ds = ds.select_columns(["prompt", "chosen", "rejected"])
    return ds


def get_hh(num_proc: int = 1, load_from_cache_file: bool = True):
    """get and process Anthropic/hh-rlhf dataset."""
    ds = load_dataset("Anthropic/hh-rlhf")

    def extract_anthropic_prompt(chosen, rejected):
        """Extract the anthropic prompt from a prompt and response pair."""
        search_term = ASSISTANT_PREFIX
        common_prefix = os.path.commonprefix([chosen, rejected])
        search_term_idx = common_prefix.rfind(search_term)
        assert (
            search_term_idx != -1
        ), f"Prompt and response does not contain '{search_term}'"
        return chosen[: search_term_idx + len(search_term)]

    def split_prompt_and_responses(row):
        prompt = extract_anthropic_prompt(row["chosen"], row["rejected"])
        return {
            "prompt": prompt,
            "chosen": row["chosen"][len(prompt) :].strip(),
            "rejected": row["rejected"][len(prompt) :].strip(),
        }

    ds = ds.map(
        split_prompt_and_responses,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc="get_hh/split_prompt_and_responses",
    )
    return ds


def get_os(num_proc: int = 1, load_from_cache_file: bool = True):
    """get and process openai/summarize_from_feedback dataset."""
    ds = load_dataset("openai/summarize_from_feedback", "comparisons")
    cnndm_batches = ["batch0_cnndm", "cnndm0", "cnndm2"]
    tldr_format_str = (
        "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"
    )
    cnndm_format_str = "Article:\n{article}\n\nTL;DR:"

    def format_process(row):
        format_str = (
            cnndm_format_str if row["batch"] in cnndm_batches else tldr_format_str
        )
        prompt = HUMAN_PREFIX + format_str.format(**row["info"]) + ASSISTANT_PREFIX
        choice = row["choice"]
        # need to remove the leading space
        chosen = row["summaries"][choice]["text"].strip()
        rejected = row["summaries"][1 - choice]["text"].strip()
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    ds = ds.map(
        format_process,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc="get_os/format_process",
    )
    ds = ds.select_columns(["prompt", "chosen", "rejected"])
    ds["test"] = ds.pop("validation")
    return ds


def get_dataset(name: str, num_proc: int = 1, load_from_cache_file: bool = True):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'os'."""
    if name == "shp":
        data = get_shp(num_proc, load_from_cache_file)
    elif name == "hh":
        data = get_hh(num_proc, load_from_cache_file)
    elif name == "os":
        data = get_os(num_proc, load_from_cache_file)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    for split, data_split in data.items():
        assert set(data_split.features.keys()) == {
            "prompt",
            "chosen",
            "rejected",
        }, f"Unexpected keys in dataset {split=} {data_split=}"
    return data


def get_datasets(config):
    if isinstance(config.datasets, str):
        ds = get_dataset(
            config.datasets,
            num_proc=config.num_proc,
            load_from_cache_file=config.load_from_cache_file,
        )
    elif isinstance(config.datasets, ListConfig):
        for name in config.datasets:
            assert isinstance(
                name, str
            ), f"{config.datasets} should be a list of str but got {name=}"
        ds_list = [
            get_dataset(
                name,
                num_proc=config.num_proc,
                load_from_cache_file=config.load_from_cache_file,
            )
            for name in config.datasets
        ]
        for d in ds_list:
            assert d.keys() == {"train", "test"}, f"Unexpected split in dataset, {d=}"
        ds = DatasetDict()
        for key in ["train", "test"]:
            ds[key] = concatenate_datasets([d[key] for d in ds_list])
    else:
        raise ValueError(f"{config.datasets=} should be either str or a list of str.")
    if config.dry_run:
        ds["train"] = ds["train"].select(range(config.global_train_batch_size * 10))
        ds["test"] = ds["test"].select(range(config.global_eval_batch_size * 10))
    return ds


def get_dataloader(
    config,
    tokenizer,
    load_from_cache_file=True,
):
    """create dataloader."""
    ds = get_datasets(config)

    ds = ds.map(
        partial(
            tokenize_row,
            tokenizer=tokenizer,
            max_prompt_length=config.max_prompt_length,
            max_length=config.max_length,
            label_pad_token_id=config.label_pad_token_id,
        ),
        num_proc=config.num_proc,
        load_from_cache_file=load_from_cache_file,
        desc="tokenize_row",
    )

    ds = ds.select_columns(
        [
            "chosen_input_ids",
            "chosen_attention_mask",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_attention_mask",
            "rejected_labels",
        ]
    )

    data_collator = DPODataCollatorWithPadding(
        max_length=config.max_length,
        label_pad_token_id=config.label_pad_token_id,
        pad_token_id=config.pad_token_id,
    )

    train_loader = DataLoader(
        ds["train"],
        batch_size=config.global_train_batch_size,
        shuffle=config.shuffle,
        drop_last=True,
        collate_fn=data_collator,
    )

    # TODO: drop_last as false after padding eval_loader up to multiple of global_batch_size
    eval_loader = DataLoader(
        ds["test"],
        batch_size=config.global_eval_batch_size,
        collate_fn=data_collator,
        drop_last=True,
    )
    return train_loader, eval_loader
