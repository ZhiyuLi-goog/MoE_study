import os
from itertools import chain

import transformers
from datasets import Features, Value, concatenate_datasets, load_dataset
from transformers import logging
from transformers.testing_utils import CaptureLogger

logger = logging.get_logger(__name__)

import pytorch_lightning as pl
from torch.utils.data import DataLoader


def get_datasets(config):
    # Downloading and loading a dataset from the hub.
    if config.dataset.dataset_name == "c4_mlperf":
        train_data_files = {
            "train": [
                f'{os.path.join(config.dataset.train_dataset_path, f"c4-train.{i:05d}-of-01024.json")}'
                for i in range(768, 1024)
            ],
        }
        eval_data_files = {
            "validation": [
                f'{os.path.join(config.dataset.eval_dataset_path, "c4-validation_24567exp.json")}'
            ],
        }
        features = Features(
            {
                "text": Value(dtype="string", id=None),
                "timestamp": Value(dtype="string", id=None),
                "url": Value(dtype="string", id=None),
            }
        )
        raw_datasets = {
            "train": load_dataset(
                "json",
                data_files=train_data_files,
                features=features,
                cache_dir=config.cache_local_dir,
                streaming=config.dataset.streaming,
                split="train",
            ),
            "validation": load_dataset(
                "json",
                data_files=eval_data_files,
                features=features,
                cache_dir=config.cache_local_dir,
                split="validation",
            ),
        }
        if config.n_eval_examples:
            raw_datasets["validation"] = raw_datasets["validation"].select(
                range(config.n_eval_examples)
            )
    else:
        raw_datasets = load_dataset(
            config.dataset.dataset_name,
            config.dataset.dataset_config_name,
            cache_dir=config.cache_local_dir,
            streaming=config.dataset.streaming,
        )
    return raw_datasets


def process_datasets(raw_datasets, tokenizer, config):
    # First we tokenize all the texts.
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def process_datasets_function(src_datasets, function, desc):
        tgt_datasets = {}
        for key in src_datasets.keys():
            # use validation batch_size to avoid dropping remainders in group_text
            batch_size = 24567 if key == "validation" else 4096
            # only apply streaming in train dataset
            if key == "train" and config.dataset.streaming:
                tgt_datasets[key] = src_datasets[key].map(
                    function,
                    batched=True,
                    batch_size=batch_size,
                )
            else:
                tgt_datasets[key] = src_datasets[key].map(
                    function,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=config.dataset.num_proc,
                    load_from_cache_file=config.dataset.load_from_cache_file,
                    desc=desc,
                )
        return tgt_datasets

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    tokenized_datasets = process_datasets_function(
        raw_datasets, tokenize_function, desc="Running tokenizer on dataset"
    )
    tokenized_datasets = {
        key: dataset.remove_columns(column_names)
        for key, dataset in tokenized_datasets.items()
    }
    block_size = config.max_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length % block_size != 0:
            pad_length = (total_length // block_size + 1) * block_size - total_length
            for k in concatenated_examples.keys():
                concatenated_examples[k].extend([config.pad_token_id] * pad_length)
            total_length += pad_length
        else:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = process_datasets_function(
        tokenized_datasets,
        group_texts,
        desc=f"Grouping texts in chunks of {block_size}",
    )
    if config.shuffle:
        lm_datasets["train"] = lm_datasets["train"].shuffle(
            seed=config.seed, buffer_size=config.dataset.shuffle_buffer_size
        )

    # pad to multiple of batch size in eval/validation dataset
    if len(lm_datasets["validation"]) % config.global_eval_batch_size > 0:
        num_eval_batches = (
            len(lm_datasets["validation"]) // config.global_eval_batch_size + 1
        )
        pad_number = num_eval_batches * config.global_eval_batch_size - len(
            lm_datasets["validation"]
        )
        logger.info(
            f"Eval data has {len(lm_datasets['validation'])} entries, padding now with "
            f"{pad_number} extra entries to get {num_eval_batches * config.global_eval_batch_size} batches."
        )

        def mask_pad(examples):
            examples["labels"] = [config.pad_token_id] * len(examples["labels"])
            return examples

        pad_validation_dataset = (
            lm_datasets["validation"].select(range(pad_number)).map(mask_pad)
        )
        lm_datasets["validation"] = concatenate_datasets(
            [lm_datasets["validation"], pad_validation_dataset]
        )

    return lm_datasets


class DatasetModule(pl.LightningDataModule):
    def __init__(self, train_dataset, eval_dataset):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=1)
