from datasets import load_dataset, Features, Value
from transformers.testing_utils import CaptureLogger
import transformers
from itertools import chain


def get_datasets(config):
    # Downloading and loading a dataset from the hub.
    if config.dataset.dataset_name == "c4_mlperf":
        data_files = {
            # "train": [f"gs://lizhiyu-multipods-eu-west/c4/en_json/3.0.1/c4-train.{i:05d}-of-01024.json" for i in range(768, 1024)],
            "train": ["gs://lizhiyu-multipods-eu-west/c4/en_val_subset_json/c4-validation_24567exp.json"],
            "validation": ["gs://lizhiyu-multipods-eu-west/c4/en_val_subset_json/c4-validation_24567exp.json"],
        }
        features = Features(
            {
                'text': Value(dtype='string', id=None),
                'timestamp': Value(dtype='string', id=None),
                'url': Value(dtype='string', id=None),
            }
        )
        raw_datasets = {}
        raw_datasets['train'] = load_dataset(
            "json",
            data_files=data_files['train'],
            features=features,
            cache_dir=config.cache_local_dir,
            streaming=config.dataset.streaming,
            split='train',
        ).shuffle(seed=42, buffer_size=10_000)
        raw_datasets['validation'] = load_dataset(
            "json",
            data_files=data_files['validation'],
            features=features,
            cache_dir=config.cache_local_dir,
            split='train',
        )
        if config.dry_run:
            raw_datasets['validation'] = raw_datasets["validation"].select(range(100))
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
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

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
    
    if not config.dataset.streaming:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=config.dataset.num_proc,
            remove_columns=column_names,
            load_from_cache_file=config.dataset.load_from_cache_file,
            desc="Running tokenizer on dataset",
        )
    else:
        tokenized_datasets = {}
        tokenized_datasets['train'] = raw_datasets['train'].map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )
        tokenized_datasets['validation'] = raw_datasets['validation'].map(
            tokenize_function,
            batched=True,
            num_proc=config.dataset.num_proc,
            remove_columns=column_names,
            load_from_cache_file=config.dataset.load_from_cache_file,
            desc="Running tokenizer on dataset",
        )
    
    block_size = config.max_length
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    if not config.dataset.streaming:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=config.dataset.num_proc,
            load_from_cache_file=config.dataset.load_from_cache_file,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    else:
        lm_datasets = {}
        lm_datasets['train'] = tokenized_datasets['train'].map(
            group_texts,
            batched=True,
        )
        lm_datasets['validation'] = tokenized_datasets['validation'].map(
            group_texts,
            batched=True,
            num_proc=config.dataset.num_proc,
            load_from_cache_file=config.dataset.load_from_cache_file,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    return lm_datasets
