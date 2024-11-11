import torch
from gpt_sft_dataset import GPTSFTDatasetHF

from nemo.utils import logging
from nemo_aligner.utils import parallel_state
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
    MegatronPretrainingRandomBatchSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)


def build_dataloader(
    cfg,
    dataset,
    consumed_samples,
    mbs,
    gbs,
    drop_last=True,
    pad_samples_to_global_batch_size=False,
    collate_fn=None,
    load_gbs=True,
    use_random_sampler=True,
):
    """Buld dataloader given an input dataset."""

    logging.info(f"Building dataloader with consumed samples: {consumed_samples}")
    # Common parameters for batch sampler creation
    common_params = {
        "total_samples": len(dataset),
        "consumed_samples": consumed_samples,
        "micro_batch_size": mbs,
        "data_parallel_rank": parallel_state.get_data_parallel_rank(),
        "data_parallel_size": parallel_state.get_data_parallel_world_size(),
        "drop_last": drop_last,
        "global_batch_size": gbs,
        "pad_samples_to_global_batch_size": pad_samples_to_global_batch_size,
    }

    if use_random_sampler:
        cls = (
            MegatronPretrainingRandomBatchSampler
            if load_gbs
            else MegatronPretrainingRandomSampler
        )
        common_params["seed"] = cfg.model.seed
    else:
        cls = (
            MegatronPretrainingBatchSampler if load_gbs else MegatronPretrainingSampler
        )
    batch_sampler = cls(**common_params)

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.model.data.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def build_sft_dataset(
    data_cfg,
    indexed_dataset,
    tokenizer,
    num_samples,
    answer_only_loss=True,
    is_chat=True,
    special_tokens=None,
):
    packed_sequence = data_cfg.get("packed_sequence", False)
    dataset_kwargs = {}

    assert is_chat == False and packed_sequence == False
    dataset_cls = GPTSFTDatasetHF

    dataset = dataset_cls(
        indexed_dataset=indexed_dataset,
        file_path=data_cfg.file_path,
        tokenizer=tokenizer,
        max_seq_length=data_cfg.max_seq_length,
        min_seq_length=data_cfg.min_seq_length,
        add_bos=data_cfg.get("add_bos", False),
        add_eos=data_cfg.get("add_eos", True),
        add_sep=data_cfg.get("add_sep", False),
        sep_id=0,
        max_num_samples=num_samples,
        seed=data_cfg.get("seed", 1234),
        label_key=data_cfg.get("label_key", "answer"),
        answer_only_loss=answer_only_loss,
        truncation_field=data_cfg.get("truncation_field", "text"),
        pad_to_max_length=data_cfg.get("pad_to_max_length", False),
        index_mapping_dir=data_cfg.get("index_mapping_dir", None),
        prompt_template=data_cfg.get("prompt_template", None),
        virtual_tokens=0,
        memmap_workers=data_cfg.get(
            "memmap_workers", None
        ),  # used to set num. of workers to create the memmap index files
        hf_dataset=data_cfg.get(
            "hf_dataset", False
        ),  # Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
        truncation_method=data_cfg.get(
            "truncation_method", "right"
        ),  # used to choose truncation method. Options: ['random', 'left', 'right']
        special_tokens=special_tokens,
        output_original_text=data_cfg.get("output_original_text", False),
        **dataset_kwargs,
    )
    return dataset


def get_input_pipeline(config, train_dataset, test_dataset, tokenizer, model_tokenizer):
    """get input_pipeline."""
    if config.model.data.data_impl == "mock":
        raise ValueError("Not supported: synthetic data")
    else:

        def decode_ids(examples):
            result = {
                "input": tokenizer.batch_decode(examples["input_ids"]),
                "output": tokenizer.batch_decode(examples["labels"]),
            }
            return result

        train_dataset = train_dataset.map(
            decode_ids,
            batched=True,
        )
        test_dataset = test_dataset.map(
            decode_ids,
            batched=True,
        )
        if config.model.data.get("sample", False):
            # if it is negative, num_samples is None
            if config.trainer.sft.max_steps < 0:
                num_samples = None
            else:
                num_samples = (
                    config.trainer.sft.max_steps * train_data_cfg.global_batch_size
                )
        else:
            num_samples = None

        train_data_cfg = config.model.data.train_ds
        val_data_cfg = config.model.data.validation_ds

        consumed_samples = 0
        train_ds = build_sft_dataset(
            train_data_cfg,
            train_dataset,
            model_tokenizer,
            num_samples,
            answer_only_loss=True,
            is_chat=config.model.data.chat,
            special_tokens=config.model.data.chat_prompt_tokens,
        )
        if config.model.data.get("sample", False):
            num_samples = (
                config.trainer.sft.limit_val_batches * val_data_cfg.global_batch_size
            )
        else:
            num_samples = None
        validation_ds = build_sft_dataset(
            val_data_cfg,
            test_dataset,
            model_tokenizer,
            num_samples,
            answer_only_loss=True,
            is_chat=config.model.data.chat,
            special_tokens=config.model.data.chat_prompt_tokens,
        )

        train_dataloader = build_dataloader(
            cfg=config,
            dataset=train_ds,
            consumed_samples=consumed_samples,
            mbs=train_data_cfg.micro_batch_size,
            gbs=train_data_cfg.global_batch_size,
            collate_fn=train_ds.collate_fn,
            drop_last=train_data_cfg.drop_last,
            pad_samples_to_global_batch_size=not train_data_cfg.drop_last,
            load_gbs=True,
        )

        val_dataloader = build_dataloader(
            cfg=config,
            dataset=validation_ds,
            consumed_samples=0,
            mbs=val_data_cfg.micro_batch_size,
            gbs=val_data_cfg.global_batch_size,
            collate_fn=validation_ds.collate_fn,
            drop_last=val_data_cfg.drop_last,
            pad_samples_to_global_batch_size=not val_data_cfg.drop_last,
            load_gbs=True,
            use_random_sampler=False,
        )
    return train_dataloader, val_dataloader, train_ds, validation_ds
