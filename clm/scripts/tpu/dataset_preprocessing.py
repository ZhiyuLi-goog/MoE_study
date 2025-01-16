from clm_datasets import get_dataset_cuda, get_datasets, process_datasets

# need to run in cpu with single process
# to walk around undefined `OmegaConf.register_new_resolver` need to overwrite `run_dir` `global_train_batch_size` `global_eval_batch_size`
# python clm_datasets.py model.name_or_path=mistralai/Mixtral-8x22B-v0.1 run_dir=/tmp global_train_batch_size=1 global_eval_batch_size=1 max_length=32768
@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        add_eos_token=False,
        add_bos_token=False,
        use_fast=False,
    )
    raw_datasets = get_datasets(config)
    lm_datasets = process_datasets(raw_datasets, tokenizer, config)

    lm_datasets.save_to_disk("/mnt/dataset/tokenized/")

    # for i, batch in enumerate(lm_datasets["validation"]):
    #     print(f"{i=}: {batch=}")


if __name__ == "__main__":
    main()