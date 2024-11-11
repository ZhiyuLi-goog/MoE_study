from typing import Mapping, Optional
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import (
    GPTSFTDataset,
)
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


class GPTSFTDatasetHF(GPTSFTDataset):
    def __init__(
        self,
        indexed_dataset,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        pad_seq_length_to_mult: int = 16,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: int = None,
        max_num_samples: int = None,
        seed: int = 1234,
        label_key: str = "answer",
        answer_only_loss: bool = True,
        truncation_field: str = "text",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        index_mapping_dir: str = None,
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        memmap_workers: Optional[int] = None,
        hf_dataset: bool = False,
        global_sample_mapping: bool = False,
        truncation_method: str = "right",
        special_tokens: Optional[
            Mapping[str, str]
        ] = None,  # special tokens, a dictory of {token_type: token}
        is_test: bool = False,
        output_original_text: bool = False,
        ceil_to_power_2: bool = False,
        get_attention_mask_from_fusion: bool = False,
    ):
        """
        file_path: Path to a JSONL GPT supervised fine-tuning dataset. Data is formatted as multiple JSON lines with each line formatted as follows. {'input': 'John von Neumann\nVon Neumann made fundamental contributions .... Q: What did the math of artificial viscosity do?', 'output': 'smoothed the shock transition without sacrificing basic physics'}
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec (ex: SentencePiece).
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        seed: int = 1234,
        label_key: Key to use for the label in your JSONL file
        answer_only_loss: If True, will compute the loss only on the answer part of the input. If False, will compute the loss on the entire input.
        truncation_field: Field to use for truncation. (Options: keys in prompt_template). Field to be used for truncation if the combined length exceeds the max sequence length.
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        index_mapping_dir: Directory to save the index mapping to. If None, will write to the same folder as the dataset.
        prompt_template: Prompt template to inject via an fstring. Formatted like Q: {context_key}\n\nA: {label_key}
        hf_dataset: Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
        global_sample_mapping: Whether to shuffle all data together, or shuffle the dataset within each epoch
        truncation_method: Truncation from which position. Options: ['left', 'right']
        special_tokens: special tokens for the chat prompts, a dictionary of {token_type: token}. Default: {'system_turn_start': '<extra_id_0>', 'turn_start': '<extra_id_1>', 'label_start': '<extra_id_2>', 'end_of_turn': '\n', "end_of_name": "\n"}
        is_test: Whether this dataset is the test split.
        output_original_text (bool): if true, will keep the original text in the output alongside the tokenized ids.
        """
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.pad_seq_length_to_mult = pad_seq_length_to_mult
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.sep_id = sep_id
        self.max_num_samples = max_num_samples
        self.seed = seed
        self.label_key = label_key
        self.answer_only_loss = answer_only_loss
        self.truncation_fields = (
            truncation_field.split(",") if truncation_field is not None else []
        )
        self.pad_to_max_length = pad_to_max_length
        self.index_mapping_dir = index_mapping_dir
        self.prompt_template = prompt_template
        self.virtual_tokens = virtual_tokens
        self.tokens_to_generate = tokens_to_generate
        self.memmap_workers = memmap_workers
        self.hf_dataset = hf_dataset
        self.global_sample_mapping = global_sample_mapping
        self.truncation_method = truncation_method
        self.is_test = is_test
        self.output_original_text = output_original_text
        self.ceil_to_power_2 = ceil_to_power_2
        self.get_attention_mask_from_fusion = get_attention_mask_from_fusion

        if special_tokens is None:
            self.special_tokens = {
                "system_turn_start": "<extra_id_0>",
                "turn_start": "<extra_id_1>",
                "label_start": "<extra_id_2>",
                "end_of_turn": "\n",
                "end_of_name": "\n",
            }
        else:
            self.special_tokens = special_tokens

        self.indexed_dataset = indexed_dataset

        # Validate prompt template
        self._maybe_validate_prompt_template()

        # Will be None after this call if `max_num_samples` is None
        self._build_samples_mapping()
