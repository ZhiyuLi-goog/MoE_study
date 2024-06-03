import torch
from transformers import AutoTokenizer
from preference_datasets import get_batch_iterator
from huggingface_hub import login
import tqdm

if __name__ == "__main__":
    token = "<hf token"
    login(token=token)

    model_id = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for dataset in ['shp', 'hh']:
        num_tokens_chosen_input = 0
        num_tokens_rejected_input = 0
        num_samples = 0
        for batch in tqdm.tqdm(get_batch_iterator([dataset], tokenizer, max_length=65536, max_prompt_length=65536, n_epochs=1, shuffle=False)):
            num_tokens_chosen_input += torch.count_nonzero(batch['chosen_input_ids']).numpy()
            num_tokens_rejected_input += torch.count_nonzero(batch['rejected_input_ids']).numpy()
            num_samples += 1
    
        print(f"{dataset}: {num_samples=}")
        print(f"{dataset}: {num_tokens_chosen_input=}")
        print(f"{dataset}: {num_tokens_rejected_input=}")
        avg_tokens_chosen_input = num_tokens_chosen_input / num_samples
        avg_tokens_rejected_input = num_tokens_rejected_input / num_samples
        print(f"{dataset}: {avg_tokens_chosen_input=}")
        print(f"{dataset}: {avg_tokens_rejected_input=}")

      

