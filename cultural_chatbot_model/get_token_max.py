"""
This file is not ran during the training of the chatbot. It is used to figure out optimal token lengths.

Information gathered:
    --> within 'final_combined_dataset.json', the maximum tokenized prompt length is 157
    --> the average is 102
    --> for the sake of this concept model, a 'max_length' of 200 will be utilized
"""
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

# auto-detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# import tokenizer of deepseek llm 7b
model_huggingface = "deepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_huggingface, trust_remote_code=True)

# dataset prep
dataset = load_dataset("json", data_files="../dataset_generation/final_combined_dataset.json")["train"]
split_dataset = dataset.train_test_split(test_size=0.2, seed=17)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]


def get_max_token_length(dataset):
    """gets the maximum token length of a dataset entry. use to optimize the 'max_length' parameter
    during tokenization."""
    token_lengths = [
        len(tokenizer(example["input_text"] + "\n" + example["target_output"], truncation=False)["input_ids"])
        for example in dataset]

    print(f"max length: {max(token_lengths)} and average length: {sum(token_lengths) // len(token_lengths)}")


get_max_token_length(dataset)
