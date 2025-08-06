"""Merges the LoRA parameters with the DeepSeek 7B model"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# instantiating pretrained model
pretrained_model_path = "deepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, trust_remote_code=True)
pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, trust_remote_code=True)

# add special token
special_tokens_dict = {'additional_special_tokens': ['<|eor|>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# resize model embeddings to include new token
pretrained_model.resize_token_embeddings(len(tokenizer))

# loading adapter
current_dir = os.path.abspath(os.path.dirname(__file__))
lora_path = os.path.join(current_dir, "best_lora_model_v4")
assert os.path.exists(lora_path), f"Adapter not found at: {lora_path}"
lora_model = PeftModel.from_pretrained(pretrained_model, lora_path, is_local=True)

# merging
final_model = lora_model.merge_and_unload()
final_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# saving
final_model.save_pretrained("deepseek_finetuned_lora_model_v4")
tokenizer.save_pretrained("deepseek_finetuned_lora_model_v4")

