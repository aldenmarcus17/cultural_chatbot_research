"""
Training DeepSeek Chatbot
This file contains code that uses the LoRA method to finetune the chatbot.

Note that for the inputs, each input will have a tag at the beginning that takes the following form:
[pd=0.0][indiv=0.0][independent=0.0][ua=0.0]

And for the adversarial debiasing, it will contain vectors that correspond to the above that will look like this:
[0.5, 0.5, 0.5, 0.5]

In google colab, had to lower version of numpy to 1.26.4
"""
import tag_adversary
import json
import os
from grl import grad_reverse
from dotenv import load_dotenv
from datasets import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

load_dotenv()

# constants
lora_rank = int(os.getenv("LORA_RANK"))
lora_alpha = int(os.getenv("LORA_ALPHA"))
lora_dropout = float(os.getenv("LORA_DROPOUT"))
main_lr = float(os.getenv("MAIN_LR"))
weight_decay = float(os.getenv("WEIGHT_DECAY"))

ADVERSARY_STRENGTH = 0.15
# TRAINING_FILE = "final_combined_dataset.json" # use this for none google colab related things
TRAINING_FILE = "/content/drive/MyDrive/University of Toronto/2024-2025/Extra Curriculars/Laidlaw/Research Stuff/Google Colab Laidlaw/final_combined_dataset.json"

ADVERSARY_WEIGHTS = "adversary_weights_v1.pt"

# auto-detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# will utilize deepseek 7b, which is open source as of july 2025
model_huggingface = "deepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_huggingface, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_huggingface, trust_remote_code=True).to(device)

# add special token
special_tokens_dict = {'additional_special_tokens': ['<|eor|>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# resize model embeddings to accommodate new token
model.resize_token_embeddings(len(tokenizer))

# LoRA configuration
lora_config = LoraConfig(r=lora_rank,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                         inference_mode=False,
                         init_lora_weights="gaussian",
                         task_type=TaskType.CAUSAL_LM)

deepseek_model = get_peft_model(model, lora_config)
deepseek_model.print_trainable_parameters()

# instantiating TagAdversary and embedder
adversary = tag_adversary.TagAdversary().to(device)
adversary.load_state_dict(torch.load(ADVERSARY_WEIGHTS))
for param in adversary.parameters():
    param.requires_grad = False
adversary.eval()

# dataset prep (use this commented block if it is not using google colab
# dataset = load_dataset("json", data_files={"train": TRAINING_FILE}, split="train")

# dataset prep (only for google colab)
with open(TRAINING_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

def add_eor_to_targets(dataset):
    for example in dataset:
        if not example["target_output"].endswith("<|eor|>"):
            example["target_output"] = example["target_output"].strip() + "<|eor|>"
    return dataset

# Add <|eor|> to the end of each target_output
data = add_eor_to_targets(data)

# Convert to Hugging Face dataset
dataset = Dataset.from_list(data)

split_dataset = dataset.train_test_split(test_size=0.2, seed=17)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]


def tokenize(example):
    """tokenizes the prompt while masking out the answer"""
    prompt = example["input_text"]
    response = example["target_output"]
    full_text = prompt + "\n" + response

    full_tokenize = tokenizer(full_text, padding="max_length", truncation=True, max_length=200)
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=200)
    prompt_len = len(prompt_tokens["input_ids"])

    # creating a tokenized version of full_text with prompt masked out, used for loss calculations
    labels = full_tokenize['input_ids'].copy()
    labels[:prompt_len] = [-100 for _ in range(prompt_len)]
    full_tokenize['labels'] = labels
    full_tokenize['cultural_labels'] = example['cultural_labels']

    return full_tokenize


# tokenizing dataset
tokenized_dataset = train_dataset.map(tokenize, batched=False)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'cultural_labels'])
val_tokenized = val_dataset.map(tokenize, batched=False)
val_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'cultural_labels'])

# using DataLoader to batch the datasets
train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=2)
val_dataloader = DataLoader(val_tokenized, shuffle=False, batch_size=2)

# optimizer (using AdamW to incoporate weight decay, prevents overfitting for small dataset)
optimizer = AdamW(deepseek_model.parameters(), lr=main_lr, weight_decay=weight_decay)

# instantiate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

# loss function
loss_fn = torch.nn.MSELoss()


# mean pool code
def mean_pool(hidden_states, attention_mask):
    """pools the embeddings together so that it is squeezed to an array of shape [1, 4096]"""
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked = hidden_states * mask
    summed = masked.sum(1)
    count = mask.sum(1).clamp(min=1e-9)
    return summed / count


# training configuration
epochs = 100
best_val_loss = float("inf")
patience = 10
patience_counter = 0

for epoch in range(epochs):
    # training section
    deepseek_model.train()
    train_loss = 0.0
    adv_loss_total = 0.0
    neutral_batches = 0

    for batch in train_dataloader:
        adv_loss = torch.tensor(0.0, device=device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        cultural_labels = batch["cultural_labels"].to(device)

        # gradient reversal, total loss calculation, only if it is [0.5, 0.5, 0.5, 0.5]
        neutral_mask = torch.all(torch.abs(cultural_labels - 0.5) < 1e-3, dim=1)

        if neutral_mask.any():
            outputs = deepseek_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True
            )

            hidden_layer = outputs.hidden_states[-1]
            pooled = mean_pool(hidden_layer, attention_mask)

            pooled_neutral = pooled[neutral_mask]
            labels_neutral = cultural_labels[neutral_mask].to(device).float()

            reversed_neutral = grad_reverse(pooled_neutral, lambda_=ADVERSARY_STRENGTH)

            preds_adversary = adversary(reversed_neutral)

            adv_loss = loss_fn(preds_adversary, labels_neutral)
            adv_loss_total += adv_loss.item()
            neutral_batches += 1

        else:
            outputs = deepseek_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                return_dict=True
            )

        # regular loss
        lm_loss = outputs.loss

        batch_loss = lm_loss + (ADVERSARY_STRENGTH * adv_loss)
        train_loss += batch_loss.item()

        # back propagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

    avg_train_loss = train_loss/len(train_dataloader)
    avg_adv_loss = adv_loss_total / neutral_batches if neutral_batches > 0 else 0.0

    # validation
    deepseek_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            cultural_labels = batch["cultural_labels"].to(device)

            outputs = deepseek_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                return_dict=True
            )

            lm_loss = outputs.loss
            val_loss += lm_loss.item()

    avg_val_loss = val_loss/len(val_dataloader)
    print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Adv Loss: {avg_adv_loss:.4f}")

    if avg_val_loss < best_val_loss:
        print("model improved! saving model...")
        best_val_loss = avg_val_loss
        patience_counter = 0
        deepseek_model.save_pretrained("checkpoints/best_lora_model")
    else:
        patience_counter += 1
        print(f"no improvement. patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("early stop.")
            break
    
    if patience_counter == 4:
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] * 0.5

    scheduler.step(avg_val_loss)







