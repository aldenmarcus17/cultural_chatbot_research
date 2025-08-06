"""
------------------------------
Synthetic Dataset (for TAGGER)
------------------------------
Dataset is split into two sections:

    response: simulated user responses.
    cultural_labels: vectorized version of metadata tags.

Lines 1-239 of curated_train_dataset_tagger.json are generated with ChatGPT-3, reviewed, then edited by the author.
"""

import json

from pyarrow.ipc import new_file
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# initialize json file and ai model
training_data = 'curated_train_dataset_tagger.json'
model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
model.eval()

with open(training_data, 'r') as file:
    data = json.load(file)


# paraphrase function. note that the settings are adjusted to make the sentences readable but also vary in word use
def paraphrase(text, extra_tokens):
    """function will take sentence and rephrase it."""
    prompt = f"Rephrase the sentence, keeping same tone, style, and content:\n{text}\nParaphrased:"
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=extra_tokens,
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    clean_return = decoded_text.replace(prompt, "").strip()
    return clean_return


# rephrases both questions and answers, excluding the tags during the actual rephrasing
for example in data:
    new_response = paraphrase(example["response"], 35)
    example["response"] = new_response

# dumps onto new file
with open("synthetic_training_data.json", "w") as f:
    json.dump(data, f, indent=2)

