"""
----------------------------
Synthetic Dataset Generation
----------------------------
Dataset is split into three sections:

    input_text: simulated user question. note that it always begins with the following:
        "[pd=x.xx][indiv=x.xx][independent=x.xx][ua=x.xx] " (49 characters including space)
    target_output: expected response.
    labels: vectorized version of metadata tags.

Lines 1-579 of training_data.json are generated with ChatGPT-3, reviewed, then edited by the author.
"""

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# initializing documents and ai models
tag_len = 49
training_data = 'training_data.json'
model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
model.eval()

with open(training_data, "r") as file:
    data = json.load(file)


# paraphrase function. note that the settings are adjusted to make the sentences readable but also vary in word use
def paraphrase(text, extra_tokens):
    """
    Function will take sentence and rephrase it.
    """
    prompt = f"Rephrase the sentence, keeping same tone, style, and content:\n{text}\nParaphrased:"
    inputs = tokenizer(prompt, return_tensors="pt")
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
for interaction in data:
    tag = interaction['input_text'][:tag_len]
    original_question = interaction['input_text'][tag_len:]
    new_question = paraphrase(original_question, 35)
    interaction['input_text'] = tag + new_question

    original_response = interaction['target_output']
    new_response = paraphrase(original_response, 110)
    interaction['target_output'] = new_response

# dumps onto new file
with open("synthetic_training_data.json", "w") as f:
    json.dump(data, f, indent=2)
