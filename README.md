# Culturally Responsive Chatbots
A prototype chatbot that uses DeepSeek's open-source ML model and integration of various psychology theories such as Hofstede's cultural dimensions to create more culturally responsive chatbots. 
Note that my chatbot is only scaled to teach about the CN tower, however the system design can be used for larger applications.
A more in-depth literature review and framework section will be written soon.

## Summary
This project explores how culture can be better integrated within educational AI. By finetuning a [DeepSeek 7B Model](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base) and attatching a cultural tagger and
an adversarial debiaser, the hope is that chatbot can generate responses that are both contextually relevant and culturally sensitive, minimizing unintended biases while adapting its tone and style to better align with diverse user backgrounds.

## Features
- Cultural tagging system based on a 4-dimensional vector
- LoRA finetuning on DeepSeek 7B
- Adversarial debiasing to reduce implicit cultural bias
- React frontend/FastAPI backend
- Supports dynamic prompting and tone modulation for questions surrounding the CN tower

## Installation
The finetuned LoRA model is uploaded to HuggingFaces [here](https://huggingface.co/auldo/cn_cultural_chatbot_v1), however to use the full tagging system or to use the prebuilt website, follow these instructions:
```python
# clone repo
git clone https://github.com/aldenmarcus17/cultural_chatbot_research.git

# navigate to project
cd cultural_chatbot_research

# Create virtual env
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# install Python dependencies
pip install -r requirements.txt

# start frontend
cd frontend
npm install
npm run dev

# since the 7B model is quite large, I recommend running the backend on Google Colab using the free T4 GPU
# you can then expose the backend using ngrok and connect it to the frontend (change link in Chatbox.jsx)
# in Google Colab, upload the website_backend folder then run...
!python3 main.py
```

## Project Structure
``` bash
├── cultural_chatbot_model/        # contains training code for the model
│   ├── adversary/                      # adversarial debiaser
│   ├── cultural_tagger/                # tagger code
├── dataset_generation/            # contains code for how the datasets were generated, along with the datasets themselves
├── frontend/                      # folder containing React frontend
├── website_backend/               # contains backend code (FastAPI)
│   ├── tagger_backend/            # contains backend for tagger
├── .env                           # not commited, you will have to make your own .env and instantiate values for training or for keys
├── .gitignore
├── README.md
```

## Training Details
- **Model Base**: DeepSeek LLM 7B
- **Fine-Tuning**: PEFT and LoRA
- **Adversary**: 3-layer adversary that mitigates cultural leakage in "neutral" prompts
- **Loss**: Combined language modeling loss and adversarial loss (using gradient reversal layer)

## Cultural Tagging
The tagging system uses 4-dimensional vectors that have each value falling within the range of 0-1. Within the code, it is represented like this:  
  
**[pd=0.0][indiv=0.0][independent=0.0][ua=0.0]**  

- pd --> power distance
- indiv --> individualism vs. collectivism
- independent --> independent vs. interdependent self-construals
- ua --> uncertainty avoidance  

If you are unfamiliar with these concepts, please look into *Hofstede's Cultural Dimensions* and *Markus and Kitayama's Theory of Self-Construal*.
Each time the user enters a prompt, their prompt is tagged with a predicted cultural vector that slowly impacts their saved cultural profile. 
The tagging is done by a simple 2-layer neural net that takes in BERT style embeddings and outputs a 4-dim vector. The DeepSeek model is fine-tuned 
to take in string versions of these cultural vectors and output a response that is more culturally in tune with the user.

```python
# here is an example tagged input into the chatbot
prompt = "[pd=0.30][indiv=0.70][independent=0.70][ua=0.20] What is the objective of the CN Tower?"
```

## Adversary 
The adversary was only used in training for examples where the prompt was tagged as neutral ([0.5, 0.5, 0.5, 0.5]). The goal for the main model was
to produce truly neutral outputs. Since the meaning of "culturally neutral" is very subjective, the main model should aim to produce neutral outputs that
do not leak cultural information detectable by the adversary, even when trained on diverse cultural examples. This encourages the model to internalize 
cultural sensitivity without explicitly relying on cultural tags during inference. In simpler terms, the adversary should not be able to predict neutral examples as neutral, as if it does, this indicates that the main model is following 
some kind of pattern to produce neutral responses, making the model itself not culturally neutral.

## Limitations
### Dataset
The datasets were generated using ChatGPT-3 and were all closely read over and edited. However, due to the nature of culture, the datasets themselves are likely
to be skewed, specifically towards a Western audience. However, the overarching goal is to show how curated datasets are far more culturally diverse and equal.
Additionally, due to the sheer amount of work it takes to make a curated dataset, I was not able to create huge datasets (the main model and adversary is trained on ~500 examples and
the tagger is trained on ~250 examples). Thus, the scope of topics the chatbot can respond effectively to is extremely limited.
### Tags
The tags are again set by me. This was based off of what I learned through my literature review, however I am inherently biased and this should be taken into account.
### Cultural Support
Realistically, the chatbot can only effectively teach people within my culture. If this project were to be expanded on in the future, I would hope that people from other cultures can have
input on the datasets so that the final output has a larger cultural scope.
