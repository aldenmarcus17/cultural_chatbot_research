from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from sentence_transformers import SentenceTransformer, util
from tagger_backend.tagger import Tagger
import os
import re
from dotenv import load_dotenv
import random

load_dotenv()

hf_model_id = os.getenv("HF_MODEL_ID")
weights_path = "./tagger_backend/tagger_weights_v1.pt"
initial_cultural_vector = [0.39, 0.80, 0.80, 0.48] # default cultural vector, see 6dim_hofsted.csv for more details 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CulturalChatbot():
    """Class responsible for calling the cultural chatbot model and returning the response"""

    def __init__(self, model_path=hf_model_id):
        print(f"Loading model and tokenizer from Hugging Face repo '{hf_model_id}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=bnb_config,
                        device_map="auto",  # lets it auto-place on available GPU
                        trust_remote_code=True
                    )
        self.model.eval()

        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        self.embedder.eval()

        self.tagger = Tagger().to(device)
        self.tagger.load_state_dict(torch.load(weights_path, map_location=device))
        self.tagger.eval()

        self.eor_token_id = self.tokenizer.convert_tokens_to_ids("<|eor|>")
        self.cultural_vector = initial_cultural_vector

        self.cn_tensor = torch.tensor([-3.0707e-02,  4.9626e-02,  5.5259e-03, -4.0710e-04, -4.3583e-02,
                                        6.6558e-03, -7.1887e-02,  2.8505e-02, -8.0233e-03,  2.2807e-02,
                                        1.8559e-03,  9.6098e-03,  1.9243e-02, -2.5250e-02, -5.1329e-02,
                                        6.6377e-03,  7.5310e-03, -3.7781e-03,  4.1206e-02,  7.5307e-03,
                                        -1.8208e-02, -6.1012e-02, -1.0112e-03,  9.2154e-03, -1.1179e-02,
                                        3.2282e-02, -5.2124e-02,  8.3446e-02,  8.0922e-02, -2.3328e-02,
                                        -7.8534e-03,  1.6436e-02, -3.9685e-02,  4.9962e-02, -4.0662e-03,
                                        4.7899e-02,  7.4531e-02, -6.4904e-02,  9.5555e-04, -3.4613e-03,
                                        1.4437e-02,  6.3508e-03,  4.9153e-02,  4.2719e-02,  3.1487e-02,
                                        2.4366e-02, -1.7480e-03, -5.5988e-02,  1.2323e-02, -6.3442e-02,
                                        6.1800e-02, -2.0246e-02,  1.5742e-02, -1.9782e-02,  1.9487e-02,
                                        1.0326e-01, -5.2356e-03, -6.1260e-02,  7.4483e-02,  5.1136e-02,
                                        4.0924e-02,  2.7070e-02, -7.0120e-02,  2.8784e-03,  9.4241e-02,
                                        -3.2426e-02, -8.4358e-02, -3.3883e-02,  2.0556e-02, -3.6000e-02,
                                        8.6133e-02, -7.4134e-02,  2.1188e-02, -1.6343e-02, -1.3063e-02,
                                        5.0470e-03, -1.9914e-03, -2.9764e-02, -3.3126e-02,  8.1784e-05,
                                        -1.8162e-02,  1.1584e-02, -4.7949e-03,  6.1517e-02,  2.1133e-02,
                                        5.1582e-02, -2.2019e-02,  5.8414e-02, -4.1777e-02, -3.0670e-02,
                                        3.5439e-02, -1.4728e-02, -9.9745e-02,  5.6468e-02, -3.1344e-02,
                                        -1.5281e-02, -2.1970e-02, -1.4489e-03, -1.3048e-02, -9.6192e-03,
                                        5.9130e-02,  8.9261e-02,  3.2599e-02,  4.8911e-03, -3.2518e-02,
                                        -2.4672e-02,  1.2669e-02,  6.3457e-02, -9.1668e-03, -1.3082e-02,
                                        -7.2513e-02, -1.0720e-02, -3.8275e-02, -1.3839e-02, -4.9040e-02,
                                        1.7144e-02,  2.9906e-02, -2.6900e-02, -1.5764e-02,  1.0433e-03,
                                        4.8869e-02,  6.3348e-03, -2.9002e-02,  2.5447e-02, -4.4969e-02,
                                        -5.0187e-02, -3.8366e-02, -5.3749e-33, -2.5669e-02,  2.9082e-02,
                                        3.6241e-03,  4.3207e-02,  9.2683e-02,  1.3505e-02, -2.0477e-02,
                                        5.5222e-03,  3.6955e-03,  6.3439e-02, -2.5415e-02,  2.1367e-02,
                                        -2.9493e-02, -3.5830e-02,  5.4901e-02, -7.6374e-02, -4.0195e-02,
                                        -7.1545e-02, -5.5269e-02, -3.4495e-02,  3.8324e-02, -3.1155e-02,
                                        4.7747e-03,  7.5494e-02,  9.0422e-02, -4.2519e-02, -4.2630e-02,
                                        3.5926e-02, -1.1553e-02,  9.9414e-03, -4.7249e-03, -1.2503e-02,
                                        -5.1123e-02, -9.2191e-04,  2.2305e-02,  5.8641e-03,  3.2794e-02,
                                        -3.2920e-02, -3.6075e-02,  2.0661e-02,  1.1419e-03,  3.4706e-03,
                                        -2.1864e-02,  2.5522e-02,  9.6153e-03,  3.8657e-02,  2.2688e-02,
                                        -8.7772e-02,  3.6810e-02, -8.4165e-02,  1.0264e-02,  4.0889e-02,
                                        -9.0252e-02, -5.0071e-02,  4.6119e-02, -2.0056e-02,  4.4601e-02,
                                        -1.8383e-02,  3.6179e-02,  5.0418e-02,  1.2370e-02, -4.5134e-03,
                                        -7.0524e-02,  6.1637e-02,  1.8400e-02,  1.6942e-02, -1.7211e-02,
                                        1.7773e-02,  3.2271e-02, -3.6192e-02, -1.8618e-02,  1.5087e-02,
                                        6.0383e-02, -5.6265e-02, -2.8516e-02,  4.2526e-02, -1.5432e-01,
                                        -4.8006e-03, -6.0604e-03,  1.4353e-02, -5.1011e-02, -7.3887e-03,
                                        2.1444e-02,  6.2641e-02,  8.6520e-02, -3.9887e-02,  3.7008e-02,
                                        4.0269e-03, -1.0117e-02, -6.5198e-03, -4.5694e-02, -4.9241e-02,
                                        6.9508e-02, -5.6826e-02, -7.7401e-02,  2.8006e-33, -8.5143e-02,
                                        1.0787e-02, -4.3575e-02, -4.9314e-02, -2.1887e-02, -2.3712e-02,
                                        -1.5731e-02,  4.6230e-03,  7.4448e-03,  3.3130e-02, -1.9575e-02,
                                        2.0920e-02,  3.8900e-02,  2.1760e-02,  1.2447e-02,  4.8869e-02,
                                        -3.2013e-02, -1.1892e-02, -4.7704e-02,  3.3592e-02,  4.1748e-02,
                                        2.8206e-02, -8.4032e-02,  1.4435e-02,  8.1508e-03,  8.2169e-02,
                                        -3.7737e-02, -7.4047e-02,  4.5938e-02,  2.1071e-02, -1.0425e-01,
                                        -1.4101e-02, -1.3258e-02,  7.3214e-02, -8.7927e-03,  6.5797e-02,
                                        4.8510e-02,  8.2948e-03,  9.8398e-03, -3.8034e-02,  2.8702e-02,
                                        3.5915e-02,  1.1151e-02,  3.9120e-02,  2.3048e-02,  1.0482e-02,
                                        -2.1475e-02, -3.0673e-03, -6.3705e-02,  1.3849e-02, -1.3855e-02,
                                        -1.2277e-03, -1.7529e-02,  1.1950e-02,  1.7288e-02, -3.2714e-02,
                                        -7.0030e-03, -2.6362e-03,  1.5816e-02, -5.2488e-02,  8.0210e-02,
                                        -7.4330e-03, -8.4046e-02,  1.1665e-02,  1.4420e-02,  6.4076e-02,
                                        1.7769e-02,  6.3017e-02, -6.7793e-03,  3.0244e-02, -8.4610e-03,
                                        3.9649e-02,  1.3278e-03, -4.1308e-02, -8.9460e-02, -1.3668e-04,
                                        3.5675e-02,  4.8213e-02, -1.5921e-02,  1.7905e-04, -4.9721e-02,
                                        -1.3447e-02, -1.5369e-02, -5.0299e-02,  9.0372e-03, -2.2732e-03,
                                        6.0352e-02, -2.9380e-03, -2.6923e-02, -1.7367e-02,  4.8849e-04,
                                        -2.5898e-02, -2.7846e-02, -2.4853e-02, -3.3109e-02, -1.6195e-08,
                                        -4.2199e-02,  2.3804e-02, -6.0824e-02, -6.5225e-02,  4.5176e-02,
                                        -9.2062e-02,  7.3566e-02,  4.3089e-02,  1.7222e-02,  1.8195e-02,
                                        2.9992e-02, -2.1450e-02,  1.0448e-02,  2.0250e-02,  4.2340e-02,
                                        2.3053e-02, -6.0398e-02, -1.7905e-02,  9.9578e-03,  2.0111e-02,
                                        -2.6157e-02,  3.4638e-02,  1.4179e-02,  1.9794e-02, -3.6213e-02,
                                        2.9275e-02, -3.1926e-02, -8.0071e-03,  2.2646e-02,  7.1183e-02,
                                        -1.5578e-03,  2.0376e-02,  1.5293e-02, -2.6965e-02,  9.6673e-04,
                                        2.1256e-02,  4.1210e-02, -3.7659e-02,  2.4178e-02, -7.2760e-02,
                                        4.2899e-02, -5.2250e-02, -5.8659e-02,  6.9418e-02,  7.2997e-02,
                                        8.8909e-04, -5.7929e-03, -1.9949e-02,  6.6790e-02, -6.0132e-02,
                                        -7.1881e-03,  1.3213e-02,  2.7020e-02, -1.8481e-02,  3.6954e-02,
                                        3.2679e-02, -8.5962e-03, -2.9961e-02, -5.5905e-03, -3.2449e-02,
                                        1.0689e-02,  1.0270e-02,  2.9335e-02,  2.5146e-02])

    def tag_user(self, prompt) -> list[float]:
        """generates cultural tags from user prompt used to update current profile."""
        embeddings = self.embedder.encode(prompt, convert_to_tensor=True)
        cultural_vector = self.tagger(embeddings.unsqueeze(0))
        return cultural_vector

    def update_user_profile(self, cultural_vector, alpha=0.05) -> list[float]:
        """updates the current user profile with the new cultural vector based on most recent response."""
        print(f'old cultural vector: {self.cultural_vector}')
        self.cultural_vector = [(1 - alpha) * old + alpha * new for old, new in zip(self.cultural_vector, cultural_vector)]
        print(f'new cultural vector: {self.cultural_vector}')
        return self.cultural_vector
    
    def tag_prompt(self, prompt) -> str:
        """tags the prompt with the current cultural vector."""
        tag_str = f"[pd={self.cultural_vector[0]:.2f}][indiv={self.cultural_vector[1]:.2f}][independent={self.cultural_vector[2]:.2f}][ua={self.cultural_vector[3]:.2f}]"
        return tag_str + " " + prompt
    
    def remove_before_newline(self, text):
        if '\n' in text:
            return text.split('\n', 1)[1].strip()
        else:
            return text.strip()
        
    def on_topic(self, prompt, threshold=0.20):
        prompt_embedding = self.embedder.encode(prompt, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(prompt_embedding, self.cn_tensor.to(device)).item()
        print(f'similarity: {similarity}')
        return similarity >= threshold
    
    def normalize_letter(self, response):
        if not response:
            return None
        first_letter = response.strip().lower()[0]
        if first_letter == 'f':
            return 'formal'
        elif first_letter == 'i':
            return 'informal'
        elif first_letter == 'e':
            return 'exploratory'
        elif first_letter == 'g':
            return 'guided'
        return None

    def generate_response(self, prompt, max_length=130) -> str:
        prompt = prompt.strip()
        punctuation = "!.?"
        if not prompt.strip().endswith(tuple(punctuation)):
            prompt += "?"
        prompt = prompt.replace(' it', ' the CN Tower')
        print(f'cleaned prompt: {prompt}')

        if not self.on_topic(prompt.split("]")[-1].strip()):
            if self.cultural_vector[0] < 50:
                return 'Sorry, I’ve only been trained talk about the CN Tower — let’s get back to that!'
            else:
                return 'Sorry, but I can only provide information about the CN Tower. Could you recenter your question?.'
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(device)
        print(f"Prompt: {prompt}")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                top_p=0.6,
                temperature=0.4,
                num_return_sequences=1,
                repetition_penalty=1.35,
                no_repeat_ngram_size=2,
                do_sample=True,
                eos_token_id=self.eor_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"raw response: {response}")

        # strip text after the <|eor|> token
        response = response.split("<|eor|>")[0]

        # Remove the prompt if still present at beginning
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        parts = re.split(r'([.!?])', response)

        # remove any sentence after 2-3 sentences
        choice = random.choice([2, 3])
        chunks = []
        for i in range(0, len(parts) - 1, 2):
            chunk = parts[i].strip() + parts[i + 1]  # sentence + punctuation
            chunks.append(chunk)
            if len(chunks) == choice:
                break

        response = " ".join(chunks)
        response = self.remove_before_newline(response)
        print(response)
        return response


    
        
