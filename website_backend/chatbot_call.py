from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from sentence_transformers import SentenceTransformer
from tagger_backend.tagger import Tagger
import os
import re
from dotenv import load_dotenv

load_dotenv()

hf_model_id = os.getenv("HF_MODEL_ID")
weights_path = "./tagger_backend/tagger_weights_v1.pt"
canadian_cultural_vector = [0.39, 0.80, 0.80, 0.48] # default cultural vector, see 6dim_hofsted.csv for more details 
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
        self.cultural_vector = canadian_cultural_vector

    def tag_user(self, prompt) -> list[float]:
        """generates cultural tags from user prompt used to update current profile."""
        embeddings = self.embedder.encode(prompt, convert_to_tensor=True)
        cultural_vector = self.tagger(embeddings.unsqueeze(0))
        return cultural_vector

    def update_user_profile(self, cultural_vector, alpha=0.1) -> list[float]:
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

    def generate_response(self, prompt: str, max_length=130) -> str:
        prompt = prompt.strip()
        punctuation = "!.?"
        if not prompt.strip().endswith(tuple(punctuation)):
            prompt += "?"
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(device)
        print(f"Prompt: {prompt}")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                top_p=0.6,
                temperature=0.5,
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

        # remove any sentence after three sentences
        chunks = []
        for i in range(0, len(parts) - 1, 2):
            chunk = parts[i].strip() + parts[i + 1]  # sentence + punctuation
            chunks.append(chunk)
            if len(chunks) == 3:
                break

        response = " ".join(chunks)
        response = self.remove_before_newline(response)
        print(response)
        return response


    
        
