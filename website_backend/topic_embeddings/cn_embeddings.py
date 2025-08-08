"""
CN Tower Embeddings

Using BERT-style embeddings to make a general embedding of the CN Tower questions. 
Will be used to keep conversation on track.
To run this file correctly, move the final_combined_dataset.json file to this folder.
"""
import json
from sentence_transformers import SentenceTransformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

with open("website_backend/topic_embeddings/final_combined_dataset.json", "r") as f:
    data = json.load(f)

cn_questions = [item["input_text"].split("]")[-1].strip() for item in data]

cn_embeddings = model.encode(cn_questions, convert_to_tensor=True)
cn_tower_embedding = cn_embeddings.mean(dim=0)

print(cn_tower_embedding) 