"""
Training TagAdversary Class

Using final_combined_dataset.json, using the responses the model will attempt to predict tags.
"""
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tag_adversary import TagAdversary
from torch.utils.data import TensorDataset, DataLoader, random_split

print('checkpoint 1')

RAW_DATA_PATH = 'final_combined_dataset.json'

# setting seed for reproducibility
torch.manual_seed(17)
np.random.seed(17)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# loading data and converting to lists
raw_data = RAW_DATA_PATH
with open(raw_data, "r") as f:
    data = json.load(f)

texts = [item["target_output"] for item in data]
labels = [item["cultural_labels"] for item in data]

print('dataset opened')

# define deepseek model
model_name = "deepseek-ai/deepseek-llm-7b-base"
print('trying model')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
deepseek_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
deepseek_model.eval()

print('model instantiated')


# pooling function
def mean_pool(hidden_states, attention_mask):
    """pools the embeddings together so that it is squeezed to an array of shape [1, 4096]"""
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked = hidden_states * mask
    summed = masked.sum(1)
    count = mask.sum(1).clamp(min=1e-9)
    return summed / count


# converting texts deepseek embeddings used to feed into adversary
deepseek_embeddings = []

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = deepseek_model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]
        pooled = mean_pool(last_hidden, inputs["attention_mask"])
        deepseek_embeddings.append(pooled.squeeze(0).cpu())
    del inputs, outputs, last_hidden, pooled
    torch.cuda.empty_cache()
    print('text embedded')

embeddings = torch.stack(deepseek_embeddings).to(device)
embeddings = embeddings.clone().detach().to(device)
labels_tensor = torch.tensor(labels, dtype=torch.float).to(device)
dataset = TensorDataset(embeddings, labels_tensor)
print('embeddings -> dataset')

# training and evaluation split (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders to split into batch sizes and have them shuffled to avoid overfitting
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# initializing tag_model and the loss_fn
tag_model = TagAdversary().to(device)
optimizer = torch.optim.Adam(tag_model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()

# configuration of training
num_epochs = 100
best_val_loss = float("inf")
patience = 10
patience_counter = 0

print("Embeddings shape:", embeddings.shape)
print("Labels shape:", labels_tensor.shape)

for epoch in range(num_epochs):
    tag_model.train()
    train_loss = 0.0

    for batch in train_loader:
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        preds = tag_model(x_batch)
        loss = loss_fn(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss/len(train_loader)

    # validation
    tag_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            text_batch, label_batch = batch[0].to(device), batch[1].to(device)
            preds = tag_model(text_batch)
            loss = loss_fn(preds, label_batch)
            val_loss += loss.item()

    avg_val_loss = val_loss/len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Eval Loss = {avg_val_loss:.4f}")

    # early stops
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0

        # saves checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': tag_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, f"best_checkpoint.pt")

        print(f"Saved checkpoint at epoch {epoch+1} (val loss improved)")
    # patience timer, stops if no improvement
    else:
        patience_counter += 1
        print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stop.")
            break
