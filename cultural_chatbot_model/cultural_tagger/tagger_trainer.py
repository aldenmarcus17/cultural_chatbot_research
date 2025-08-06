"""Training file for the Tagger class."""
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tagger import Tagger
from torch.utils.data import TensorDataset, DataLoader, random_split

RAW_DATA_PATH = "../../dataset_generation/tagger_sets/combined_dataset_tagger.json"

# setting seed
torch.manual_seed(17)
np.random.seed(17)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}")

# loading data and making lists
with open(RAW_DATA_PATH, "r") as file:
    data = json.load(file)

texts = [item["response"] for item in data]
tags = [item["cultural_labels"] for item in data]

# define MiniLM model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(texts, convert_to_tensor=True)

# dataset
tags_tensor = torch.tensor(tags, dtype=torch.float).to(device)
dataset = TensorDataset(embeddings, tags_tensor)

# splitting dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# dataloaders to split
batch_size = 12
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# initializing tag model
tagger = Tagger().to(device)
optimizer = torch.optim.Adam(tagger.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# configuration of training
num_epochs = 100
best_val_loss = float("inf")
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    # training process
    tagger.train()
    train_loss = 0.0

    for batch in train_loader:
        text_batch, label_batch = batch[0].to(device), batch[1].to(device)

        preds = tagger(text_batch)
        loss = loss_fn(preds[0], label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss/len(train_loader)

    # evaluation
    tagger.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            text_batch, label_batch = batch[0].to(device), batch[1].to(device)
            preds = tagger(text_batch)
            loss = loss_fn(preds[0], label_batch)
            val_loss += loss.item()

    avg_val_loss = val_loss/len(val_loader)

    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Eval Loss = {avg_val_loss:.4f}")

    # early stops
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience = 0

        # saves checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': tagger.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, f"best_checkpoint_tagger.pt")

        print(f"Saved checkpoint at epoch {epoch + 1} (val loss improved)")
    else:
        patience_counter += 1

        if patience_counter >= patience:
            print("Early stop.")
            break


