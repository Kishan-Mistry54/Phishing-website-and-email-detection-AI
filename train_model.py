import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load the dataset
df = pd.read_csv("combined_phishing_dataset.csv")

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer(
    list(df["combined_text"]),
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# Labels
labels = torch.tensor(df["label"].values)

# Dataset
dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], labels)

# Train/val split
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training
model.train()
for epoch in range(3):  # you can increase epochs
    print(f"Epoch {epoch + 1}")
    for batch in tqdm(train_loader):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())

# Report
print(classification_report(true_labels, predictions))

# Save the model
model.save_pretrained("saved_phishing_model")
tokenizer.save_pretrained("saved_phishing_model")
