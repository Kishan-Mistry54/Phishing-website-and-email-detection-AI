import pandas as pd

# Load your files
email_df = pd.read_csv("phishing_emails_sample.csv")
url_df = pd.read_csv("phishing_urls_sample (1).csv")

# Merge datasets
merged_df = pd.merge(email_df, url_df, left_on="email_id", right_on="url_id")

# Create combined input text
merged_df["combined_text"] = (
    "[EMAIL] " + merged_df["subject"] + " FROM " + merged_df["sender"] +
    " [URL] " + merged_df["url"]
)

# Select relevant columns and rename
final_df = merged_df[["combined_text", "is_phishing_x"]].rename(
    columns={"is_phishing_x": "label"}
)

# Save to new CSV (optional)
final_df.to_csv("combined_phishing_dataset.csv", index=False)

# Preview
print(final_df.head())

from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch

# Load your final dataframe
import pandas as pd
df = pd.read_csv("combined_phishing_dataset.csv")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize
tokens = tokenizer(
    list(df["combined_text"]),
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# Labels
labels = torch.tensor(df["label"].values)

# Train-test split
from torch.utils.data import TensorDataset, random_split

dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], labels)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
