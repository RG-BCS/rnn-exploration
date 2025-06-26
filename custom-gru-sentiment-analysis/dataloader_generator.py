# dataloader_generator.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
from collections import Counter, OrderedDict

# ------------------------------------------------------------
# ⚙️ Config
# ------------------------------------------------------------

SEED = 11
BATCH_SIZE = 32

np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------
# 🧼 Tokenizer + Vocabulary Utilities
# ------------------------------------------------------------

def tokenizer(text):
    """
    Tokenizes the text by:
    - Lowercasing
    - Removing HTML tags and special characters
    - Extracting emoticons
    - Splitting into words
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text.split()

def build_vocab(texts, min_freq=1):
    """
    Builds vocabulary from list of texts.
    """
    token_counts = Counter()
    for text in texts:
        token_counts.update(tokenizer(text))
    
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_tokens.insert(0, ("<pad>", 0))
    sorted_tokens.insert(1, ("<unk>", 1))

    vocab = dict(zip([token for token, _ in sorted_tokens], range(len(sorted_tokens))))
    return vocab, token_counts

def label_to_int(label):
    return 1. if label == 'positive' else 0.

def word_to_int(text, label, vocab):
    """
    Converts tokens to integer indices based on vocab, handling unknowns.
    """
    tokenized = tokenizer(text)
    indices = [vocab.get(token, vocab["<unk>"]) for token in tokenized]
    return indices, label_to_int(label)

# ------------------------------------------------------------
# 🧺 Dataset and Collation
# ------------------------------------------------------------

class MyDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_batch(batch, vocab):
    """
    Prepares padded sequences, lengths, and label tensors for a batch.
    """
    label_list, text_list, lengths = [], [], []
    for text, label in batch:
        indices, label_val = word_to_int(text, label, vocab)
        label_list.append(label_val)
        text_tensor = torch.tensor(indices, dtype=torch.int64)
        text_list.append(text_tensor)
        lengths.append(len(indices))

    padded_texts = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return padded_texts, torch.tensor(label_list), torch.tensor(lengths)

# ------------------------------------------------------------
# DataLoader Generator
# ------------------------------------------------------------

def get_dataloaders(data_path="hf://datasets/scikit-learn/imdb/IMDB Dataset.csv", batch_size=BATCH_SIZE):
    """
    Loads the dataset, splits it, builds vocabulary, and returns train/valid dataloaders.
    """
    df = pd.read_csv(data_path)
    
    indices = np.arange(len(df))
    train_indices = np.random.choice(indices, 25000, replace=False)
    test_indices = [idx for idx in indices if idx not in train_indices]

    train_df = df.iloc[train_indices]
    train_df, valid_df = train_df[:20000], train_df[20000:]

    train_texts = train_df['review'].values
    train_labels = train_df['sentiment'].values
    valid_texts = valid_df['review'].values
    valid_labels = valid_df['sentiment'].values

    vocab, token_counts = build_vocab(train_texts)

    train_ds = MyDataset(train_texts, train_labels)
    valid_ds = MyDataset(valid_texts, valid_labels)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, vocab))
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, vocab))

    return train_dl, valid_dl, vocab
