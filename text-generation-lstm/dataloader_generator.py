import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, index):
        chunk = self.text_chunks[index]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

def create_dataloaders(text_encoded, sequence_length, batch_size, split_ratio=0.9):
    chunk_size = sequence_length + 1
    text_chunks = [text_encoded[i:chunk_size + i] for i in range(len(text_encoded) - chunk_size + 1)]
    dataset = TextDataset(text_chunks)
    train_size = int(split_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, drop_last=True)
    return train_dl, valid_dl
