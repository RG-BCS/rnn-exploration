import torch
import numpy as np
from model import RNNCharGenModel
from dataloader_generator import create_dataloaders
from utils import train_model, generate_text, random_next_text
from torch import nn
import matplotlib.pyplot as plt

# Assume you load or download and preprocess text here (you can move preprocessing to a separate file if needed)

# Set constants here (or pass via CLI args)
SEED = 10
SEQUENCE_LENGTH = 40
BATCH_SIZE = 64
NUM_EPOCHS = 50
EMBED_DIM = 256
RNN_HIDDEN_SIZE = 512
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your preprocessing code here (load text, encode, etc.)

# For demo, I assume text_encoded, char2int, char_array, train_dl, valid_dl are ready

model = RNNCharGenModel(len(char_array), EMBED_DIM, RNN_HIDDEN_SIZE).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loss = train_model(model, loss_fn, optimizer, train_dl, valid_dl, NUM_EPOCHS, SEQUENCE_LENGTH, DEVICE)

text_initiator = "The island"

print("Generated texts by the model shown below:\n")

print("1. Greedy (argmax) text generation:")
print(generate_text(model, text_initiator, 500, char2int, char_array, DEVICE))
print("#" * 80)

print("2. Random sampling with temperature = 1.0:")
print(random_next_text(model, text_initiator, 500, 1.0, char2int, char_array, DEVICE))
print("#" * 80)

print("3. Random sampling with temperature = 2.0:")
print(random_next_text(model, text_initiator, 500, 2.0, char2int, char_array, DEVICE))
print("#" * 80)

print("4. Random sampling with temperature = 0.5:")
print(random_next_text(model, text_initiator, 500, 0.5, char2int, char_array, DEVICE))
print("#" * 80)

plt.plot(train_loss)
plt.title("Training Loss")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
