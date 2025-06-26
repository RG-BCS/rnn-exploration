# demo_script.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataloader_generator import get_dataloaders
from models import GRU_Sentiment_Analysis, GRULeftToRight, BidirectionalGRU
from utils import train_each_models, plot_confusion_matrix

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

NUM_EPOCHS = 10
EMBED_DIM = 20
RNN_HIDDEN = 64
FC_HIDDEN = 64
DATA_PATH = "hf://datasets/scikit-learn/imdb/IMDB Dataset.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
#  Train + Plot Function
# ------------------------------------------------------------

def run_model(model_class, seed, model_name, vocab_size, train_dl, valid_dl,lr=0.001):
    print(f"\n Training {model_name}")
    torch.manual_seed(seed)
    model = model_class(vocab_size, EMBED_DIM, RNN_HIDDEN, FC_HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    train_acc, train_loss, valid_acc, valid_loss = train_each_models(
        model, loss_fn, optimizer, train_dl, valid_dl, device=DEVICE, NUM_EPOCHS=NUM_EPOCHS
    )

    # Plot Accuracy & Loss
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Acc', color='blue')
    plt.plot(valid_acc, label='Valid Acc', color='red')
    plt.title(f"{model_name} Accuracy")
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(valid_loss, label='Valid Loss', color='red')
    plt.title(f"{model_name} Loss")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot confusion matrix
    plot_confusion_matrix(model, valid_dl, DEVICE, title=f"{model_name} Confusion Matrix")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("[INFO] Loading data...")
    train_dl, valid_dl, vocab = get_dataloaders(DATA_PATH)
    vocab_size = len(vocab)

    run_model(GRU_Sentiment_Analysis, seed=1, model_name="Custom GRU", vocab_size=vocab_size,
              train_dl=train_dl, valid_dl=valid_dl)
    
    run_model(GRULeftToRight, seed=2, model_name="Uni-directional GRU", vocab_size=vocab_size,
              train_dl=train_dl, valid_dl=valid_dl)
    
    run_model(BidirectionalGRU, seed=3, model_name="Bi-directional GRU", vocab_size=vocab_size,
              train_dl=train_dl, valid_dl=valid_dl)

if __name__ == "__main__":
    main()
