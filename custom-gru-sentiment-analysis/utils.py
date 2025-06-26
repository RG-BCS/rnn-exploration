# utils.py

import torch
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ------------------------------------------------------------
# Confusion Matrix Plotting
# ------------------------------------------------------------

def plot_confusion_matrix(model, dataloader, device, labels=['Negative', 'Positive'], 
                          normalize=False, title='Confusion Matrix'):
    """
    Plots a confusion matrix from model predictions on a dataloader.
    """
    model.eval()
    y_pred, y_true = [], []

    with torch.no_grad():
        for x, y, lengths in dataloader:
            x = x.to(device)
            y = y.to(device).float()
            lengths = lengths.to(device)

            preds = model(x, lengths).squeeze(-1)
            preds = (preds >= 0.5).float()

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='.2f' if normalize else 'd')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Gradient Norm Tracker
# ------------------------------------------------------------

def grad_norm(model):
    """
    Computes total gradient L2 norm (useful for monitoring training stability).
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# ------------------------------------------------------------
# Training & Evaluation Functions
# ------------------------------------------------------------

def train(model, loss_fn, optimizer, dataloader, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_acc, total_loss = 0.0, 0.0

    for text_batch, label_batch, lengths in dataloader:
        text_batch = text_batch.to(device)
        label_batch = label_batch.to(device)
        lengths = lengths.to(device)

        pred = model(text_batch, lengths).squeeze(1)
        loss = loss_fn(pred, label_batch)

        loss.backward()
        norm_grad = grad_norm(model)
        optimizer.step()
        optimizer.zero_grad()

        total_acc += ((pred >= 0.5).float() == label_batch).sum().item()
        total_loss += loss.item() * label_batch.size(0)

    dataset_size = len(dataloader.dataset)
    return total_acc / dataset_size, total_loss / dataset_size, norm_grad

def evaluate(model, loss_fn, dataloader, device):
    """
    Evaluates the model on a validation/test set.
    """
    model.eval()
    total_acc, total_loss = 0.0, 0.0

    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            text_batch = text_batch.to(device)
            label_batch = label_batch.to(device)
            lengths = lengths.to(device)

            pred = model(text_batch, lengths).squeeze(1)
            loss = loss_fn(pred, label_batch)

            total_acc += ((pred >= 0.5).float() == label_batch).sum().item()
            total_loss += loss.item() * label_batch.size(0)

    dataset_size = len(dataloader.dataset)
    return total_acc / dataset_size, total_loss / dataset_size

# ------------------------------------------------------------
# Epoch Loop Trainer
# ------------------------------------------------------------

def train_each_models(model, loss_fn, optimizer, train_dl, valid_dl, device, NUM_EPOCHS=10):
    """
    Trains and evaluates the model for multiple epochs.
    """
    train_accs, train_losses = [], []
    valid_accs, valid_losses = [], []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        acc_train, loss_train, grad = train(model, loss_fn, optimizer, train_dl, device)
        acc_valid, loss_valid = evaluate(model, loss_fn, valid_dl, device)

        train_accs.append(acc_train)
        train_losses.append(loss_train)
        valid_accs.append(acc_valid)
        valid_losses.append(loss_valid)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {elapsed:.2f}s | "
              f"Train Acc: {acc_train:.4f} | Train Loss: {loss_train:.4f} | "
              f"Valid Acc: {acc_valid:.4f} | Valid Loss: {loss_valid:.4f} | Grad Norm: {grad:.2f}")

    return train_accs, train_losses, valid_accs, valid_losses
