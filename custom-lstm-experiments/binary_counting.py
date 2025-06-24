import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd

from lstm_cell import LSTMCell  # Import your custom LSTMCell

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)

        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=device)

        for t in range(seq_len):
            h_t, c_t = self.lstm_cell(x[:, t, :], (h_t, c_t))

        return self.fc(h_t)

def train_model(model, loss_fn, optimizer, train_dl, test_dl, epochs=20, tolerance=0.5):
    train_loss, train_accuracy = [], []
    valid_loss, valid_accuracy = [], []

    for epoch in range(epochs):
        model.train()
        current_loss = 0
        current_accuracy = 0
        for x_batch, y_batch in train_dl:
            y_batch = y_batch.view(-1, 1).float()
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            current_loss += loss.item() * x_batch.size(0)

            y_pred_int = torch.round(y_pred).long().view(-1)
            y_true_int = y_batch.long().view(-1)
            current_accuracy += (y_pred_int == y_true_int).sum().item()

        current_loss /= len(train_dl.dataset)
        current_accuracy /= len(train_dl.dataset)
        train_loss.append(current_loss)
        train_accuracy.append(current_accuracy)

        model.eval()
        current_loss = 0
        current_accuracy = 0
        with torch.no_grad():
            for x_batch, y_batch in test_dl:
                y_batch = y_batch.view(-1, 1).float()
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                current_loss += loss.item() * x_batch.size(0)

                y_pred_int = torch.round(y_pred).long().view(-1)
                y_true_int = y_batch.long().view(-1)
                current_accuracy += (y_pred_int == y_true_int).sum().item()

        current_loss /= len(test_dl.dataset)
        current_accuracy /= len(test_dl.dataset)
        valid_loss.append(current_loss)
        valid_accuracy.append(current_accuracy)

        print(f'Epoch {epoch+1}/{epochs} | '
              f'train_loss: {train_loss[-1]:.4f} | train_acc: {train_accuracy[-1]:.4f} | '
              f'valid_loss: {valid_loss[-1]:.4f} | valid_acc: {valid_accuracy[-1]:.4f}')

    return train_loss, train_accuracy, valid_loss, valid_accuracy

def main():
    torch.manual_seed(2)

    EPOCHS = 20
    TRAINING_SAMPLES = 10000
    TEST_SAMPLES = 1000
    SEQUENCE_LENGTH = 20
    HIDDEN_UNITS = 20
    BATCH_SIZE = 16

    # Generate dataset: sequences of 0/1, target is sum of 1's per sequence
    X_train = torch.randint(0, 2, (TRAINING_SAMPLES, SEQUENCE_LENGTH, 1)).float()
    X_test = torch.randint(0, 2, (TEST_SAMPLES, SEQUENCE_LENGTH, 1)).float()

    y_train = torch.sum(X_train, dim=1)
    y_test = torch.sum(X_test, dim=1)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = LSTMModel(input_size=1, hidden_size=HIDDEN_UNITS, output_dim=1)
    model_untrained = copy.deepcopy(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_loss, train_acc, valid_loss, valid_acc = train_model(
        model, loss_fn, optimizer, train_dl, test_dl, epochs=EPOCHS, tolerance=1
    )

    # Plotting
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.figure(figsize=(12,4))

    # Before training predictions
    plt.subplot(1, 3, 1)
    y_pred = model_untrained(X_test).detach().numpy().round()
    y_true = y_test.numpy()
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.title("Before Training")
    plt.xlabel("True Sum")
    plt.ylabel("Predicted Sum")

    # Training history
    plt.subplot(1, 3, 2)
    pd.DataFrame({
        'train_loss': train_loss,
        'train_acc': train_acc,
        'valid_loss': valid_loss,
        'valid_acc': valid_acc
    }).plot(ax=plt.gca())
    plt.grid(True)

    # After training predictions
    plt.subplot(1, 3, 3)
    y_pred = model(X_test).detach().numpy().round()
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.title("After Training")
    plt.xlabel("True Sum")
    plt.ylabel("Predicted Sum")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
