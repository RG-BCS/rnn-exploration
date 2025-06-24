import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from lstm_cell import LSTMCell  # Import your custom LSTM cell

class LSTMModelTrig(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_dim=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        c_t = torch.zeros((batch_size, self.hidden_size), device=device)
        h_t = torch.zeros((batch_size, self.hidden_size), device=device)
        for t in range(x.size(1)):
            h_t, c_t = self.lstm_cell(x[:, t, :], (h_t, c_t))
        return self.fc(h_t)

def input_data(data, window_size):
    data = data.tolist()
    sliding_windows = []
    for i in range(len(data) - window_size):
        x_seq = torch.tensor(data[i:i+window_size])
        y_seq = torch.tensor(data[i+window_size:i+window_size+1])
        sliding_windows.append((x_seq, y_seq))
    return sliding_windows

def main():
    torch.manual_seed(20)

    window_size = 40
    num_epochs = 10
    learning_rate = 0.01

    # Generate sine wave data
    x = torch.linspace(0, 799, 800)
    y = torch.sin(x * torch.pi * 2 / 40)
    y_train, y_test = y[:-40], y[-40:]

    train_data = input_data(y_train, window_size)

    model = LSTMModelTrig(input_size=1, hidden_size=50)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = 0.0
        start_time = time.time()
        model.train()

        for x_batch, y_batch in train_data:
            x_batch = x_batch.view(1, window_size, 1)
            y_pred = model(x_batch)[0]
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        train_loss /= len(train_data)

        # Predict next values for visualization
        sample_input = y_train[-window_size:].tolist()
        model.eval()
        with torch.no_grad():
            for _ in range(window_size):
                input_tensor = torch.tensor(sample_input[-window_size:]).view(1, window_size, 1)
                pred = model(input_tensor).item()
                sample_input.append(pred)

        test_loss = loss_fn(torch.tensor(sample_input[-window_size:]), y_train[-window_size:])

        print(f'Epoch {epoch + 1}/{num_epochs} | {time.time() - start_time:.4f} sec | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')

        plt.figure(figsize=(12, 2))
        plt.plot(y.numpy(), color='#8000ff')
        plt.plot(range(760, 800), sample_input[-window_size:], color='#ff8000')
        plt.legend(['True Signal', 'Predicted'], loc='upper left')
        plt.title(f'Epoch: {epoch + 1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}')
        plt.xlabel('Time step')
        plt.ylabel('sin(x)')
        plt.show()

if __name__ == "__main__":
    main()
