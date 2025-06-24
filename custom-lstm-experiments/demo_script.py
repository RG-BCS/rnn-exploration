import torch
import matplotlib.pyplot as plt
from binary_counting import LSTMModel, train_model
from sine_wave_prediction import LSTMModelTrig, input_data
from lstm_cell import LSTMCell
from torch.utils.data import DataLoader, TensorDataset
import copy

def run_binary_counting_demo():
    print("Running Binary Counting Demo...")

    # Parameters
    EPOCHS = 5
    TRAINING_SAMPLES = 1000
    BATCH_SIZE = 16
    TEST_SAMPLES = 200
    SEQUENCE_LENGTH = 20
    HIDDEN_UNITS = 20

    # Create binary sum data
    X_train = torch.randint(0, 2, (TRAINING_SAMPLES, SEQUENCE_LENGTH, 1)).float()
    X_test = torch.randint(0, 2, (TEST_SAMPLES, SEQUENCE_LENGTH, 1)).float()
    y_train = torch.sum(X_train, dim=1)
    y_test = torch.sum(X_test, dim=1)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMModel(input_size=1, hidden_size=HIDDEN_UNITS, output_dim=1)
    model_untrained = copy.deepcopy(model)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_loss, train_acc, valid_loss, valid_acc = train_model(
        model, loss_fn, optimizer, train_dl, test_dl, tolerance=1
    )

    # Plot results before and after training
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    y_pred_before = model_untrained(X_test).detach().numpy().round()
    plt.scatter(y_test.numpy(), y_pred_before, alpha=0.3)
    plt.title("Before Training")
    plt.xlabel("True Sum")
    plt.ylabel("Predicted Sum")

    plt.subplot(1, 2, 2)
    y_pred_after = model(X_test).detach().numpy().round()
    plt.scatter(y_test.numpy(), y_pred_after, alpha=0.3)
    plt.title("After Training")
    plt.xlabel("True Sum")
    plt.ylabel("Predicted Sum")

    plt.suptitle("Binary Counting Model")
    plt.show()

def run_sine_wave_demo():
    print("Running Sine Wave Prediction Demo...")

    import time

    torch.manual_seed(20)
    window_size = 40
    num_epochs = 3
    learning_rate = 0.01

    x = torch.linspace(0, 799, 800)
    y = torch.sin(x * torch.pi * 2 / 40)
    y_train, y_test = y[:-40], y[-40:]

    train_data = input_data(y_train, window_size)

    model = LSTMModelTrig(input_size=1, hidden_size=50)
    loss_fn = torch.nn.MSELoss()
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

        sample_input = y_train[-window_size:].tolist()
        model.eval()
        with torch.no_grad():
            for _ in range(window_size):
                input_tensor = torch.tensor(sample_input[-window_size:]).view(1, window_size, 1)
                pred = model(input_tensor).item()
                sample_input.append(pred)

        test_loss = loss_fn(torch.tensor(sample_input[-window_size:]), y_train[-window_size:])
        print(f"Epoch {epoch+1}/{num_epochs} | {time.time() - start_time:.4f} sec | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(y.numpy(), color="#8000ff", label="True Signal")
    plt.plot(range(760, 800), sample_input[-window_size:], color="#ff8000", label="Predicted")
    plt.legend(loc="upper left")
    plt.title("Sine Wave Prediction Model")
    plt.xlabel("Time Step")
    plt.ylabel("sin(x)")
    plt.show()

def main():
    run_binary_counting_demo()
    run_sine_wave_demo()

if __name__ == "__main__":
    main()
