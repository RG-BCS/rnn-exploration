import torch
import time
from torch.distributions import Categorical

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_model(model, loss_fn, optimizer, train_dl, valid_dl, num_epochs, sequence_length, device):
    train_losses = []

    def evaluate(x_batch, y_batch, train=True):
        model.train() if train else model.eval()
        batch_size = x_batch.size(0)
        hidden, cell = model.init_hidden(batch_size)
        hidden, cell = hidden.to(device), cell.to(device)
        loss = 0.0
        if train:
            optimizer.zero_grad()
        for c in range(sequence_length):
            input_ = x_batch[:, c].to(device)
            pred, hidden, cell = model(input_, hidden, cell)
            hidden = hidden.detach()
            cell = cell.detach()
            y_batch_step = y_batch[:, c].to(device)
            loss += loss_fn(pred, y_batch_step)
        if train:
            loss.backward()
            optimizer.step()
        return loss.item() / sequence_length

    for epoch in range(num_epochs):
        start = time.time()
        train_epoch_loss = 0.0
        for x_batch, y_batch in train_dl:
            train_epoch_loss += evaluate(x_batch, y_batch, train=True)
        train_losses.append(train_epoch_loss)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            valid_epoch_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in valid_dl:
                    valid_epoch_loss += evaluate(x_batch, y_batch, train=False)

            train_epoch_loss /= len(train_dl)
            valid_epoch_loss /= len(valid_dl)
            elapsed = time.time() - start
            norm_grad = grad_norm(model)
            print(f'Epoch {epoch}/{num_epochs} | time_elapsed: {elapsed:.3f}s | train_loss: {train_epoch_loss:.4f} | '
                  f'valid_loss: {valid_epoch_loss:.4f} | grad_norm: {norm_grad:.4f}')
    return train_losses

def generate_text(model, text, char_length, char2int, char_array, device):
    tokens = torch.tensor([char2int[c] for c in text]).to(device)
    model.eval()
    hidden, cell = model.init_hidden(1)
    hidden, cell = hidden.to(device), cell.to(device)
    for i in range(len(tokens)):
        input_ = tokens[i:i+1]
        out, hidden, cell = model(input_, hidden, cell)
    for _ in range(char_length):
        y_pred = torch.argmax(out, dim=1).item()
        text += char_array[y_pred]
        input_ = torch.tensor([y_pred]).to(device)
        out, hidden, cell = model(input_, hidden, cell)
    return text

def random_next_text(model, text, char_length, scale_factor, char2int, char_array, device):
    model.eval()
    for _ in range(char_length):
        tokens = torch.tensor([char2int[c] for c in text]).to(device)
        hidden, cell = model.init_hidden(1)
        hidden, cell = hidden.to(device), cell.to(device)
        for i in range(len(tokens)):
            input_ = tokens[i:i+1]
            out, hidden, cell = model(input_, hidden, cell)
        m = Categorical(logits=out * scale_factor)
        y_pred = m.sample((1,)).item()
        text += char_array[y_pred]
    return text
