# model.py

import torch
import torch.nn as nn

# ------------------------------------------------------------
# Custom GRU Cell Implementation
# ------------------------------------------------------------

class GRUCell(nn.Module):
    """
    A custom implementation of a single GRU cell, mimicking PyTorch internals.
    """
    def __init__(self, input_dim, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        self.fc_x = nn.Linear(input_dim, 3 * hidden_units)
        self.fc_h = nn.Linear(hidden_units, 3 * hidden_units)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights using Xavier for multi-dim params and zeros for biases.
        """
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.xavier_uniform_(weight)
            else:
                nn.init.zeros_(weight)

    def forward(self, x, hidden_state):
        """
        Forward pass for a single time step.
        """
        x_proj = self.fc_x(x)
        h_proj = self.fc_h(hidden_state)
        r_x, z_x, n_x = x_proj.chunk(3, dim=1)
        r_h, z_h, n_h = h_proj.chunk(3, dim=1)

        r = torch.sigmoid(r_x + r_h)
        z = torch.sigmoid(z_x + z_h)
        n = torch.tanh(n_x + r * n_h)

        h_t = (1 - z) * n + z * hidden_state
        return h_t

# ------------------------------------------------------------
# Custom GRU Model (Sequence Classifier)
# ------------------------------------------------------------

class GRUModel(nn.Module):
    """
    Custom GRU-based encoder using the GRUCell, handling variable lengths.
    """
    def __init__(self, input_size, hidden_size, output_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_cell = GRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x, lengths):
        """
        x: (batch_size, seq_len, embed_dim)
        lengths: actual lengths of sequences before padding
        """
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            # Masking: ignore padding tokens
            mask = (t < lengths).float().unsqueeze(1)  # (batch_size, 1)
            h_t_new = self.gru_cell(x[:, t, :], h_t)
            h_t = h_t_new * mask + h_t * (1 - mask)

        return self.fc(h_t)

# ------------------------------------------------------------
# Full Sentiment Classifier using Custom GRU
# ------------------------------------------------------------

class GRU_Sentiment_Analysis(nn.Module):
    """
    Sentiment classifier using custom GRU model.
    """
    def __init__(self, vocab_size, embed_dim, gru_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = GRUModel(input_size=embed_dim, hidden_size=gru_hidden_size, output_dim=fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = self.gru(out, lengths)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# ------------------------------------------------------------
# Baseline: Bidirectional GRU using PyTorch
# ------------------------------------------------------------

class BidirectionalGRU(nn.Module):
    """
    Baseline model using PyTorch's bidirectional GRU.
    """
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, rnn_hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(rnn_hidden_size * 2, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(out)
        # hidden shape: (2, batch, hidden_size)
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# ------------------------------------------------------------
# Left-to-Right (1-directional) GRU using PyTorch
# ------------------------------------------------------------

class GRULeftToRight(nn.Module):
    """
    Left-to-right (unidirectional) GRU model using PyTorch.
    """
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(rnn_hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(out)
        out = self.fc1(hidden[-1])
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
