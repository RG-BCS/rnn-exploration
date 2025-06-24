import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    """
    Custom LSTM Cell implementation from scratch.
    Compatible with batch inputs.
    """

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Linear layers for input and hidden state
        self.x_fc = nn.Linear(input_size, 4 * hidden_size)
        self.h_fc = nn.Linear(hidden_size, 4 * hidden_size)

        self.reset_parameters()

    def forward(self, x_t, hidden_cell_tuple):
        """
        Forward pass for one time step.

        Args:
            x_t: Input at time t, shape (batch, input_size)
            hidden_cell_tuple: Tuple of (h_{t-1}, c_{t-1}), each of shape (batch, hidden_size)

        Returns:
            h_t, c_t: Updated hidden and cell states
        """
        h_t_1, c_t_1 = hidden_cell_tuple

        # Flatten input if necessary
        x_t = x_t.view(-1, x_t.size(1))  # Ensure 2D

        gates = self.x_fc(x_t) + self.h_fc(h_t_1)
        i_t, f_t, candidate_c_t, o_t = gates.chunk(4, dim=1)

        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        o_t = torch.sigmoid(o_t)
        candidate_c_t = torch.tanh(candidate_c_t)

        c_t = f_t * c_t_1 + i_t * candidate_c_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

    def reset_parameters(self):
        """
        Initialize weights using Xavier uniform initialization for matrices
        and zeros for biases.
        """
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.xavier_uniform_(weight)
            else:
                nn.init.zeros_(weight)
