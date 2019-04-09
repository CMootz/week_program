import torch


# RNN - LSTM
class Lstm(torch.nn.Module):
    def __init__(self, embeded_dim, hidden_dim, d_out):
        super(Lstm, self).__init__()
        self.embeded_dim = embeded_dim
        self.hidden_dim = hidden_dim
        self.d_out = d_out

        self.lstm = torch.nn.LSTM(self.embeded_dim, self.hidden_dim)
        self.out = torch.nn.Linear(self.hidden_dim, self.d_out)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

    def forward(self, cell_input):
        lstm_out, self.hidden = self.lstm(cell_input, self.hidden)
        score = self.out(lstm_out)
        return score
