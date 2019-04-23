import torch
import torch.nn as nn
import torch.nn.functional as F


# RNN - LSTM
class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, name, output_size=1, num_layers=1, dtype=torch.float,
                 device='cpu', batch_first=False, dropout=0.):
        super(Lstm, self).__init__()
        self.input_size = input_size
        self. hidden_size = hidden_size
        self.batch_first = batch_first
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.name = name
        self.batch_size = batch_size

        self.dtype = dtype
        self.device = device
        self.hidden = self.init_hidden(self.batch_size, self.hidden_size)

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def init_hidden(self, batch_size, hidden_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers, batch_size, hidden_size),
                torch.zeros(self.num_layers, batch_size, hidden_size))

    def forward(self, cell_input):
        lstm_out, self.hidden = self.rnn(cell_input, self.hidden)
        score = self.out(lstm_out)

        return score

    @classmethod
    # summarize scores
    def summarize_scores(cls, name, score, scores):
        s_scores = ', '.join(['%.1f' % s for s in scores])
        print('%s: [%.3f] %s' % (name, score, s_scores))
