import torch
import torch.nn as nn
import numpy as np
import time
import train
import preprocessing


# RNN - LSTM
class Lstm(nn.Module):
    def __init__(self, d_in, hidden_dim, d_out, folder, data_source):
        super(Lstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.d_in = d_in
        self.d_out = d_out

        self.lstm = nn.LSTM(self.d_in, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.d_out)
        self.data = preprocessing.Preprocessing.load_df(folder, data_source)
        self.data['last_updated'] = self.data['last_updated'].astype(np.datetime64)
        self.train_set, self.val_set, self.test_set = train.train_val_test_split(self.data, 0.1, 0.1)

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

    #@classmethod
    ## evaluate one or more daily forecasts against expected values
    #def evaluate_forecasts(cls, actual, predicted):
    #    scores = list()
    #    # calculate an RMSE score for each day
    #    for i in range(actual.shape[1]):
    #        # calculate mseabs
    #        mse = #mean_squared_error(actual[:, i], predicted[:, i])
    #        # calculate rmse
    #        rmse = sqrt(mse)
    #        # store
    #        scores.append(rmse)
    #    # calculate overall RMSE
    #    s = 0
    #    for row in range(actual.shape[0]):
    #        for col in range(actual.shape[1]):
    #            s += (actual[row, col] - predicted[row, col]) ** 2
    #    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    #    return score, scores

    @classmethod
    # convert history into inputs and outputs
    def to_supervised(cls, train, n_input, n_out):
        # flatten data
        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end < len(data):
                x_input = data[in_start:in_end, 1:]
                # x_input = x_input.reshape((len(x_input), x_input.shape[1]))
                X.append(x_input)
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1

        return np.array(X), np.array(y)

    @classmethod
    # summarize scores
    def summarize_scores(cls, name, score, scores):
        s_scores = ', '.join(['%.1f' % s for s in scores])
        print('%s: [%.3f] %s' % (name, score, s_scores))

    #def train_model(self, n_epochs, model, batch_size,):
    #    # Train
    #    epochs = range(n_epochs)
    #    idx = 0
#
    #    for t in epochs:
    #        start = time.time()
    #        # or batch in range(0, int(N/batch_size)+1):
    #        #   if(batch<int(N/batch_size)):
    #        #   # Step 1. Calculate Batch
    #        #       batch_x = X[batch * batch_size : (batch + 1) * batch_size, :,:]
    #        #       # convert to: sequence x batch_size x n_features
    #        #       #batch_x = batch_x.reshape(batch_size, samples_per_day, features)#.transpose(0,1)
    #        #       batch_y = y[batch * batch_size : (batch + 1) * batch_size]
    #        #
    #        #       #print(X.shape, batch_x.shape, batch_y.shape)
    #        #   else:
    #        #       batch_x = X[(batch - 1) * batch_size +(N % batch_size): batch * batch_size + (N % batch_size), :]
    #        #       # convert to: sequence x batch_size x n_features
    #        #       #batch_x = batch_x.reshape(batch_size, samples_per_day, features).transpose(0,1)
    #        #       batch_y = y[(batch - 1) * batch_size + (N % batch_size): (batch + 1) * batch_size + (N % batch_size)]    # Step 2. Remember that Pytorch accumulates gradients.
#
    #        # We need to clear them out before each instance
    #        model.zero_grad()
#
    #        # Also, we need to clear out the hidden state of the LSTM,
    #        # detaching it from its history on the last instance.
    #        model.hidden = model.init_hidden(batch_size)
#
    #        # Step 3. Run our forward pass.
    #        output = model(X)
    #        # outputs.append(output)
    #        # Step 4. Berechne den Fehler mit dem letzten output
    #        loss = criterion(output[-1, :, -1], batch_y[-1, :])
    #        # print(output.shape, batch_y.shape)
#
    #        optimizer.zero_grad()
    #        loss.backward()
    #        optimizer.step()
#
    #        # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
    #        if t % 1 == 0:
    #            loss_hist.append(loss.item())
    #            print(t, loss.item(), time.time() - start)
