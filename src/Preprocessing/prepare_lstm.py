import numpy as np
from .preprocessing import Preprocessing
import train


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out):
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
            x_input = data[in_start:in_end, :]
            # x_input = x_input.reshape((len(x_input), x_input.shape[1]))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += n_out

    return np.array(X), np.array(y)


def load_lstm_data(folder, filename, batch_size, n_output, split_set=True):
    df = Preprocessing.load_df(folder, filename)
    df['last_updated'] = df['last_updated'].astype(np.datetime64)
    if split_set:
        train_set, val_set, test_set = train.train_val_test_split(df, batch_size, 0.1, 0.1)
        x_train, y_train = to_supervised(train_set, batch_size, n_output)
        x_val, y_val = to_supervised(train_set, batch_size, n_output)
        x_test, y_test = to_supervised(train_set, batch_size, n_output)
        return x_train, x_val, x_test, y_train, y_val, y_test
    else:
        data_set = train.prepare_single_set(df, batch_size)
        x, y = to_supervised(data_set, batch_size, n_output)
        return x, y
