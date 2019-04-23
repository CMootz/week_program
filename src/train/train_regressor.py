import torch
import os
import pandas as pd
import json
import time
import numpy as np

import models
import preprocessing as prep
import train


class TrainLstm:
    def __init__(self, model, x_train, x_val, y_train, y_val, exp_dir):
        self.model = model
        self.X_train = x_train
        self.X_val = x_val
        self.y_train = y_train
        self.y_val = y_val
        self.X = None
        self.y = None
        self.X_v = None
        self. y_v = None

        self.exp_dir = exp_dir
        self.data_is_prepared = False

        # define the name of the directory to be created
        if not os.path.isdir(self.exp_dir):
            os.makedirs(self.exp_dir)

    def prepare_data(self):
        self.X = torch.from_numpy(self.X_train).to(self.model.device).type(self.model.dtype)
        self.y = torch.from_numpy(self.y_train).to(self.model.device).type(self.model.dtype)
        self.X_v = torch.from_numpy(self.X_val).to(self.model.device).type(self.model.dtype)
        self.y_v = torch.from_numpy(self.y_val).to(self.model.device).type(self.model.dtype)

        self.data_is_prepared = True

    def run_train(self, n_epochs, lr=0.01, weight_decay=0.01):
        if not self.data_is_prepared:
            self.prepare_data()

        # Loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Train
        loss_hist = np.zeros(n_epochs)
        loss_validate_hist = np.zeros(n_epochs)
        start = time.time()
        for t in range(n_epochs):

            # Berechne den Batch
            #batch_x = self.X[:, batch:batch+1, :]
            #batch_y = self.y[batch:batch+1, :]
            # Berechne die Vorhersage (foward step)
            # We need to clear them out before each instance
            self.model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            self.model.hidden = self.model.init_hidden(self.model.batch_size, self.model.hidden_size)
            # Step 3. Run our forward pass.
            output = self.model(self.X)

            # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
            loss = criterion(output[:, -1, :], self.y)
            # Berechne die Gradienten und Aktualisiere die Gewichte (backward step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.model.hidden = self.model.init_hidden(self.model.batch_size, self.model.hidden_size)
            outputs_val = self.model(self.X_v)
            loss_val = criterion(outputs_val[:, -1, :], self.y_v)
            loss_hist[t] = loss.item()
            loss_validate_hist[t] = loss_val.item()

            # Berechne den Fehler (Ausgabe des Fehlers alle 50 Iterationen)
            if t % 10 == 0:
                print('Epoch ', t, ' train_loss: ', loss.item(), 'validate_loss: ', loss_val.item(),
                      'Time ', time.time() - start)
                start = time.time()

        torch.save(self.model, self.exp_dir + self.model.name + '.pt')
        pd.DataFrame(loss_hist).to_csv(self.exp_dir + '/errors__' + self.model.name + '__.csv', index=False)
        pd.DataFrame(loss_validate_hist).to_csv(self.exp_dir + '/errors_val__' + self.model.name + '__.csv',
                                                index=False)

        meta_data = {'n_epochs': n_epochs, 'lr': lr, 'batch_size': self.X.shape[0]}
        with open(self.exp_dir + '/data.json', 'w') as fp:
            json.dump(meta_data, fp)

        return

