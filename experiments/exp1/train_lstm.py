import torch

import models
import preprocessing as prep
import train

dtype = torch.float
device = torch.device("cpu")
rooms = ['schlafzimmer', 'kuche', 'wohnzimmer']

#
# load data
#
#M_data_2019_04_16 = prep.Preprocessing('M_data_2019_04_16','../../data/', 'home-assistant_v2.db')
#M_data_2019_04_16.load_data()
#M_data_2019_04_16.extract_weather_data()
#M_data_2019_04_16.extract_weather_data()


experiment_path = '../../experiments/exp1/'

X_train, X_val, _, y_train, y_val, _= prep.load_lstm_data('../../data/M_data_2019_04_16', 'train_climatewohnzimmer',
                                                          320, 320)

model = models.Lstm(X_train.shape[2], 64, 1, dtype, device)
training = train.TrainLstm(model, X_train, X_val, y_train, y_val, experiment_path)
training.run_train(10)
