import numpy as np
import pandas as pd


def count_days(data_set):
    date_list = []
    for item in data_set.index:
        date = '{:4d}-{:02d}-{:02d}'.format(data_set.loc[item, 'year'],
                                      data_set.loc[item, 'month'],
                                      data_set.loc[item, 'day'])

        if date not in date_list:
            date_list.append(date)
    return date_list


def split_days(date_list, val_size=0.25, test_size=0.25):
    if int(len(date_list)*val_size) <= 0:
        length_val = 1
    else:
        length_val = int(len(date_list)*val_size)
    if int((len(date_list) - length_val) * test_size) <= 0:
        length_test = 1
    else:
        length_test = int((len(date_list) - length_val) * test_size)
    dates_test = date_list[-length_test-1:-1]
    dates_val = date_list[-length_test-length_val-1:-length_test-1]
    dates_train = date_list[:-length_test-length_val-1]
    print('train: ', dates_train, 'val :', dates_val, 'test: ',dates_test)
    return dates_train, dates_val, dates_test


def extract_data_by_date(data_set, date_list, number_samples):
    df = data_set
    new_df = pd.DataFrame()
    for item in date_list:
        start_date = np.datetime64(item+' 00:00:00')
        end_date = np.datetime64(item+' 23:59:59')
        #if len(df[(df['last_updated'] > start_date) & (df['last_updated'] < end_date)]) >= number_samples:
        if new_df.empty:
            new_df = new_df.append(df[(df['last_updated'] > start_date) & (df['last_updated'] < end_date)])
        else:
            new_df = pd.concat([new_df, df[(df['last_updated'] > start_date) & (df['last_updated'] < end_date)]])
        new_df = new_df.reset_index(drop=True)
    return new_df


def train_val_test_split(data_set, number_samples, val_size=0.25, test_size=0.25, drop_col=['last_updated', 'day',
                                                                                            'month', 'year',
                                                                                            'symbol']):
    start_date = np.datetime64('2019-04-07 00:00:00')
    data_set = data_set[(data_set['last_updated'] < start_date)]
    days = count_days(data_set)

    train_days, val_days, test_days = split_days(days, val_size, test_size)
    train_set = extract_data_by_date(data_set, train_days, number_samples)
    val_set = extract_data_by_date(data_set, val_days, number_samples)
    test_set = extract_data_by_date(data_set, test_days, number_samples)
    print(train_set.shape, val_set.shape, test_set.shape)
    train_set = train_set.drop(drop_col, axis=1)
    val_set = val_set.drop(drop_col, axis=1)
    test_set = test_set.drop(drop_col, axis=1)

    train_set_last = train_set[len(train_set)//number_samples*number_samples:].to_numpy()
    train_set = train_set[:len(train_set)//number_samples*number_samples]
    val_set_last = val_set[len(val_set)//number_samples*number_samples:].to_numpy()
    val_set = val_set[:len(val_set)//number_samples*number_samples]
    test_set_last = test_set[len(test_set)//number_samples*number_samples:].to_numpy()
    test_set = test_set[:len(test_set)//number_samples*number_samples]

    print(len(train_set), len(val_set), len(test_set))

    train_set = np.stack(np.split(train_set, len(train_set)/number_samples))
    val_set = np.stack(np.split(val_set, len(val_set) / number_samples))
    test_set = np.stack(np.split(test_set, len(test_set) / number_samples))

    return train_set, val_set, test_set, train_set_last, val_set_last, test_set_last


def prepare_single_set(data_set,  number_samples, drop_col=['last_updated', 'day', 'month', 'year', 'symbol']):
    start_date = np.datetime64('2019-04-07 00:00:00')
    data_set = data_set[(data_set['last_updated'] < start_date)]
    days = count_days(data_set)
    data_set = extract_data_by_date(data_set, days, number_samples)
    data_set = data_set.drop(drop_col, axis=1)
    data_set_last = data_set[len(data_set)//number_samples*number_samples:].to_numpy()
    data_set = data_set[:len(data_set)//number_samples*number_samples]
    print(data_set.shape)
    data_set = np.stack(np.split(data_set, len(data_set) / number_samples))
    return data_set, data_set_last
