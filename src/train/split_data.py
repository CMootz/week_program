import sklearn
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

    return dates_train, dates_test, dates_val


def extract_data_by_date(data_set, date_list, number_samples=1439):
    df = data_set
    new_df = pd.DataFrame()
    for item in date_list:
        start_date = np.datetime64(item+' 00:00:00')
        end_date = np.datetime64(item+' 23:59:59')
        if new_df.empty:
            new_df = new_df.append(df[(df['last_updated'] > start_date) & (df['last_updated'] < end_date)])
        else:
            new_df = pd.concat([new_df, df[(df['last_updated'] > start_date) & (df['last_updated'] < end_date)]])
        new_df = new_df.reset_index(drop=True)
    return new_df


def train_val_test_split(data_set, val_size=0.25, test_size=0.25, number_samples=1439, drop_col=['last_updated']):
    days = count_days(data_set)
    train_days, val_days, test_days = split_days(days, val_size, test_size)
    train_set = extract_data_by_date(data_set, train_days)
    val_set = extract_data_by_date(data_set, val_days)
    test_set = extract_data_by_date(data_set, test_days)

    train_set = train_set.drop(columns=drop_col)
    val_set = val_set.drop(columns=drop_col)
    test_set = test_set.drop(columns=drop_col)

    print(len(train_set), len(val_set), len(test_set))
    train_set = np.stack(np.split(train_set, len(train_set)/number_samples))
    val_set = np.stack(np.split(val_set, len(val_set) / number_samples))
    test_set = np.stack(np.split(test_set, len(test_set) / number_samples))

    return train_set, val_set, test_set
