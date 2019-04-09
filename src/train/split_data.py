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


def split_days(date_list, val_size=0.25, test_size=0.25, random_state=None):
    dates_train, dates_test = sklearn.model_selection.train_test_split(date_list,
                                                                       test_size=test_size,
                                                                       random_state=random_state)
    dates_train, dates_val = sklearn.model_selection.train_test_split(dates_train,
                                                                      test_size=val_size,
                                                                      random_state=random_state)
    return dates_train, dates_test, dates_val


def extract_data_by_date(data_set, date_list):
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


def train_val_test_split(data_set, val_size=0.25, test_size=0.25, random_state=None):
    days = count_days(data_set)
    train_days, val_days, test_days = split_days(days, val_size, test_size, random_state)
    train_set = extract_data_by_date(data_set, train_days)
    val_set = extract_data_by_date(data_set, val_days)
    test_set = extract_data_by_date(data_set, test_days)
    return train_set, val_set, test_set
