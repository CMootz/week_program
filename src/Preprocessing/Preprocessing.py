import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import dateutil
import sqlite3 as sql
import json
import shutil


class Preprocessing:
    def __init__(self, name, databasefolder, databasename):
        self.name = name.lower()
        self.data = {}
        self.splits = {}
        self.weather_opts = ['weather_temperature', 'symbol', 'precipitation', 'wind_speed', 'pressure',
                             'wind_direction','humidity', 'fog', 'cloudiness', 'low_clouds', 'medium_clouds',
                             'high_clouds', 'dewpoint_temperature']
        self.weather_data = {}

        root_dir = os.path.dirname(os.path.abspath(__file__))
        directory_template = '{root_dir}/../../data/{name}/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

        if not os.path.exists(self.directory):
            print(f'Creating "{name}" directory for you!')
            os.makedirs(self.directory)

        shutil.copyfile(databasefolder + databasename, self.directory + databasename)
        self.conn = sql.connect(self.directory + databasename)

    def load_data(self, *, table='states', domain='climate', rooms, dataname='train_raw', **kwargs):
        for room in rooms:
            sqlquery = 'select entity_id,state,attributes,last_updated from ' \
                       '' + table + ' where entity_id like "%' + domain + '%' \
                       '' + room + '%wand%"'
            df = pd.read_sql_query(sqlquery, self.conn)
            self.set(dataname, df, domain + room)

            if domain == 'climate':
                current_temp = []
                set_temp = []
                for value in df['attributes']:
                    json_acceptable_string = value.replace("'", "\"")
                    dicti = json.loads(json_acceptable_string)
                    current_temp.append(dicti['current_temperature'])
                    set_temp.append(dicti['temperature'])

                df['current_temp'] = current_temp
                df['set_temp'] = set_temp

            df.loc[:, 'last_updated'] = df.loc[:, 'last_updated'].map(lambda x: dateutil.parser.parse(x))
            df = df.set_index('last_updated').resample('10T').pad()
            df = df.reset_index()
            self.set(dataname, df, domain + room)
            self.save_df(dataname, domain + room)

    def get(self, name, appendices=''):
        if appendices != '':
            exp_name = name + '_' + appendices
        else:
            exp_name = name
        return self.data[exp_name]

    def set(self, name, value, appendices=''):
        if appendices != '':
            exp_name = name + '_' + appendices
        else:
            exp_name = name
        self.data[exp_name] = value

    def _shuffle_df(self, name):
        self.data[name] = self.data[name].sample(frac=1).reset_index(drop=True)
        return self.data[name]

    def save_df(self, dataname, appendices='', filetype='csv', *, index=False, **kwargs):
        filename = dataname+' '+appendices
        filepath = f'{self.directory}/{filename}.{filetype}'
        if appendices != '':
            exp_name = dataname + '_' + appendices
        else:
            exp_name = dataname
        getattr(self.data[exp_name], f'to_{filetype}')(filepath, index=index, **kwargs)

    @classmethod
    def load_df(cls, directory, name, filetype='csv', **kwargs):
        filepath = f'{directory}/{name}.{filetype}'
        return getattr(pd, f'read_{filetype}')(filepath, **kwargs)

    @classmethod
    def _select_feature_value(cls, time_steps, set_temps, time_row):
        for index, ti in enumerate(time_steps):
            if time_row < ti:
                return set_temps[index-1]

    def extract_weather_data(self, table='states'):
        for opt in self.weather_opts:
            sqlquery = 'select entity_id,state,last_updated from ' \
                       '' + table + ' where entity_id like "%' + opt + '%"'
            df = pd.read_sql_query(sqlquery, self.conn)
            df = df[df['entity_id'].str.contains(opt)][['state', 'last_updated']]
            df = df[df.state != 'unknown']
            df = df[df.state != 'none']
            df['state'] = df['state'].astype('float')
            df.loc[:, 'last_updated'] = df.loc[:, 'last_updated'].map(lambda x: dateutil.parser.parse(x))
            df = df.reset_index(drop=True)
            self.weather_data[opt] = df

    @classmethod
    def _add_feature_diff_t(clf, time_steps, feature, time_row):
        for index, ti in enumerate(time_steps):
            if time_row < ti:
                return feature[index]

    @classmethod
    def normalize_datet(clf, dataframe):
        dataframe['day'] = dataframe['last_updated'].dt.day
        dataframe['month'] = dataframe['last_updated'].dt.month
        dataframe['year'] = dataframe['last_updated'].dt.year
        #dataframe['time'] = dataframe['last_updated'].dt.time
        #dataframe['weekday'] = dataframe['last_updated'].dt.weekday
        #dataframe['time'] = dataframe['time'].map(lambda x: x.hour + x.minute / 60.0)

    def build_x_frame(self, rooms, domain='climate', dataname='train_raw', new_name='train'):
        feature = []
        for room in rooms:
            main_frame = self.get(dataname + '_' + domain + room)
            for opt in self.weather_opts:
                for row in main_frame.index:
                    feature.append(self._add_feature_diff_t(self.weather_data[opt]['last_updated'],
                                                            self.weather_data[opt]['state'],
                                                            main_frame.loc[row, 'last_updated']))
                main_frame[opt] = feature
                feature = []

            self.normalize_datet(main_frame)
            main_frame = main_frame.drop(columns=['entity_id', 'state', 'attributes'])
            main_frame = main_frame.dropna(subset=['current_temp', 'set_temp'], how='all')
            main_frame = main_frame.dropna(thresh=main_frame.shape[1] - 2)
            main_frame = main_frame.dropna(axis=1)

            new_name = dataname.replace('_raw', '')

            self.set(new_name, main_frame, domain + room)
            self.save_df(new_name, domain + room)

    def shuffle_df(self, name):
        self.data[name] = self.data[name].sample(frac=1).reset_index(drop=True)
        return self.data[name]
