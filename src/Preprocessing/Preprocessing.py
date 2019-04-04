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
        self.weather_opts = ['temperature', 'symbol', 'precipitation', 'windSpeed', 'pressure', 'windDirection',
                             'humidity', 'fog', 'cloudiness', 'lowClouds', 'mediumClouds', 'highClouds',
                             'dewpointTemperature']
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
                self.set(dataname, df, domain + room)

            self.save_df(dataname, domain + room)

    def load_weather_data(self, *, table='states', domain='sensor', dataname='weather', **kwargs):
        for opt in self.weather_opts:
            sqlquery = 'select state,last_updated from ' \
                       '' + table + ' where entity_id like "%' + opt + '%"'
            df[opt] = pd.read_sql_query(sqlquery, self.conn)



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

    def add_weather_features(self, dataname):
        feature = []
        for row in state_climate_living_room_as_x.index:
            feature.append(add_wp(time_steps, set_temp_wp, state_climate_living_room_as_x.loc[row, 'time']))

