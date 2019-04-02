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

        root_dir = os.path.dirname(os.path.abspath(__file__))
        directory_template = '{root_dir}/../../data/{name}/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

        if not os.path.exists(self.directory):
            print(f'Creating "{name}" directory for you!')
            os.makedirs(self.directory)

        shutil.copyfile(databasefolder + databasename, self.directory + databasename)
        self.conn = sql.connect(self.directory + databasename)
        self.load_data()

    def load_data(self, *, room, dataname='train_raw', **kwargs):

        sqlquery = 'select entity_id,state,attributes,last_updated from states where entity_id like "%climate%' \
                   '' + room + '%wand%"'
        self.set(dataname, pd.read_sql_query(sqlquery, self.conn))
        self.save_df(dataname)

    def get(self, name):
        return self.data[name]

    def set(self, name, value):
        self.data[name] = value

    def _shuffle_df(self, name):
        self.data[name] = self.data[name].sample(frac=1).reset_index(drop=True)
        return self.data[name]

    def save_df(self, name, filetype='csv', *, index=False, **kwargs):
        filepath = f'{self.directory}/{name}.{filetype}'
        getattr(self.data[name], f'to_{filetype}')(filepath, index=index, **kwargs)

    @classmethod
    def load_df(cls, directory, name, filetype='csv', **kwargs):
        filepath = f'{directory}/{name}.{filetype}'
        return getattr(pd, f'read_{filetype}')(filepath, **kwargs)
