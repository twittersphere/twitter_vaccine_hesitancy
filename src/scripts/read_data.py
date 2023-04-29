import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from nltk.tokenize import RegexpTokenizer

from tools import Tools

class ReadData:
    def __init__(self, data_path, column_list=None, filter_tweets=True, custom_filter=None) -> None:
        self.data_path = data_path
        self.column_list = column_list
        self.filter_tweets = filter_tweets
        self.TOKENIZER = RegexpTokenizer(r'\w+')
        self.file_list = os.listdir(self.data_path)
        self.custom_filter = custom_filter
        self.tools = Tools()
        self.data = None

    def _read_a_csv(self, file_name):
        df = pd.read_csv(f"{self.data_path}/{file_name}")
        if self.filter_tweets:
            df = df[df['text'].notnull()]
            df = df[df['text'].apply(lambda x: len(self.TOKENIZER.tokenize(x)) >= 10)]

        if self.custom_filter is not None:
            df = df[eval(self.custom_filter)]
        
        if self.column_list is None:
            return df
        return df[self.column_list]

    def read_csvs(self, processes=8):
        with Pool(processes=processes) as pool:
            self.data = pool.map(self._read_a_csv, self.file_list)

    def combine_data(self, batch_size=10):
        self.data = self.tools.concatenate_data(self.data, batch_size, concat_type='pd')
        if self.filter_tweets:
            self.data = self.data.drop_duplicates('text')
            self.data = self.data.reset_index(drop=True)

    def read_csvs_and_combine_data(self, processes=8, batch_size=10):
        self.read_csvs(processes=processes)
        self.combine_data(batch_size=batch_size)

    