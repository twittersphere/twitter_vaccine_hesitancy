import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from nltk.tokenize import RegexpTokenizer

from tools import Tools

class ReadData:
    def __init__(self, data_path, column_list=None, filter_tweets=True,
                 custom_filter=None, file_format='parquet') -> None:
        self.data_path = data_path
        self.column_list = column_list
        self.filter_tweets = filter_tweets
        self.TOKENIZER = RegexpTokenizer(r'\w+')
        self.file_list = os.listdir(self.data_path)
        self.custom_filter = custom_filter
        self.tools = Tools()
        self.data = None
        self.file_format = file_format

    def _read_a_file(self, file_name):
        if self.file_format == 'csv':
            df = pd.read_csv(f"{self.data_path}/{file_name}")
        elif self.file_format == 'parquet':
            df = pd.read_parquet(f"{self.data_path}/{file_name}")
            
        if self.filter_tweets:
            df = df[df['text'].notnull()]
            df = df[df['text'].apply(lambda x: len(
                                        self.TOKENIZER.tokenize(x)) >= 10)]

        if self.custom_filter is not None:
            df = df[eval(self.custom_filter)]
        
        if self.column_list is None:
            return df
        return df[self.column_list]

    def read_files(self, processes=8):
        with Pool(processes=processes) as pool:
            self.data = pool.map(self._read_a_file, self.file_list)

    def combine_data(self, batch_size=10):
        self.data = self.tools.concatenate_data(self.data, batch_size,
                                                concat_type='pd')
        if self.filter_tweets:
            df_path = "data/processed/data_frames"
            if os.path.isfile(f"{df_path}/processed_ids.parquet"):
                processed_ids = pd.read_parquet(
                                f"{df_path}/processed_ids.parquet")
                self.data = self.data[~self.data['id'].isin(
                                                    processed_ids['id'])]
            else:
                self.data = self.data.drop_duplicates('text')
                os.makedirs(f"{df_path}", exist_ok=True)
                pd.DataFrame({'id': self.data['id'].values}).to_parquet(
                                f"{df_path}/processed_ids.parquet")
            self.data = self.data.reset_index(drop=True)

    def read_files_and_combine_data(self, processes=8, batch_size=10):
        self.read_files(processes=processes)
        self.combine_data(batch_size=batch_size)

    