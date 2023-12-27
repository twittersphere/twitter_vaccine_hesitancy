import numpy as np
import pandas as pd
from tqdm import tqdm

class Tools:
    def create_chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def chunks_pop(self, lst, n, dtype=None):
        if dtype is None:
            for i in range(0, len(lst), n):
                yield [lst.pop(0) for i in range(len(lst[:n]))]
        
        else:
            for i in range(0, len(lst), n):
                yield [lst.pop(0).astype(dtype) for i in range(len(lst[:n]))]

    def concatenate_data(self, list_of_data, n_concat, concat_type='np', dtype=None, axis=0):
        if dtype:
            concatenated_data = list_of_data.pop(0).astype(dtype)
        else:
            concatenated_data = list_of_data.pop(0)

        batches = list(self.chunks_pop(list_of_data, n_concat, dtype=dtype))

        if concat_type == 'np':
            for batch in tqdm(batches):
                concatenated_data = np.concatenate([concatenated_data, *batch], axis=axis)

        elif concat_type == 'pd':
            for batch in tqdm(batches):
                concatenated_data = pd.concat([concatenated_data, *batch], axis=axis)

        del batches
        return concatenated_data

    def smooth_data(self, df, smoothing_steps, columns=None, window_size=15, std=3):
        for _ in range(smoothing_steps):
            if type(columns) == list:
                for clm in columns:
                    self._smooth(df, clm, window_size=window_size, std=std)

            elif type(columns) == str:
                self._smooth(df, columns, window_size=window_size, std=std)

            elif columns is None:
                for clm in df.columns:
                    self._smooth(df, clm, window_size=window_size, std=std)

    def _smooth(self, df, clm, window_size=15, std=3):
        df[clm] = df[clm].rolling(window_size, win_type='gaussian', min_periods=1).mean(std=std)