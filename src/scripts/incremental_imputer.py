import numpy as np

class Imputer:
    def __init__(self) -> None:
        pass

    def impute_data(self, df, columns=None, inplace=True):
        if inplace is False:
            df = df.copy()

        if columns is None:
            for clm in df.columns:
                isna_ = df[clm].isna().values
                values = df[clm].values
                df[clm] = self._impute_an_array(values, isna_)

        elif type(columns) is list:
            for clm in columns:
                isna_ = df[clm].isna().values
                values = df[clm].values
                df[clm] = self._impute_an_array(values, isna_)

        elif type(columns) is str:
            isna_ = df[columns].isna().values
            values = df[columns].values
            df[columns] = self._impute_an_array(values, isna_)
        
        if inplace is False:
            return df
        
    def _impute_an_array(self, values, isna_):
        len_of_values = len(values)
        new_values = []

        idx = 0
        while idx < len_of_values:
            if not isna_[idx]:
                new_values.append(values[idx])
                idx += 1
            else:
                counter = 1
                while True:
                    if idx+counter == len_of_values:
                        for i in range(counter):
                            new_values.append(values[idx-1])
                        break
                    if not isna_[idx+counter]:
                        if idx == 0:
                            for i in range(idx+counter):
                                new_values.append(values[idx+counter])
                            break
                        diff = values[idx+counter] - values[idx-1]
                        increment_val = diff / (counter + 1)
                        for i in range(1, counter+1):
                            new_values.append(round(values[idx-1] + (increment_val * i), 2))

                        break
                    counter += 1
                idx += counter
            
        return np.ravel(new_values)