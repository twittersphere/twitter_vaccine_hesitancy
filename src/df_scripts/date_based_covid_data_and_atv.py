import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.scripts.incremental_imputer import Imputer

def date_range(start, end):
    delta = end - start  # as timedelta
    days = [start + timedelta(days=i) for i in range(delta.days)]
    return days

def get_covid_data(covid_data_path):
    covid_data = pd.read_csv(f"{covid_data_path}/owid-covid-data.csv")
    covid_data = covid_data[covid_data['iso_code'] == "USA"][[
        'date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
        'people_vaccinated_per_hundred']]
    covid_data['new_cases'] = covid_data['new_cases'].fillna(1.0)
    covid_data.loc[:, ['total_deaths', 'new_deaths']] = covid_data[[
        'total_deaths', 'new_deaths']].fillna(0.0)
    covid_data['people_vaccinated_per_hundred'] = covid_data[
                        'people_vaccinated_per_hundred'].fillna(0.0)

    dummy_values = [0.0]*len(date_range(datetime(2020, 1, 1), datetime(2020, 1, 22)))
    dummy_df = pd.DataFrame(np.array([date_range(datetime(2020, 1, 1),
                                                 datetime(2020, 1, 22)),
                                        dummy_values, dummy_values, dummy_values,
                                        dummy_values, dummy_values]).T,
                            columns=covid_data.columns)

    covid_data = pd.concat([dummy_df, covid_data], axis=0).reset_index(drop=True)

    covid_data['date'] = pd.to_datetime(covid_data['date'])
    covid_data = covid_data[covid_data['date'] < '2022-01-01']

    return covid_data

def join_fisher_and_arrange_values(covid_data, fisher_date):
    fisher_date.rename(columns={'created_at':'date'}, inplace=True)
    covid_data = covid_data.join(fisher_date.set_index('date'), on='date').drop(
        columns=['padj'])

    covid_data.iloc[:, 1:-3] = np.log10(covid_data.iloc[:, 1:-3].values.astype(np.float32))
    covid_data.iloc[:, -3] = covid_data.iloc[:, -3].values / 100
    covid_data = covid_data.replace(-np.inf, 0)

    return covid_data

def date_format(x):
    return '-'.join(np.array(x.split('/'))[[-1, 0, 1]].tolist())

def get_date_base_unemployment(date_base_unemployment):
    date_base_unemployment = date_base_unemployment.iloc[1:, [0, -3]].rename(
                columns={'Unnamed: 0':'date', 'I.U.R':'unemployment'}).dropna()
    date_base_unemployment['date'] = pd.to_datetime(date_base_unemployment['date'].apply(date_format))
    date_base_unemployment = date_base_unemployment[date_base_unemployment['date'] >= '2020-01-01']

    date_base_unemployment['unemployment'] = date_base_unemployment['unemployment'].astype(np.float32)

    return date_base_unemployment

def impute_covid_data(date_base_unemployment, incremental_imputer):
    days = date_range(datetime(2020, 1, 1), datetime(2022, 1, 1))
    date_base_unemployment_all = pd.DataFrame({'date': days,
                                            'unemployment': [np.nan] * len(days)})
    date_base_unemployment_all = date_base_unemployment_all.join(
        date_base_unemployment.set_index('date'),
        on='date', lsuffix='_').drop(columns=['unemployment_'])

    incremental_imputer.impute_data(date_base_unemployment_all, 'unemployment')

    covid_data = covid_data.join(date_base_unemployment_all.set_index('date'),
                                 on='date')
    
    return covid_data

def main():
    incremental_imputer = Imputer()

    dataframes_path = "/data/processed/data_frames"
    covid_data_path = "/data/raw/covid_data"
    socio_economic_params_path = "/data/raw/socio_economic_params"

    fisher_date = pd.read_parquet(f"{dataframes_path}/daily_us_fisher_values.parquet")
    fisher_date['date'] = pd.to_datetime(fisher_date['date'])

    date_base_unemployment = pd.read_parquet(
        f"{socio_economic_params_path}/date_base_unemployment.parquet")

    covid_data = get_covid_data(covid_data_path)
    covid_data = join_fisher_and_arrange_values(covid_data, fisher_date)

    date_base_unemployment = get_date_base_unemployment(date_base_unemployment)
    covid_data = impute_covid_data(date_base_unemployment, incremental_imputer)

    covid_data.to_parquet(f"{dataframes_path}/date_based_covid_data_ATV" \
                          "_and_unemployment.parquet", index=False)

    


