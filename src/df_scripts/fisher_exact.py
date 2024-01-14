import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

def remove_rest_label(data):
    data = data[data['label'] != 'Rest']
    data = data.reset_index(drop=True)
    return data

def read_data(df_path, drop_id=True, remove_rest=True):
    us_ready = pd.read_parquet(f"{df_path}/us_id_date_filteredlabels.parquet")
    world_ready = pd.read_parquet(f"{df_path}/world_id_date_filteredlabels.parquet")

    if drop_id:
        us_ready = us_ready.drop(columns=['id'])
        world_ready = world_ready.drop(columns=['id'])

    if remove_rest:
        us_ready = remove_rest_label(us_ready)
        world_ready = remove_rest_label(world_ready)

    return us_ready, world_ready

def time_to_day_and_month(data, format='month'):
    format = 7 if format == 'month' else 11

    adjusted = data['created_at'].apply(lambda x: x[:format])
    data['created_at'] = pd.to_datetime(adjusted)

    return data

def group_by_created_at_and_label(data):
    data = data.value_counts(['created_at', 'label']).reset_index(name='counts')
    data = data.sort_values('created_at').reset_index(drop=True)
    return data

def to_pivot_table(data):
    data = data.pivot_table(values='counts', index='created_at',
                            columns='label').reset_index()
    return data

def prepare_data(us_ready, world_ready, daily_us_ready):
    # convert created_at to datetime object in day and month format
    us_ready = time_to_day_and_month(us_ready, format='month')
    world_ready = time_to_day_and_month(world_ready, format='month')
    daily_us_ready = time_to_day_and_month(daily_us_ready, format='day')

    # group by created_at and label
    us_ready = group_by_created_at_and_label(us_ready)
    world_ready = group_by_created_at_and_label(world_ready)
    daily_us_ready = group_by_created_at_and_label(daily_us_ready)

    # pivot table
    us_ready = to_pivot_table(us_ready)
    world_ready = to_pivot_table(world_ready)
    daily_us_ready = to_pivot_table(daily_us_ready)

    return us_ready, world_ready, daily_us_ready

def date_table(table, all_dates):
    fisher_tables = []

    for date in all_dates:
        date_count = table.loc[table['created_at'] == date, ["Pro", "Anti"]
                               ].values.tolist()[0]
        
        rest = table.loc[table['created_at'] != date, ["Pro", "Anti"]
                         ].values.sum(axis=0).tolist()

        fisher_tables.append(np.array([date_count, rest]))
    
    return fisher_tables

def convert2df_and_save(ready_data, fisher_vals, unique_dates, file_path):
    df = pd.DataFrame({'date': unique_dates,
                   'tweet_counts': ready_data[['Anti', 'Pro']].sum(axis=1).values,
                   'p_val':[i[1] for i in fisher_vals],
                   'odd_ratios':np.log([i[0] for i in fisher_vals])})
    df.to_parquet(file_path, index=False)

def main():
    df_path = "data/processed/data_frames"

    us_ready, world_ready = read_data(df_path)
    daily_us_ready = us_ready.copy(deep=True)

    us_ready, world_ready, daily_us_ready = prepare_data(us_ready, world_ready,
                                                        daily_us_ready)
    
    unique_dates = us_ready['created_at'].values
    daily_unique_dates = daily_us_ready['created_at'].values

    us_fisher_values = [fisher_exact(table, alternative='two-sided') \
                        for table in date_table(us_ready, unique_dates)]
    world_fisher_values = [fisher_exact(table, alternative='two-sided') \
                           for table in date_table(world_ready, unique_dates)]

    daily_us_fisher_values = [fisher_exact(table, alternative='two-sided') \
                for table in date_table(daily_us_ready, daily_unique_dates)]
    
    os.makedirs(f"{df_path}/fisher_exact", exist_ok=True)
    convert2df_and_save(us_ready, us_fisher_values, unique_dates,
                        f"{df_path}/fisher_exact/us_fisher_values.parquet")
    convert2df_and_save(world_ready, world_fisher_values, unique_dates,
                        f"{df_path}/fisher_exact/world_fisher_values.parquet")
    convert2df_and_save(daily_us_ready, daily_us_fisher_values,
                        daily_unique_dates,
                    f"{df_path}/fisher_exact/daily_us_fisher_values.parquet")

    