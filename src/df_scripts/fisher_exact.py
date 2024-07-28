import os
import numpy as np
import pandas as pd
from betabinomial import pval_adj
from scipy.stats import fisher_exact


def remove_rest_label(data):
    data = data[data['label'] != 'Rest']
    data = data.reset_index(drop=True)
    return data


def read_date_data(df_path, drop_id=True, remove_rest=True):
    us_ready = pd.read_parquet(
        f"{df_path}/us_id_date_state_filteredlabels.parquet")
    world_ready = pd.read_parquet(
        f"{df_path}/world_id_date_filteredlabels.parquet")

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


def group_by_label(data, pivot_index='created_at'):
    data = data.value_counts([pivot_index, 'label']).reset_index(name='counts')
    return data


def to_pivot_table(data, pivot_index='created_at'):
    data = data.pivot_table(values='counts', index=pivot_index,
                            columns='label').reset_index()
    data = data.sort_values(pivot_index).reset_index(drop=True)
    return data


def prepare_data(us_ready, world_ready, daily_us_ready, state_us_ready):
    # convert created_at to datetime object in day and month format
    us_ready = time_to_day_and_month(us_ready, format='month')
    world_ready = time_to_day_and_month(world_ready, format='month')
    daily_us_ready = time_to_day_and_month(daily_us_ready, format='day')
    daily_world_ready = time_to_day_and_month(world_ready, format='day')

    # group by created_at and label
    us_ready = group_by_label(us_ready)
    world_ready = group_by_label(world_ready)
    daily_world_ready = group_by_label(daily_world_ready)
    daily_us_ready = group_by_label(daily_us_ready)
    state_us_ready = group_by_label(state_us_ready, pivot_index='state')

    # pivot table
    us_ready = to_pivot_table(us_ready)
    world_ready = to_pivot_table(world_ready)
    daily_world_ready = to_pivot_table(daily_world_ready)
    daily_us_ready = to_pivot_table(daily_us_ready)
    state_us_ready = to_pivot_table(state_us_ready, pivot_index='state')

    return us_ready, world_ready, daily_world_ready, daily_us_ready, state_us_ready


def date_table(table, all_dates):
    fisher_tables = []

    for date in all_dates:
        date_count = table.loc[table['created_at'] == date, ["Pro", "Anti"]
                               ].values.tolist()[0]

        rest = table.loc[table['created_at'] != date, ["Pro", "Anti"]
                         ].values.sum(axis=0).tolist()

        fisher_tables.append(np.array([date_count, rest]))

    return fisher_tables


def state_table(table, all_states):
    fisher_tables = []

    for state in all_states:
        state_count = table.loc[table['state'] == state, ["Pro", "Anti"]
                                ].values.tolist()[0]

        rest = table.loc[table['state'] != state, ["Pro", "Anti"]
                         ].values.sum(axis=0).tolist()

        fisher_tables.append(np.array([state_count, rest]))

    return fisher_tables


def convert2df_and_save(ready_data, fisher_vals, state_date, value, file_path):
    p_values = [i[1] for i in fisher_vals]
    odd_values = [i[0] for i in fisher_vals]

    df = pd.DataFrame(
        {state_date: value,
         'tweet_counts': ready_data[['Anti', 'Pro']].sum(axis=1).values,
         'padj': -np.log10(pval_adj(np.array(p_values))),
         'odd_ratios': np.log(odd_values)})
    df.to_parquet(file_path, index=False)


def main():
    df_path = "data/processed/data_frames"

    us_ready, world_ready = read_date_data(df_path)
    daily_us_ready = us_ready.copy(deep=True)

    us_ready, world_ready, daily_world_ready, daily_us_ready, state_us_ready = prepare_data(
        us_ready, world_ready, daily_us_ready)

    unique_dates = us_ready['created_at'].values
    daily_unique_dates = daily_us_ready['created_at'].values

    us_fisher_values = [fisher_exact(table, alternative='two-sided')
                        for table in date_table(us_ready, unique_dates)]
    world_fisher_values = [fisher_exact(table, alternative='two-sided')
                           for table in date_table(world_ready, unique_dates)]

    daily_us_fisher_values = [
        fisher_exact(table, alternative='two-sided')
        for table in date_table(daily_us_ready, daily_unique_dates)]

    daily_world_fisher_values = [
        fisher_exact(table, alternative='two-sided')
        for table in date_table(daily_world_ready, daily_unique_dates)]

    state_us_fisher_values = [
        fisher_exact(table, alternative='two-sided')
        for table in state_table(
            state_us_ready, state_us_ready['state'].values)]

    os.makedirs(f"{df_path}/fisher_exact", exist_ok=True)
    convert2df_and_save(
        us_ready, us_fisher_values, 'created_at', unique_dates,
        f"{df_path}/fisher_exact/monthly_us_fisher_values.parquet")
    convert2df_and_save(
        world_ready, world_fisher_values, 'created_at', unique_dates,
        f"{df_path}/fisher_exact/monthly_world_fisher_values.parquet")
    convert2df_and_save(
        daily_us_ready, daily_us_fisher_values, 'created_at',
        daily_unique_dates,
        f"{df_path}/fisher_exact/daily_us_fisher_values.parquet")
    convert2df_and_save(
        daily_world_ready, daily_world_fisher_values, 'created_at',
        daily_unique_dates,
        f"{df_path}/fisher_exact/daily_world_fisher_values.parquet")
    convert2df_and_save(
        state_us_ready, state_us_fisher_values, 'state',
        state_us_ready['state'].values,
        f"{df_path}/fisher_exact/state_us_fisher_values.parquet")
