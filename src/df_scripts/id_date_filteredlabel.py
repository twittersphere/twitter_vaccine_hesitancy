import os
import numpy as np
import pandas as pd
from src.scripts.read_data import ReadData

sentiment_pred_path = "data/processed/tweet_sentiment_predictions"

def filter_data(sentiment_preds, data):
    filter_ = sentiment_preds.loc[:, ['Rest', 'Pro', 'Anti']].values
    filter_ = np.max(filter_, axis=1) >= 0.99
    filtered_data = sentiment_preds.loc[filter_, :]
    del filter_

    labels = np.argmax(filtered_data.loc[:, ['Rest', 'Pro', 'Anti']].values,
                       axis=1)
    filtered_data.loc[:, 'label'] = labels
    
    # map argmax values to Rest, Pro, Anti
    filtered_data.loc[:, 'label'] = filtered_data['label'].apply(
                                    lambda x: 'Rest' if x == 0 else (
                                    'Pro' if x == 1 else 'Anti'))
    filtered_data = filtered_data.drop(columns=['Rest', 'Pro', 'Anti'])

    data_ready = filtered_data.join(data.set_index("id"), on='id').dropna()
    data_ready = data_ready.reset_index(drop=True)

    return data_ready

def read_world_tweets():
    world_data_path = "/data/raw/daily_data_parquet"
    world_file = f"{sentiment_pred_path}/world_data_sentiments_raw.parquet"

    world_sentiment_preds = pd.read_parquet(world_file)

    read_data_world = ReadData(world_data_path, ['id', 'author_id', 'created_at'],
                               filter_tweets=True)
    read_data_world.read_csvs_and_combine_data()

    return world_sentiment_preds, read_data_world.data

def read_us_tweets():
    us_data_path = "/data/raw/US_daily_data_parquet"
    us_file = f"{sentiment_pred_path}/us_data_sentiments_raw.parquet"

    us_sentiment_preds = pd.read_parquet(us_file)

    read_data_us = ReadData(us_data_path, ['id', 'author_id', 'state', 'created_at'],
                               filter_tweets=True)
    read_data_us.read_csvs_and_combine_data()

    return us_sentiment_preds, read_data_us.data

def main():
    dataframes_path = "data/processed/data_frames"
    os.makedirs(dataframes_path, exist_ok=True)

    world_sentiment_preds, world_data = read_world_tweets()
    world_ready = filter_data(world_sentiment_preds, world_data)
    world_ready.to_parquet(f"{dataframes_path}/world_id_date_filteredlabels.parquet")
    del world_ready, world_data, world_sentiment_preds

    us_sentiment_preds, us_data = read_us_tweets()
    us_ready = filter_data(us_sentiment_preds, us_data)
    us_ready.to_parquet(f"{dataframes_path}/us_id_date_state_filteredlabels.parquet")
    del us_ready, us_data, us_sentiment_preds

if __name__ == "__main__":
    main()



    