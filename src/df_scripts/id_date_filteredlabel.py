import os
import numpy as np
import pandas as pd
from src.scripts.read_data import ReadData

def filter_data(sentiment_preds, data):
    filter_ = sentiment_preds.loc[:, ['Rest', 'Pro', 'Anti']].values
    filter_ = np.max(filter_, axis=1) >= 0.99
    filtered_data = sentiment_preds.loc[filter_, :]

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

def main():
    world_data_path = "/data/raw/daily_data_parquet"
    sentiment_pred_path = "data/processed/tweet_sentiment_predictions"
    world_file = f"{sentiment_pred_path}/world_data_sentiments_raw.parquet"
    us_file = f"{sentiment_pred_path}/us_data_sentiments_raw.parquet"

    world_sentiment_preds = pd.read_parquet(world_file)
    us_sentiment_preds = pd.read_parquet(us_file)

    read_data_world = ReadData(world_data_path, ['id', 'created_at'],
                               filter_tweets=True)
    read_data_world.read_csvs_and_combine_data()

    us_ready = filter_data(us_sentiment_preds, read_data_world.data)
    world_ready = filter_data(world_sentiment_preds, read_data_world.data)

    dataframes_path = "data/processed/data_frames"
    os.makedirs(dataframes_path, exist_ok=True)

    us_ready.to_parquet(f"{dataframes_path}/us_id_date_filteredlabels.parquet")
    world_ready.to_parquet(f"{dataframes_path}/world_id_date_filteredlabels.parquet")



    