import pandas as pd
from datetime import timedelta

from src.embedding_generation_and_CTM.embedding_generation import read_data_and_filter


def main():
    world_data_path = "data/raw/daily_data_parquet"
    dataframes_path = "data/processed/data_frames"
    cleaned_world_anti_ids = pd.read_parquet(
        f"{dataframes_path}/cleaned_world_anti_ids.parquet")

    id_date = read_data_and_filter(world_data_path, cleaned_world_anti_ids,
                                   columns=['id', 'created_at'])

    id_date = id_date.drop(columns=['id'])  # free id memory
    id_date.data['created_at'] = pd.to_datetime(
        id_date.data['created_at']) - timedelta(hours=5)

    df = id_date.data.copy(deep=True)
    df = df.set_index('created_at')

    hours = [f'{i:02d}:00' for i in range(24)] + ['00:00']
    tweet_counts = [df.between_time(hours[idx],
                                    hours[idx + 1]).shape[0]
                    for idx in range(len(hours) - 1)]

    plot_df = pd.DataFrame({'hour': hours[:-1], 'tweet_count': tweet_counts})
    plot_df.to_parquet(f'{dataframes_path}/time_tweet_number_analysis.parquet')


if __name__ == '__main__':
    main()
