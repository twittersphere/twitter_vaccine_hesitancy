import pandas as pd
from src.scripts.read_data import ReadData

def get_data_readers(world_data_path, us_data_path):
    world_data_reader = ReadData(world_data_path, ['id', 'text', 'author_id'],
                               filter_tweets=True)
    world_data_reader.read_csvs_and_combine_data()

    us_data_reader = ReadData(us_data_path, ['id', 'text', 'latitude'],
                              filter_tweets=True)
    us_data_reader.read_csvs_and_combine_data()

    return world_data_reader, us_data_reader

def main():
    world_data_path = "data/raw/daily_data_parquet" # change path
    us_data_path = "data/raw/US_daily_data_parquet"
    dataframes_path = "data/processed/data_frames"

    world_data_reader, us_data_reader = get_data_readers(world_data_path,
                                                         us_data_path)

    num_of_tweets = world_data_reader.data.shape[0]
    num_of_author_ids = world_data_reader.data['author_id'].unique().shape[0]
    num_of_locational_tweets = us_data_reader.data.shape[0]

    numbers = [num_of_tweets, num_of_author_ids, num_of_locational_tweets]
    labels = ['Tweets', 'Users', 'Tweets with Geo-Location']

    df_duplicates_removed = pd.DataFrame({'Counts': numbers, 'Labels': labels})
    df_duplicates_removed.to_parquet(f"{dataframes_path}/eda_df.parquet",
                                     index=False)

if __name__ == '__main__':
    main()