import pandas as pd
from src.scripts.read_data import ReadData

# TODO: use world_id_date_filteredlabels.parquet and us version to generate ...tweetcounts.parquet

def get_combined_data():
    dataframes_path = "data/processed/data_frames"
    world_df_path = f"{dataframes_path}/world_id_date_filteredlabels.parquet"
    world_df = pd.read_parquet(world_df_path)

    world_data_path = "/data/raw/daily_data_parquet"
    read_data_world = ReadData(world_data_path, ['id', 'author_id'],
                               filter_tweets=True)
    read_data_world.read_csvs_and_combine_data()

    world_df = world_df.join(read_data_world.data.set_index("id"), on='id')
    world_df = world_df.drop(columns=['id'])
    # remove Rest from label
    world_df = world_df[world_df['label'] != 'Rest']

    return world_df

def format_date(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].dt.date
    return df

def main():
    world_df = get_combined_data()
    world_df = format_date(world_df)
    world_df = world_df.value_counts().reset_index(name='tweet_counts')
    world_df = world_df.rename(columns={'label': 'sentiment',
                                        'created_at': 'date'})

    world_df.to_parquet("data/processed/data_frames/" \
                        "authorid_date_sentiment_tweetcounts.parquet")

    