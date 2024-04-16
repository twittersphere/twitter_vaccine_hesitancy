import pandas as pd
from src.df_scripts.id_date_filteredlabel import read_world_tweets, read_us_tweets

def get_combined_data(func):
    def wrapper(dataframes_path):
        df_path, data = func(dataframes_path)

        df = pd.read_parquet(df_path)
        df = df.join(data.set_index("id"), on='id')
        df = df.drop(columns=['id'])

        # remove Rest from label
        df = df[df['label'] != 'Rest']

        return df
    return wrapper

@get_combined_data
def get_combined_world_data(dataframes_path):
    world_df_path = f"{dataframes_path}/world_id_date_filteredlabels.parquet"
    world_data = read_world_tweets(['id', 'author_id'])

    return world_df_path, world_data

@get_combined_data
def get_combined_us_data(dataframes_path):
    us_df_path = f"{dataframes_path}/us_id_date_state_filteredlabels.parquet"
    us_data = read_us_tweets(['id', 'author_id'])

    return us_df_path, us_data

def format_date(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_at'] = df['created_at'].dt.date
    return df

def get_tweet_counted_df(df):
    df = df.value_counts().reset_index(name='tweet_counts')
    df = df.rename(columns={'label': 'sentiment',
                            'created_at': 'date'})
    return df

def main():
    dataframes_path = "data/processed/data_frames"
    world_df = get_combined_world_data(dataframes_path)
    us_df = get_combined_us_data(dataframes_path)

    world_df = format_date(world_df)
    us_df = format_date(us_df)

    world_df = get_tweet_counted_df(world_df)
    us_date_df = get_tweet_counted_df(us_df[['author_id', 'created_at', 'label']])
    us_state_df = get_tweet_counted_df(us_df[['author_id', 'state', 'label']])

    world_df.to_parquet("data/processed/data_frames/" \
                        "authorid_date_sentiment_tweetcounts.parquet")
    us_date_df.to_parquet("data/processed/data_frames/" \
                          "authorid_date_sentiment_tweetcounts_us.parquet")
    us_state_df.to_parquet("data/processed/data_frames/" \
                           "authorid_state_sentiment_tweetcounts_us.parquet")
    
if __name__ == "__main__":
    main()