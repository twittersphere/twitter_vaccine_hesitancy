import pandas as pd
import numpy as np
from src.embedding_generation_and_CTM.embedding_generation import read_data_and_filter


def main():
    world_data_path = "data/raw/daily_data_parquet"
    dataframes_path = "data/processed/data_frames"
    probs_saving_path = "data/processed/CTM/probs"
    cleaned_world_anti_ids = pd.read_parquet(
        f"{dataframes_path}/cleaned_world_anti_ids.parquet")

    id_date = read_data_and_filter(world_data_path, cleaned_world_anti_ids,
                                   columns=['id', 'author_id', 'created_at'])

    id_date['created_at'] = pd.to_datetime(id_date['created_at'])
    cleaned_probs = pd.read_parquet(
        f"{probs_saving_path}/cleaned_probs_8_200_0.2.parquet")

    # Assign the topic with the highest probability to each tweet
    id_date['topic'] = np.argmax(
        cleaned_probs.values, axis=1) + 1  # start from 1

    # Save the id_date dataframe
    id_date[['id', 'created_at']].to_parquet(
        f"{dataframes_path}/id_date_topic_8_200_0.2.parquet")

    id_date.drop(columns=['id'], inplace=True)
    del cleaned_probs

    # Count the number of tweets per topic per day
    date_topic_count = id_date[['created_at', 'topic']].value_counts(
    ).reset_index().rename(columns={0: 'counts'}).sort_values('created_at')

    date_topic_count.to_parquet(
        f"{dataframes_path}/date_topic_count_8_200_0.2.parquet")

    # Authorid Date Topic Count
    authorid_date_topic_count = id_date.groupby(
        ['author_id', 'created_at', 'topic']).size().reset_index(
        name='count').sort_values('created_at')

    authorid_date_topic_count.to_parquet(
        f"{dataframes_path}/authorid_date_topic_count_8_200_0.2.parquet")


if __name__ == '__main__':
    main()
