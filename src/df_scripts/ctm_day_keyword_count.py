import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.embedding_generation_and_CTM.embedding_generation import read_data_and_filter


def save_topic_keyword_count(date_text_topic, top_50_keywords,
                             day_keyword_count_path, unique_date_df):
    for topic in tqdm(range(1, 9)):
        topic_df = date_text_topic[date_text_topic['topic'] == topic]

        topic_keywords = None
        for keyword in top_50_keywords[f'topic_{topic}']:
            keyword_df = topic_df[topic_df['text'].apply(
                lambda x: keyword in x)]
            keyword_df = keyword_df.value_counts(
                ['date']).reset_index(name='count')
            keyword_df = unique_date_df.join(
                keyword_df.set_index('date'), on='date')
            keyword_df = keyword_df[['date', 'count']].fillna(0)
            keyword_df['keyword'] = np.array([keyword]*keyword_df.shape[0])
            keyword_df = keyword_df[['date', 'keyword', 'count']]

            if topic_keywords is None:
                topic_keywords = keyword_df
            else:
                topic_keywords = pd.concat(
                    [topic_keywords, keyword_df], axis=0)

        topic_keywords.to_parquet(
            f"{day_keyword_count_path}/topic{topic}.parquet", index=False)


def main():
    world_data_path = "data/raw/daily_data_parquet"
    dataframes_path = "data/processed/data_frames"
    topics_saving_path = "data/processed/CTM/topics"

    cleaned_world_anti_ids = f"{dataframes_path}/cleaned_world_anti_ids.parquet"

    id_date_text = read_data_and_filter(
        world_data_path, cleaned_world_anti_ids,
        columns=['id', 'text'])

    id_date_topic = pd.read_parquet(
        f"{dataframes_path}/id_date_topic_8_200_0.2.parquet")

    id_date_text_topic = id_date_text.join(
        id_date_topic.set_index('id'), on='id')

    del id_date_text, id_date_topic
    id_date_text_topic.drop(columns=['id'], inplace=True)

    id_date_text_topic['created_at'] = id_date_text['created_at'].astype(
        'str').apply(lambda x: x[:10])

    top_50_keywords = pd.read_parquet(
        f"{topics_saving_path}/topics_8_200_0.2.parquet")

    day_keyword_count_path = f"{topics_saving_path}/day_keyword_count_for_" \
        "topics_8_200_0.2"
    os.makedirs(day_keyword_count_path, exist_ok=True)

    dummy_value = np.arange(len(id_date_text_topic['created_at'].unique()))
    unique_date_df = pd.DataFrame(
        {'date': id_date_text_topic['created_at'].unique(),
         'dummy_value': dummy_value}).sort_values('date')

    save_topic_keyword_count(id_date_text_topic, top_50_keywords,
                             day_keyword_count_path, unique_date_df)


if __name__ == '__main__':
    main()
