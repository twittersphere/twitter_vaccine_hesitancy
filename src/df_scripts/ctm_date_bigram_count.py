import pandas as pd
from src.embedding_generation_and_CTM.embedding_generation import read_data_and_filter
from src.scripts.nlp_tools import NLPTools
from tqdm import tqdm


def bigram_count(collocations, id_date_text_topic):
    bigram_counter = {}

    for topic, collocation in tqdm(collocations.items()):
        collocation_str = ' '.join(collocation)
        bigram_counter[collocation_str] = {}
        for idx, row_serie in id_date_text_topic[id_date_text_topic['topic'] ==
                                                 topic].iterrows():
            bigram_counter[collocation_str].setdefault(
                row_serie['created_at'], 0)
            splitted_text = row_serie['text'].lower().split()

            is_in = True
            for word in collocation:
                if word not in splitted_text:
                    is_in = False
                    break

            if is_in:
                bigram_counter[collocation_str][row_serie['created_at']] += 1

    return bigram_counter


def main():
    nlp_tools = NLPTools()

    world_data_path = "data/raw/daily_data_parquet"
    dataframes_path = "data/processed/data_frames"

    cleaned_world_anti_ids = f"{dataframes_path}/cleaned_world_anti_ids.parquet"

    id_date_text = read_data_and_filter(
        world_data_path, cleaned_world_anti_ids,
        columns=['id', 'text'])

    id_date_text['text'] = nlp_tools.tokenize_and_remove_stopwords(
        id_date_text['text'].values)

    id_date_text['text'] = nlp_tools.pos_tag_sents(
        id_date_text['text'].values)

    id_date_text['text'] = nlp_tools.lemmatize_text(
        id_date_text['text'].values)

    id_date_topic = pd.read_parquet(
        f"{dataframes_path}/id_date_topic_8_200_0.2.parquet")

    id_date_text_topic = id_date_text.join(
        id_date_topic.set_index('id'), on='id')

    del id_date_text, id_date_topic

    id_date_text_topic['created_at'] = id_date_text['created_at'].astype(
        'str').apply(lambda x: x[:10])

    collocations = {
        2: ['blood', 'clot'],
        4: ['flu', 'shot'],
        5: ['long', 'term', 'side', 'effect'],
        6: ['big', 'pharma'],
        7: ['immune', 'system']}

    bigram_counter = bigram_count(collocations, id_date_text_topic)
    unique_dates = id_date_text_topic['created_at'].unique()
    keywords = list(bigram_counter.keys())

    date_and_bigram_collocations_df = pd.DataFrame(
        {'date': unique_dates,
         keywords[0]:
         [bigram_counter[keywords[0]][date]
          if date in bigram_counter[keywords[0]] else 0
          for date in unique_dates],
         keywords[1]:
         [bigram_counter[keywords[1]][date]
          if date in bigram_counter[keywords[1]] else 0
          for date in unique_dates],
         keywords[2]:
         [bigram_counter[keywords[2]][date]
          if date in bigram_counter[keywords[2]] else 0
          for date in unique_dates],
         keywords[3]:
         [bigram_counter[keywords[3]][date]
          if date in bigram_counter[keywords[3]] else 0
          for date in unique_dates],
         keywords[4]:
         [bigram_counter[keywords[4]][date]
          if date in bigram_counter[keywords[4]] else 0
          for date in unique_dates]})

    date_and_bigram_collocations_df.to_parquet(
        f"{dataframes_path}/date_and_bigram_collocations_df.parquet")


if __name__ == "__main__":
    main()
