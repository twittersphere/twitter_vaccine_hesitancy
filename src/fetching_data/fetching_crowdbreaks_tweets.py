import os
import pickle
import numpy as np
import pandas as pd
import tweepy as tw
from tqdm import tqdm
import src.scripts.utils as utils
from src.scripts.tools import Tools

def save_fetched_data(tools, data, api, raw_dbs_path):
    batches = tools.create_chunks(data['tweet_id'].values, 100)
    for counter, batch in tqdm(enumerate(list(batches))):
        data_saving_path = f"{raw_dbs_path}/batch_{counter}.db"
        if os.path.isfile(data_saving_path):
            continue
        call_ = api.lookup_statuses(batch)
        
        with open(data_saving_path, 'wb') as f:
            pickle.dump(call_, f)

def concatenate_fetched_data(raw_dbs_path, data_path):
    raw_dbs = os.listdir(raw_dbs_path)

    text = []
    for file_name in raw_dbs:
        with open(f"{raw_dbs_path}/{file_name}", 'rb') as f:
            db_obj = pickle.load(f)
        for status in db_obj:
            text.append([status.id_str, status.text])

    id_label_path = f"{data_path}/crowdbreaks_tweet_ids_and_labels.csv"
    ids_and_labels = pd.read_csv(id_label_path, dtype=str)

    ids_and_labels_hash = {}
    for i in ids_and_labels.values:
        ids_and_labels_hash[i[0]] = i[-1]

    for idx in range(len(text)):
        text[idx].append(ids_and_labels_hash[text[idx][0]])

    pd.DataFrame(np.array(text), columns=['id', 'text', 'label']).to_parquet(
        f"{data_path}/crowdbreaks_tweets.parquet", index=False)


def main():
    tools = Tools()
    credentials = utils.load_twitter_credentials_json()

    data_path = "data/crowdbreaks_data"
    crowdbreaks_data = pd.read_csv(f"{data_path}/crowdbreaks_data.csv")

    auth = tw.OAuthHandler(credentials['consumer_key'],
                           credentials['consumer_secret'])
    api = tw.API(auth)

    # save each batch of 100 tweets in a separate file
    # to avoid losing all the data if the connection is lost
    raw_dbs_path = f"{data_path}/raw_dbs"
    os.makedirs(raw_dbs_path, exist_ok=True)
    save_fetched_data(tools, crowdbreaks_data, api, raw_dbs_path)

    # concatenate all the batches into one file
    # to make it easier to work with
    concatenate_fetched_data(raw_dbs_path, data_path)

if __name__ == '__main__':
    main()
