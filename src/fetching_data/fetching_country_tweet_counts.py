import os
import pickle
import pandas as pd
from tqdm import tqdm
from searchtweets import gen_request_parameters, collect_results
from src.scripts.utils import save_log, load_twitter_cretentials_yaml


def find_tweet_counts_per_country(search_args, country):
    # ie. place_country:US, lang:en
    query = f"(vaccine OR vaccination) place_country:{country} " \
        f"has:geo lang:en -is:retweet"
    query = gen_request_parameters(query, granularity="",
                                   start_time="2020-01-01T00:00",
                                   end_time="2022-01-01T00:00")

    vaccine_vaccination_tweets = collect_results(query,
                                                 result_stream_args=search_args)

    return vaccine_vaccination_tweets


def fetch_country_tweet_counts(countries, saving_path):
    os.makedirs(saving_path, exist_ok=True)

    for country in tqdm(countries['Code'].values):
        country_saving_path = f'{saving_path}/{country}.db'
        if os.path.exists(country_saving_path):
            continue

        try:
            counts = find_tweet_counts_per_country(country)
            with open(country_saving_path, 'wb') as f:
                pickle.dump(counts, f)
        except:
            pass


def main():
    search_args = load_twitter_cretentials_yaml()
    search_args['endpoint'] = "https://api.twitter.com/2/tweets/counts/all"

    countries = pd.read_csv('data/raw/country_list.csv')
    saving_path = 'data/raw/country_tweet_counts'
    fetch_country_tweet_counts(countries, saving_path)
    save_log('fetching_country_tweet_counts')


if __name__ == '__main__':
    main()
