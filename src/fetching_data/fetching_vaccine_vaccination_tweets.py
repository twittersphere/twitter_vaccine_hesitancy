import os
import json
import pandas as pd
from searchtweets import gen_request_parameters, collect_results
from src.scripts.utils import load_twitter_cretentials_yaml, save_log

def split_text(text):
    splitted_text = text.split()
    hashtags = []
    mentions = []
    links = []
    raw_text = []
    for i in splitted_text:
        if i.startswith('#'):
            hashtags.append(i)
        elif i.startswith('http'):
            links.append(i)
        elif i.startswith('@'):
            mentions.append(i)
        else:
            raw_text.append(i)
    
    return [' '.join(raw_text), ' '.join(hashtags),
            ' '.join(mentions), ' '.join(links)]

def day_step(start, end, search_args, columns):
    saving_path = f"data/raw/daily_data_parquet/{start[:10]}.parquet"

    if os.path.isfile(saving_path):
        return

    print("start:", start, "end:", end)
    vaccine_vaccination_tweets = get_collected_tweets(start, end, search_args)
    save_tweets(vaccine_vaccination_tweets, start, saving_path, columns)

def get_collected_tweets(start, end, search_args):
    # place_country:US, lang:en
    query = "(vaccine OR vaccination) lang:en -is:retweet"
    tweet_fields = "id,text,created_at,geo,author_id,lang"
    query = gen_request_parameters(query,
                                    results_per_call=500, granularity="",
                                    tweet_fields=tweet_fields,
                                    start_time=start, end_time=end)

    vaccine_vaccination_tweets = collect_results(query,
                                                 max_tweets=5000000,
                                                 result_stream_args=search_args)
    return vaccine_vaccination_tweets

def save_tweets(vaccine_vaccination_tweets, start, saving_path, columns):
    with open(f"data/raw/daily_data_json/{start[:10]}.json", 'w') as f:
        json.dump(vaccine_vaccination_tweets, f)

    daily_tweets = []
    for query in vaccine_vaccination_tweets:
        for atweet in query['data']:
            id_ = atweet['id']
            author_id = atweet['author_id']
            language = atweet['lang']
            text = split_text(atweet['text'])
            created_at = atweet['created_at']
            geo = atweet.get('geo')
            if geo:
                geo_coordinates = geo.get('coordinates')
                place_id = geo.get('place_id', '')
                if geo_coordinates:
                    geo_array = [place_id, geo['coordinates']['type'],
                                 *geo['coordinates']['coordinates']]
                else:
                    geo_array = [place_id, '', '', '']
            else:
                geo_array = ['', '', '', '']

            daily_tweets.append([id_, author_id, created_at,
                                 language, *text, *geo_array])

    df = pd.DataFrame(daily_tweets, columns=columns)
    df.to_parquet(saving_path, index=False)

def main():
    columns = ['id', 'author_id', 'created_at', 'language', 'text', 'hashtags',
               'mentions', 'links', 'geo_place_id', 'coordinates_type',
               'longitude', 'latitude']

    search_args = load_twitter_cretentials_yaml()
    search_args['endpoint'] = "https://api.twitter.com/2/tweets/search/recent"

    days = pd.date_range('2020-01-01', '2022-01-01', freq='D')
    for idx in range(len(days)-1):
        start = str(days[idx])[:16].replace(' ', 'T')
        end = str(days[idx+1])[:16].replace(' ', 'T')

        day_step(start, end, search_args, columns)

    save_log("fetching_vaccine_vaccination_tweets")

if __name__ == "__main__":
    main()