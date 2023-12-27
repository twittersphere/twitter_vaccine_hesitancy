import os
import pickle
import pandas as pd
from tqdm import tqdm

def load_country_tweet_counts(country_counts, country_tweet_counts_path):
    countries_and_counts = {}

    for country in tqdm(country_counts):
        code = country.split('.')[0]

        country_saving_path = f'{country_tweet_counts_path}/{country}'
        with open(country_saving_path, 'rb') as f:
            counts = pickle.load(f)

        countries_and_counts[code] = 0
        for request in counts:
            countries_and_counts[code] += request['meta']['total_tweet_count']

    return countries_and_counts


def convert_to_dataframe(countries_and_counts):
    srtd_counts = sorted(countries_and_counts.items(),
                           key=lambda item: item[1],
                           reverse=True)
    
    srtd_counts = pd.DataFrame([[k, v] for k,v in srtd_counts],
                          columns=['country', 'count'])

    srtd_counts['ratio'] = srtd_counts['count'].values / srtd_counts['count'].values.sum()

    countries = pd.read_csv('data/raw/country_list.csv')
    srtd_counts = srtd_counts.rename(columns={'country':'Code'})
    srtd_counts = srtd_counts.join(countries.set_index('Code'), on="Code")
    srtd_counts = srtd_counts[["Name", 'Code', 'count', 'ratio']].rename(
        columns={"Name":'name', "Code":'code'})

    return srtd_counts


def main():
    country_tweet_counts_path = "data/raw/country_tweet_counts"
    country_counts = os.listdir(country_tweet_counts_path)

    countries_and_counts = load_country_tweet_counts(country_counts,
                                                     country_tweet_counts_path)
    
    srtd_counts = convert_to_dataframe(countries_and_counts)

    file_name = 'data/tables/s1_table.xlsx'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    srtd_counts.to_excel(file_name, index=False)    

if __name__ == '__main__':
    main()