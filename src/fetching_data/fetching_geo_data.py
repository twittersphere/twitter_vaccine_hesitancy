import os
import time
import pickle
import tweepy as tw
import pandas as pd
import reverse_geocoder as rg
import src.scripts.utils as utils
from multiprocessing import Pool

class GeoDataFetcher:
    def __init__(self, credentials, main_configs, save_path):
        self.credentials = credentials
        self.main_configs = main_configs
        self.save_path = save_path
        self.auth = tw.OAuthHandler(credentials['consumer_key'],
                                    credentials['consumer_secret'])
        self.api = tw.API(self.auth)

        self.time_delays = [4, 16, 36, 64, 100, 144, 196, 256]
        self.length_of_td = len(self.time_delays)

    def get_ids(self, file_path):
        df = pd.read_parquet(file_path)
        return df['geo_place_id'].values

    def get_unique_ids(self, data_path):
        parquet_files = os.listdir(data_path)
        parquet_files = [f"{data_path}/{i}" for i in parquet_files]
        with Pool(os.cpu_count()) as p:
            unique_ids = list(p.map(self.get_ids, parquet_files))

        unique_ids = list(set([i for j in unique_ids for i in j]))
        return unique_ids

    def get_state(self, place_obj):
        us_states_and_abbreviations = self.main_configs[
                                        'us_states_and_abbreviations']

        aplace_type = place_obj.place_type
        if place_obj.full_name == "[Place name removed]":
            return ""
        elif aplace_type == 'city':
            return place_obj.full_name.split(",")[-1].strip()
        elif aplace_type == 'neighborhood':
            return place_obj.contained_within[0].full_name[-2:]
        elif aplace_type == 'admin':
            return us_states_and_abbreviations[place_obj.name]
        elif aplace_type == 'poi':
            centroid = place_obj.centroid
            search_result = rg.search(centroid[::-1])[0]
            try:
                return us_states_and_abbreviations[search_result['admin1']]
            except Exception as e:
                print(search_result['cc'])
                return search_result['cc']
        elif aplace_type == 'country':
            return ""
        
    def _get_place_obj(self, geo_id):
        sleep_idx = 0
        while True:
            try:
                place_obj = self.api.geo_id(geo_id)
                sleep_idx = 0
                break
            except:
                print(f"sleep {self.time_delays[sleep_idx]} seconds")
                time.sleep(self.time_delays[sleep_idx])
                sleep_idx += 1
                sleep_idx %= self.length_of_td

        return place_obj
        
    def get_place_obj(self, geo_id):
        if os.path.isfile(f"{self.save_path}/{geo_id}.db"):
            with open(f"{self.save_path}/{geo_id}.db", 'rb') as f:
                return pickle.load(f)
        else:
            return self._get_place_obj(geo_id)
    
    def save_place_obj(self, geo_id, place_obj):
        with open(f"{self.save_path}/{geo_id}.db", 'wb') as f:
            pickle.dump(place_obj, f)
    
    def fetch_all_place_objs(self, unique_ids):
        for geo_id in unique_ids:
            place_obj = self.get_place_obj(geo_id)
            self.save_place_obj(geo_id, place_obj)

    def convert2df(self, unique_ids, df_path):
        columns = ['id', 'name', 'state', 'place_type', 'geo_tag_count',
                   'longitude', 'latitude']
        
        locations = []
        for geo_id in unique_ids:
            place_obj = self.get_place_obj(geo_id)
            state = self.get_state(place_obj)
            locations.append([geo_id,
                                place_obj.name,
                                state,
                                place_obj.place_type,
                                place_obj.attributes.get('geotagCount', ''),
                                *getattr(place_obj, 'centroid', ['', ''])])   

        pd.DataFrame(locations, columns=columns).to_parquet(df_path,
                                                            index=False)  


def main():
    credentials = utils.load_twitter_credentials_json()
    main_configs = utils.load_main_config()
    data_path = "data/processed/daily_us_data_parquet"
    save_path = "data/processed/geo_data"
    df_path = "data/processed/dataframes/unique_geo_ids_with_states.parquet"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.dirname(df_path), exist_ok=True)

    geo_data_fetcher = GeoDataFetcher(credentials, main_configs, save_path)
    unique_ids = geo_data_fetcher.get_unique_ids(data_path)
    geo_data_fetcher.fetch_all_place_objs(unique_ids)
    geo_data_fetcher.convert2df(unique_ids, df_path)


if __name__ == "__main__":
    main()