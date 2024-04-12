import os
import pandas as pd
from tqdm import tqdm
from src.scripts.utils import save_log

def main():
    data_path = "data/processed/daily_us_data_parquet"
    df_path = "data/processed/dataframes/unique_geo_ids_with_states.parquet"
    files = os.listdir(data_path)

    geo_locations = pd.read_parquet(df_path)
    geo_locations = geo_locations.rename(columns={'id':'geo_place_id'})
    geo_locations = geo_locations.set_index('geo_place_id')

    for file_name in tqdm(files):
        df = pd.read_parquet(f"{data_path}/{file_name}")
        df = df.drop(columns=['longitude', 'latitude', 'coordinates_type'])

        joined_df = df.join(geo_locations, on='geo_place_id')
        joined_df.to_parquet(f"{data_path}/{file_name}", index=False)

    save_log("locational_tweets_merged_with_states")

if __name__ == "__main__":
    main()