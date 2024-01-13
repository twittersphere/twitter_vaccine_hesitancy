import pandas as pd
from tqdm import tqdm
from src.scripts.read_data import ReadData
from src.scripts.utils import save_log

def insert_missed_tweets(date, us_data, world_data_path):
    sub_df = us_data[us_data['created_at'][:10] == date]
    us_ids = set(sub_df['id'].values.tolist())

    world_data = pd.read_parquet(f"{world_data_path}/{date}.parquet")
    missed_ids = us_ids.difference(set(world_data['id'].values.tolist()))

    if len(missed_ids) > 0:
        missed_df = sub_df[sub_df['id'].isin(missed_ids)]
        world_data = pd.concat([world_data, missed_df], axis=0)

        world_data = world_data.sort_values('created_at')
        missed_df.to_parquet(f"{world_data_path}/{date}.parquet",
                                index=False)

def main():
    us_data_path = "data/raw/daily_us_data_parquet"
    world_data_path = "data/raw/daily_data_parquet"

    us_data = ReadData(us_data_path, filter_tweets=False,
                       file_format='parquet')
    
    us_data.read_csvs_and_combine_data()
    unique_dates = set([row['created_at'][:10] for idx, row in us_data.iterrows()])
        
    for date in tqdm(unique_dates):
        insert_missed_tweets(date, us_data.data, world_data_path)

    save_log("insert_missed_tweets")
        

if __name__ == "__main__":
    main()