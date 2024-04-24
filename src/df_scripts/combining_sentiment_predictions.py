import os
from src.scripts.read_data import ReadData

combined_data_saving_path = "data/processed/tweet_sentiment_predictions"
sentiment_path = f"{combined_data_saving_path}/all_world"

def read_world_sentiments():
    max_idx = [int(i.split('-')[-1][:-4]) for i in os.listdir(sentiment_path)]
    max_idx = max(max_idx) // 1200
    file_names = [f"between-{i*1200}-{(i+1)*1200}.parquet" for i in range(max_idx)]

    world_data_sentiments = ReadData(sentiment_path, file_format='parquet',
                                     filter_tweets=False)
    world_data_sentiments.file_list = file_names
    world_data_sentiments.read_files_and_combine_data()

    return world_data_sentiments.data

def get_us_ids():
    us_data_path = "data/raw/daily_us_data_parquet"
    us_data = ReadData(us_data_path, column_list=['id'], file_format='parquet')
    us_data.read_files_and_combine_data()
    us_ids = set(us_data.data['id'].values.tolist())

    return us_ids

def save_combined_data(data, file_):
    data.to_parquet(file_, index=False)

def main():
    world_data_sentiments = read_world_sentiments()
    file_ = f"{combined_data_saving_path}/world_data_sentiments_raw.parquet"
    save_combined_data(world_data_sentiments, file_)

    us_ids = get_us_ids()
    us_data_sentiments = world_data_sentiments[
                    world_data_sentiments['id'].isin(us_ids)
                    ].reset_index(drop=True)
    
    file_ = f"{combined_data_saving_path}/us_data_sentiments_raw.parquet"
    save_combined_data(us_data_sentiments, file_)

if __name__ == "__main__":
    main()