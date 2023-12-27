import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from src.scripts.tools import Tools
from src.scripts.read_data import ReadData
from src.sentiment_analysis.bert_model_fine_tuning import load_model
from src.scripts.utils import save_log

def main():
    tools = Tools()
    
    model_dir = "models/sentiment_models"
    model = load_model(model_dir, 'best_model')
    tweet_texts = ReadData("data/raw/daily_data_parquet", column_list=['text'],
                           filter_tweets=True)
    tweet_texts = tweet_texts.read_files_and_combine_data()['text'].values

    result_saving_path = "data/processed/tweet_sentiment_predictions/all_world"
    idx = 0
    for batch in tqdm(tools.create_chunks(tweet_texts, 1200)):
        file = f"{result_saving_path}/between-{idx*1200}-{(idx+1)*1200}.csv"
        if os.path.isfile(file):
            idx += 1
            continue
        predictions, raw_outputs = model.predict(batch.reshape(-1,).tolist())
        probs = tf.nn.softmax(raw_outputs, axis=1).numpy().astype(np.float16)
        pd.DataFrame(probs, columns=['Rest', "Pro", 'Anti']).to_csv(file,
                                                                    index=False)
        idx += 1

    save_log("tweet_sentiment_prediction")

if __name__ == "__main__":
    main()