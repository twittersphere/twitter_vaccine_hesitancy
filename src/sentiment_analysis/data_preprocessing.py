import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def filter_data_by_agreement(ids_and_labels, crowdbreaks_data):
    tweet_and_label = ids_and_labels.join(crowdbreaks_data.rename(
        columns={'id':'tweet_id'}).set_index('tweet_id'),
        on='tweet_id', rsuffix='_')
    tweet_and_label = tweet_and_label.dropna().drop(columns=['label_'])

    mapping = {0:0, 1:1, -1:2}
    tweet_and_label['label'] = tweet_and_label['label'].map(mapping)

    filter_gt_66 = tweet_and_label['agreement'] >= 0.66
    tweet_and_label = tweet_and_label[filter_gt_66].reset_index(drop=True)

    filter_gt_66 = tweet_and_label['agreement'] >= 0.66
    filter_eq_1 = tweet_and_label['agreement'] == 1.0
    filter_lt_1 = tweet_and_label['agreement'] < 1.0

    agreement1 = tweet_and_label[filter_eq_1].reset_index(drop=True)
    agreement66 = tweet_and_label[(filter_gt_66) & (filter_lt_1)].reset_index(
                                                            drop=True)

    return agreement1, agreement66, tweet_and_label

def train_test_split_dfs(agreement1, agreement66, tweet_and_label):
    X_train, X_test, y_train, y_test = train_test_split(
                    agreement1['text'].values,
                    agreement1['label'].astype('int').values,
                    test_size=int(tweet_and_label.shape[0] * 0.2),
                    random_state=42,
                    stratify = agreement1['label'].astype('int').values)
    
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=0.5,
                                                    random_state=42,
                                                    stratify = y_test)
    
    X_train = np.concatenate([X_train, agreement66['text'].values])
    y_train = np.concatenate([y_train, agreement66['label'].values])

    df_train = pd.DataFrame(np.concatenate([X_train.reshape((-1, 1)),
                                            y_train.reshape((-1, 1))], axis=1),
                                    columns=['text', 'labels'])
    df_test = pd.DataFrame(np.concatenate([X_test.reshape((-1, 1)),
                                           y_test.reshape((-1, 1))], axis=1),
                                    columns=['text', 'labels'])
    df_val = pd.DataFrame(np.concatenate([X_val.reshape((-1, 1)),
                                          y_val.reshape((-1, 1))], axis=1),
                                    columns=['text', 'labels'])
    
    return df_train, df_test, df_val

def get_train_val_test_dfs():
    data_path = "data/crowdbreaks_data"
    ids_and_labels_path = f"{data_path}/crowdbreaks_tweet_ids_and_labels.csv"
    crowdbreaks_data_path = f"{data_path}/crowdbreaks_tweets.parquet"

    ids_and_labels = pd.read_csv(ids_and_labels_path)
    crowdbreaks_data = pd.read_csv(crowdbreaks_data_path)

    agreement1, agreement66, tweet_and_label = filter_data_by_agreement(
                                                        ids_and_labels,
                                                        crowdbreaks_data)
    
    df_train, df_test, df_val = train_test_split_dfs(agreement1,
                                                     agreement66,
                                                     tweet_and_label)
    
    return df_train, df_test, df_val