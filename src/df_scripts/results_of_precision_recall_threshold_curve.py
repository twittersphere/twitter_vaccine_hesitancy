import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split


def data_preparation(agreement1, agreement66):
    X_train, X_test, y_train, y_test = train_test_split(
        agreement1['text'].values, agreement1['label'].astype('int').values,
        test_size=0.2, random_state=42,
        stratify=agreement1['label'].astype('int').values)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    X_train = np.concatenate([X_train, agreement66['text'].values])
    y_train = np.concatenate([y_train, agreement66['label'].values])

    df_test = pd.DataFrame(
        np.concatenate(
            [X_test.reshape((-1, 1)),
             y_test.reshape((-1, 1))],
            axis=1),
        columns=['text', 'labels'])

    return X_train, X_test, y_train, y_test, X_val, y_val, df_test


def preparing_labels(y_test, probas_pred):
    zero_vs_rest_prob = probas_pred[:, 0]
    one_vs_rest_prob = probas_pred[:, 1]
    two_vs_rest_prob = probas_pred[:, 2]

    zero_vs_rest_true = np.where(y_test == 0, 1, 0)
    one_vs_rest_true = np.where(y_test == 1, 1, 0)
    two_vs_rest_true = np.where(y_test == 2, 1, 0)

    return zero_vs_rest_prob, one_vs_rest_prob, two_vs_rest_prob, zero_vs_rest_true, one_vs_rest_true, two_vs_rest_true


def precision_recall_threshold_curve(
        zero_vs_rest_true, one_vs_rest_true, two_vs_rest_true,
        zero_vs_rest_prob, one_vs_rest_prob, two_vs_rest_prob):
    precision_0, recall_0, thresholds_0 = precision_recall_curve(
        zero_vs_rest_true,
        zero_vs_rest_prob)
    precision_1, recall_1, thresholds_1 = precision_recall_curve(
        one_vs_rest_true,
        one_vs_rest_prob)
    precision_2, recall_2, thresholds_2 = precision_recall_curve(
        two_vs_rest_true,
        two_vs_rest_prob)

    zero_auc = auc(recall_0, precision_0)
    one_auc = auc(recall_1, precision_1)
    two_auc = auc(recall_2, precision_2)

    thresholds_0 = np.array([0] + thresholds_0.tolist())
    thresholds_1 = np.array([0] + thresholds_1.tolist())
    thresholds_2 = np.array([0] + thresholds_2.tolist())

    zero_auc_th = auc(thresholds_0, precision_0)
    one_auc_th = auc(thresholds_1, precision_1)
    two_auc_th = auc(thresholds_2, precision_2)

    threshes = [thresholds_0, thresholds_1, thresholds_2]
    idxs = [
        np.arange(threshes[idx].shape[0])[threshes[idx] >= 0.99][0]
        for idx in range(3)]

    precs_and_recs = [[precision_0[idxs[0]], recall_0[idxs[0]]], [
        precision_1[idxs[1]], recall_1[idxs[1]]], [precision_2[idxs[2]], recall_2[idxs[2]]]]

    result_dict = {'recall': {'neutral': recall_0, 'positive': recall_1, 'negative': recall_2},
                   'precision': {'neutral': precision_0, 'positive': precision_1, 'negative': precision_2},
                   'threshold': {'neutral': thresholds_0, 'positive': thresholds_1, 'negative': thresholds_2},
                   'precs_and_recs_auc': {'neutral': zero_auc, 'positive': one_auc, 'negative': two_auc},
                   'precs_and_recs': precs_and_recs,
                   'th_auc': {'neutral': zero_auc_th, 'positive': one_auc_th, 'negative': two_auc_th}}

    return result_dict


def main():
    data_path = "data/crowdbreaks_data"
    dataframes_path = "data/processed/data_frames"
    ids_and_labels_path = f"{data_path}/crowdbreaks_tweet_ids_and_labels.csv"
    crowdbreaks_data_path = f"{data_path}/crowdbreaks_tweets.parquet"

    ids_and_labels = pd.read_csv(ids_and_labels_path)
    tweets = pd.read_parquet(crowdbreaks_data_path)

    tweet_and_label = ids_and_labels.join(
        tweets.rename(columns={'id': 'tweet_id'}).set_index('tweet_id'),
        on='tweet_id', rsuffix='_')
    tweet_and_label = tweet_and_label.dropna().drop(columns=['label_'])
    tweet_and_label = tweet_and_label[tweet_and_label['agreement'] >= 0.66].reset_index(
        drop=True)

    mapping = {0: 0, 1: 1, -1: 2}
    tweet_and_label['label'] = tweet_and_label['label'].map(mapping)

    agreement1 = tweet_and_label[tweet_and_label['agreement'] == 1.0].reset_index(
        drop=True)
    agreement66 = tweet_and_label[(tweet_and_label['agreement'] >= 0.66) & (
        tweet_and_label['agreement'] < 1.0)].reset_index(drop=True)

    # data preparation
    X_train, X_test, y_train, y_test, X_val, y_val, df_test = data_preparation(
        agreement1, agreement66)

    # loading model
    with open("models/sentiment_models/best_model.db", 'rb') as f:
        model = pickle.load(f)

    predictions, raw_outputs = model.predict(df_test['text'].tolist())
    probas_pred = np.array(tf.nn.softmax(raw_outputs))

    # preparing labels
    zero_vs_rest_prob, one_vs_rest_prob, two_vs_rest_prob, zero_vs_rest_true, one_vs_rest_true, two_vs_rest_true = preparing_labels(
        y_test, probas_pred)

    # precision recall curve
    result_dict = precision_recall_threshold_curve(
        zero_vs_rest_true, one_vs_rest_true, two_vs_rest_true,
        zero_vs_rest_prob, one_vs_rest_prob, two_vs_rest_prob)

    with open(f"{dataframes_path}/precision_recall_threshold_curve_results.pkl", 'wb') as f:
        pickle.dump(result_dict, f, protocol=4)


if __name__ == '__main__':
    main()
