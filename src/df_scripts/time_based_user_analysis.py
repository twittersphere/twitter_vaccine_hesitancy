import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm


class TimeBasedUserAnalysis:
    def __init__(self, authorid_date_sentiment_counts):
        self.authorid_date_sentiment_counts = authorid_date_sentiment_counts

    def find_next_date(self, date, month=True):
        if month:
            greater_and_equal_to = date + pd.DateOffset(months=1)
        else:
            greater_and_equal_to = date + pd.DateOffset(days=1)

        return greater_and_equal_to

    def find_users_and_generate_new_dfs(self, date, next_date):
        df_before_date = self.authorid_date_sentiment_counts[
            self.authorid_date_sentiment_counts['created_at'] < date]
        df_after_next_date = self.authorid_date_sentiment_counts[
            self.authorid_date_sentiment_counts['created_at'] >= next_date]

        df_before_date['before_after'] = ['Before'] * df_before_date.shape[0]
        df_after_next_date['before_after'] = [
            'After'] * df_after_next_date.shape[0]

        users_before = df_before_date['author_ids'].values
        users_after = df_after_next_date['author_ids'].values

        common_users = set(users_before).intersection(users_after)

        return common_users, pd.concat(
            [df_before_date, df_after_next_date],
            axis=0).reset_index(
            drop=True)

    def creating_df(self, date, month=True):
        next_date = self.find_next_date(date, month=month)

        unique_users, filtered_df = self.find_users_and_generate_new_dfs(
            date, next_date)
        average = filtered_df[filtered_df['author_ids'].isin(unique_users)]
        average = average.groupby(
            ['author_ids', 'before_after', 'sentiment_labels']).agg(
            {'counts': 'sum'}).reset_index()
        average = average.pivot_table(
            index=['author_ids', 'before_after'],
            columns='sentiment_labels', values='counts', dropna=False).fillna(
            0).reset_index()
        average['positive_ratio'] = average[1].values / \
            (average[1].values + average[0].values)
        average = average[['author_ids', 'before_after',
                           'positive_ratio']].rename_axis(None, axis=1)
        return average

    def create_diff_df(self, date, month=True):
        avrg = self.creating_df(date, month=month)
        avrg = avrg.pivot_table(
            index='author_ids', values='positive_ratio',
            columns='before_after').reset_index()
        avrg['diff'] = avrg['After'].values - avrg['Before'].values
        avrg = avrg.drop(columns=['Before', 'After'])
        avrg['date'] = [date] * avrg.shape[0]
        return avrg


def main():
    dataframes_path = "data/processed/data_frames"
    author_ids_and_sentiments = pd.read_parquet(
        f"{dataframes_path}/authorid_date_sentiment_tweetcounts.parquet")

    author_ids_and_sentiments.rename(
        columns={'tweet_counts': 'counts'},
        inplace=True)

    unique_dates = np.sort(
        author_ids_and_sentiments['created_at'].unique())[1:-1]

    time_based_user_analysis = TimeBasedUserAnalysis(author_ids_and_sentiments)

    diff_dfs = []
    for date in tqdm(pd.to_datetime(unique_dates)):
        diff_dfs.append(time_based_user_analysis.create_diff_df(
            date, month=False))

    with open(f'{dataframes_path}/diff_dfs.pkl', 'wb') as f:
        pickle.dump(diff_dfs, f)


if __name__ == '__main__':
    main()
