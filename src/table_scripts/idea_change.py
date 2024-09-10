import pandas as pd
from tqdm import tqdm


def find_authors_has_two_tweet(counts):
    authors_has_two_tweet = counts.groupby('author_id').agg(
        {'tweet_counts': 'sum'}).reset_index()
    filter_ = authors_has_two_tweet['tweet_counts'] == 2
    authors_has_two_tweet = authors_has_two_tweet[filter_]['author_id'].values

    return authors_has_two_tweet


def find_changes(authorid_date_sentiment_only_2_tweets):
    changes_df = {}

    for idx, arrow in tqdm(authorid_date_sentiment_only_2_tweets.iterrows()):
        is_exist = changes_df.get(arrow['author_id'], False)
        if is_exist is not False:
            changes_df[arrow['author_id']
                       ] = is_exist + f" to {arrow['sentiment_label']}"
        else:
            changes_df[arrow['author_id']] = str(arrow['sentiment_label'])

    changes_df = changes_df.items()
    changes_df = pd.DataFrame(
        changes_df, columns=['author_id', 'label_change'])

    return changes_df


def main():
    dataframes_path = "data/processed/data_frames"
    df_path = f"{dataframes_path}/authorid_date_sentiment_tweetcounts.parquet"
    authorid_date_sentiment_tweetcounts = pd.read_parquet(df_path)
    authors_has_two_tweet = find_authors_has_two_tweet(
        authorid_date_sentiment_tweetcounts)

    authorid_date_sentiment_tweetcounts['created_at'] = pd.to_datetime(
        authorid_date_sentiment_tweetcounts['created_at'])

    filter_ = authorid_date_sentiment_tweetcounts['author_id'].isin(
        set(authors_has_two_tweet.tolist()))
    authorid_date_sentiment_only_2_tweets = authorid_date_sentiment_tweetcounts[filter_].sort_values(
        'created_at').reset_index(drop=True)

    changes_df = find_changes(authorid_date_sentiment_only_2_tweets)
    n_of_authors = changes_df.shape[0]
    n_of_changes_df = changes_df.value_counts(
        'label_change').reset_index(name='counts')
    n_of_changes_df['ratio'] = n_of_changes_df['counts'].values / n_of_authors

    mapping = {"1 -> 1": "Positive to Positive",
               "0 -> 0": "Negative to Negative",
               "0 -> 1": "Negative to Positive",
               "1 -> 0": "Positive to Negative"}
    n_of_changes_df['label_change'] = n_of_changes_df['label_change'].map(
        mapping)

    n_of_changes_df.to_excel("data/tables/s4_table.xlsx", index=False)


if __name__ == "__main__":
    main()
