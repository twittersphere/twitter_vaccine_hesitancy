import numpy as np
import pandas as pd


def filter_users(authorid_date_topic_count):
    filter_ = authorid_date_topic_count.iloc[:, 1:].values.sum(axis=1) > 1
    filtered_df = authorid_date_topic_count[filter_].reset_index(drop=True)

    topic_count_df = []

    for clm in filtered_df.columns[1:]:
        a_tpc_count = filtered_df[filtered_df[clm].values > 0]
        a_tpc_count[clm] = a_tpc_count[clm].values - 1

        sub_topic_count_df = []
        for clm2 in a_tpc_count.columns[1:]:
            count = a_tpc_count[a_tpc_count[clm2] > 0].shape[0]
            sub_topic_count_df.append([clm, clm2, count])

        sub_topic_count_df = pd.DataFrame(sub_topic_count_df, columns=[
                                          "topic", "other_topic", "count"])
        topic_count_df.append(sub_topic_count_df)

    topic_count_df = pd.concat(topic_count_df, axis=0).reset_index(drop=True)

    return topic_count_df


def topic_percentage(topic_count_df):
    topic_total = topic_count_df.groupby('topic').agg(
        {'count': 'sum'}).reset_index().rename(
        columns={'count': 'total'})
    topic_count_df = topic_count_df.join(
        topic_total.set_index('topic'), on='topic')
    topic_count_df['perc'] = topic_count_df['count'] / topic_count_df['total']

    links = topic_count_df[['topic', 'other_topic', 'count']].astype(int)
    links.columns = ['source', 'target', 'value']

    return links


def prepare_links(links):
    exists = set()
    prepared_links = []
    for idx, row in links.iterrows():
        srtd_ = np.sort(row.iloc[:2].values).astype(str).tolist()
        srtd_ = ":".join(srtd_)
        if srtd_ in exists:
            continue
        exists.add(srtd_)
        if row['source'] == row['target']:
            prepared_links.append(
                [row['source'],
                 row['target'],
                 row['value'] // 2])
        else:
            prepared_links.append([row['source'], row['target'], row['value']])

    prepared_links = pd.DataFrame(prepared_links, columns=[
                                  'source', 'target', 'value'], dtype=int)

    return prepared_links


def main():
    dataframes_path = "data/processed/data_frames"

    authorid_date_topic_count = pd.read_parquet(
        f"{dataframes_path}/authorid_date_topic_count_8_200_0.2.parquet")
    authorid_date_topic_count['topic'] = authorid_date_topic_count['topic'].astype(
        'str')

    authorid_date_topic_count = authorid_date_topic_count.drop(
        columns=['created_at']).groupby(
        ['author_id', 'topic']).agg(
        {'count': 'sum'}).reset_index()

    authorid_date_topic_count = authorid_date_topic_count.pivot_table(
        index='author_id', columns='topic', values='count', dropna=False,
        fill_value=0).reset_index()

    # Filtering Users > 1
    topic_count_df = filter_users(authorid_date_topic_count)

    # Topic Percentage
    links = topic_percentage(topic_count_df)

    # Prepare the links dataframe
    prepared_links = prepare_links(links)

    # Save the prepared_links dataframe
    prepared_links.to_csv(f"{dataframes_path}/prepared_links.csv", index=False)


if __name__ == '__main__':
    main()
