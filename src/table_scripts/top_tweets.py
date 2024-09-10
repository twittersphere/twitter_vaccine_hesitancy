import pandas as pd
from src.scripts.utils import load_main_config


def get_top_tweets():
    top_tweets_path = "data/processed/CTM/top_tweets/8_200_0.2"
    top_tweets = []

    for topic in range(1, 9):
        top_tweets_for_topic = pd.read_csv(
            f"{top_tweets_path}/topic_{topic}.csv").iloc[:10, :]
        top_tweets_for_topic.loc[:,
                                 'text'] = top_tweets_for_topic.loc[:, 'text']
        top_tweets.append(top_tweets_for_topic)

    return top_tweets


def main():
    main_configs = load_main_config()

    top_tweets = get_top_tweets()
    for idx in range(len(top_tweets)):
        top_tweets[idx].columns = [main_configs['topic_names'][idx]]

    combined_df = pd.concat(top_tweets, axis=1)
    combined_df = combined_df.melt(var_name='topic', value_name='tweet')
    combined_df.to_excel("data/tables/s7_table.xlsx", index=False)


if __name__ == "__main__":
    main()
