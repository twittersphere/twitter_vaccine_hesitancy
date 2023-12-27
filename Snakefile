include: "src/sentiment_analysis/Snakefile"
include: "src/table_scripts/Snakefile"
include: "src/df_scripts/Snakefile"
include: "src/fetching_data/Snakefile"
include: "src/figures/Snakefile"

rule all:
    input:
      rules.eda_of_fetched_data.output,
      rules.fetch_crowdbreaks_data.output,
      rules.fetch_country_tweet_counts.output,
      rules.fetch_vaccine_vaccination_tweets.output,
      rules.fetch_geo_data.output,
      rules.fetched_data_barplot.output,
      rules.bert_model_fine_tuning.output,
      rules.tweet_sentiment_predictions.output,
      rules.country_tweet_counts_to_table.output,
    
