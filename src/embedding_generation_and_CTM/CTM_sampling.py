import os
import torch
import numpy as np
import pandas as pd
import pyLDAvis as vis
from tqdm import tqdm
from src.scripts.utils import save_log
from contextualized_topic_models.models.ctm import ZeroShotTM
from src.embedding_generation_and_CTM.CTM_preprocessing import load_h5file
from src.embedding_generation_and_CTM.CTM_training import load_preprocess_results, load_training_dataset_and_qt, get_grid_search_params

def init_zero_shot_tm(bow_size, contextual_size, k_numbers,
                      hidden_dimensions, dropout, model_path, training_dataset):
    ctm = ZeroShotTM(bow_size=bow_size, contextual_size=contextual_size,
                    n_components=k_numbers, num_epochs=2,
                    hidden_sizes=hidden_dimensions, dropout=dropout,
                    shuffle=False)
    ctm.load(model_path)
    ctm.train_data = training_dataset
    ctm.USE_CUDA = False
    ctm.device = torch.device('cpu')
    ctm.batch_size = 1024

    return ctm

def get_model_name_path(grid, model_saving_path):
    model_name = f"model_{grid['k_numbers']}_{grid['hidden_dimensions'][0]}_" \
                f"{grid['dropout']}.pth"
    model_path = model_saving_path + "/" + model_name

    return model_name, model_path

def save_topics_predictions(topics_predictions, grid, topics_predictions_path):
    columns = [f"topic_{i+1}" for i in range(grid['k_numbers'])]
    topics_predictions = pd.DataFrame(topics_predictions, columns=columns)
    topics_predictions.to_parquet(topics_predictions_path, index=False)
    
def main():
    probs_saving_path = "data/processed/CTM/probs"
    plot_saving_path = "data/processed/CTM/ldavis_figures"
    model_saving_path = 'models/ctm_models'
    data_saving_path = "data/processed/CTM"

    embedding_h5file = load_h5file()
    embeddings = embedding_h5file['embeddings']

    training_dataset, qt = load_training_dataset_and_qt(data_saving_path)
    _, unpreprocessed_documents = load_preprocess_results(data_saving_path)
    del _

    grid_search = get_grid_search_params()

    for grid in tqdm(grid_search):
        model_name, model_path = get_model_name_path(grid, model_saving_path)
        topics_predictions_path = f"{probs_saving_path}/probs_{grid['k_numbers']}_" \
                            f"{grid['hidden_dimensions'][0]}_{grid['dropout']}.parquet"
        if os.path.exists(topics_predictions_path):
            continue

        if not os.path.exists(model_path):
            print(f"{model_name} doesn't exists!")
            continue

        # Initialize ZeroShotTM model and get topics predictions
        ctm = init_zero_shot_tm(len(qt.vocab), embeddings.shape[1],
                                grid['k_numbers'], grid['hidden_dimensions'],
                                grid['dropout'], model_path, training_dataset)

        topics_predictions = ctm.get_thetas(training_dataset)
        save_topics_predictions(topics_predictions, grid, topics_predictions_path)

        # Get top 100 tweets for each topic
        tops = []
        for topic in topics_predictions.columns:
            tops.append(np.argsort(topics_predictions[topic].values)[::-1][:100])

        # Save top tweets
        top_tweets_saving_path = f"{data_saving_path}/top_tweets/{grid['k_numbers']}" \
                                f"_{grid['hidden_dimensions'][0]}_{grid['dropout']}"
        os.makedirs(top_tweets_saving_path, exist_ok=True)

        # Save top 100 tweets for each topic
        for idx, i in enumerate(tops):
            top_100_tweets = pd.DataFrame({'text': unpreprocessed_documents[i]})
            top_100_tweets.to_csv(f"{top_tweets_saving_path}/topic_{idx+1}.csv", index=False)

        # LDA visualization
        lda_vis_data = ctm.get_ldavis_data_format(qt.vocab, training_dataset, 20, topics_predictions.values)
        ctm_pd = vis.prepare(sort_topics=False, **lda_vis_data)
        vis.save_html(ctm_pd, f"{plot_saving_path}/lda_vis_figure_{grid['k_numbers']}_{grid['hidden_dimensions'][0]}_{grid['dropout']}.html")

        save_log("CTM_sampling")

if __name__ == "__main__":
    main()