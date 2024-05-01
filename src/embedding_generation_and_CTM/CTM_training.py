import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from contextualized_topic_models.models.ctm import ZeroShotTM
from src.embedding_generation_and_CTM.CTM_preprocessing import load_h5file

def load_training_dataset_and_qt(data_saving_path):
    with open(f'{data_saving_path}/training_dataset.pkl', 'rb') as f:
        training_dataset = pickle.load(f)

    with open(f'{data_saving_path}/qt.pkl', 'rb') as f:
        qt = pickle.load(f)

    return training_dataset, qt

def load_preprocess_results(data_saving_path):
    path_ = f"{data_saving_path}/sp_preprocess_results.parquet"
    sp_preprocess_results = pd.read_parquet(path_)
    retained_indices = sp_preprocess_results['retained_indices'].values
    unpreprocessed_documents = sp_preprocess_results['text'].values

    del sp_preprocess_results

    return retained_indices, unpreprocessed_documents

def get_param_grid():
    param_grid = {'k_numbers':list(range(5, 11)),
              'hidden_dimensions': [(200, 200), (500, 500), (700, 700)],
              'dropout': [0.2, 0.5, 0.8]}
    return param_grid

def get_grid_search_params():
    param_grid = get_param_grid()

    grid_search = list(ParameterGrid(param_grid))
    return grid_search

def start_training(grid_search, training_dataset, embeddings,
                   model_saving_path, topics_saving_path, qt):
    for grid in tqdm(grid_search):
        model_name = f"model_{grid['k_numbers']}_" \
            f"{grid['hidden_dimensions'][0]}_{grid['dropout']}.pth"

        model_path = model_saving_path + "/" + model_name
        if os.path.exists(model_path):
            continue

        ctm = ZeroShotTM(bow_size=len(qt.vocab), contextual_size=embeddings.shape[1],
                        n_components=grid['k_numbers'], num_epochs=2,
                        hidden_sizes=grid['hidden_dimensions'],
                        dropout=grid['dropout'], shuffle=False)
        ctm.fit(training_dataset, n_samples=None)

        topic_list = ctm.get_topic_lists(50)

        columns = [f"topic_{i+1}" for i in range(grid['k_numbers'])]
        topic_list = pd.DataFrame(np.array(topic_list).T, columns=columns)

        saving_ = f"{topics_saving_path}/topics_{grid['k_numbers']}" \
            f"_{grid['hidden_dimensions'][0]}_{grid['dropout']}.csv"
        topic_list.to_csv(saving_, index=False)

        ctm.train_data = None
        ctm.save(model_path)

def main():
    topics_saving_path = "data/processed/CTM/topics"
    model_saving_path = 'models/ctm_models'
    data_saving_path = "data/processed/CTM"
    os.makedirs(topics_saving_path, exist_ok=True)
    os.makedirs(model_saving_path, exist_ok=True)

    embedding_h5file = load_h5file()
    embeddings = embedding_h5file['embeddings']

    training_dataset, qt = load_training_dataset_and_qt(data_saving_path)
    training_dataset.X_contextual = embeddings
    
    grid_search = get_grid_search_params()
    start_training(grid_search, training_dataset, embeddings,
                   model_saving_path, topics_saving_path, qt)
    

if __name__ == "__main__":
    main()