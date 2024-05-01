import os
import sys
import h5py
import nltk
import pickle
import numpy as np
import pandas as pd
from src.embedding_generation_and_CTM.embedding_generation import read_data_and_filter
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords

def preprocess_data(text):
    stop_words = nltk.corpus.stopwords.words('english')
    sp = WhiteSpacePreprocessingStopwords(text.tolist(), stop_words, 10000)

    # returns preprocessed_documents, unpreprocessed_documents, vocab, retained_indices
    return sp.preprocess()

def save_preprocessed_documents(preprocessed_documents, data_saving_path):
    pd.DataFrame({'docs': np.array(preprocessed_documents)}).to_parquet(
        f"{data_saving_path}/preprocessed_documents.parquet", index=False)
    
def save_unpreprocessed_documents(unpreprocessed_documents, data_saving_path,
                                  retained_indices):
    retained_indices = np.array(retained_indices)

    unpreprocessed_documents = np.array(unpreprocessed_documents)

    unpreprocessed_documents = pd.DataFrame({'retained_indices': retained_indices,
                'text': unpreprocessed_documents})

    path_ = f"{data_saving_path}/sp_preprocess_results.parquet"
    unpreprocessed_documents.to_parquet(path_, index=False)

def save_training_dataset(training_dataset, qt, data_saving_path):
    with open(f'{data_saving_path}/training_dataset.pkl', 'wb') as f:
        pickle.dump(training_dataset, f)

    with open(f'{data_saving_path}/qt.pkl', 'wb') as f:
        pickle.dump(qt, f)

def load_h5file():
    h5file_path = "data/processed/world_anti_embeddings.hdf5"

    embedding_cache_size = sys.getsizeof(np.random.random(
        (64,1024)).astype(np.float32)) * 1.1
    cache_size = max(embedding_cache_size, 1024*1024)

    embedding_h5file = h5py.File(h5file_path, "r", rdcc_nbytes=cache_size)
    return embedding_h5file

def main():
    data_saving_path = "data/processed/CTM"
    os.makedirs(data_saving_path, exist_ok=True)

    world_data_path = "data/raw/daily_data_parquet"
    dataframes_path = "data/processed/data_frames"
    text = read_data_and_filter(world_data_path, dataframes_path)

    embedding_h5file = load_h5file()
    embeddings = embedding_h5file['embeddings']

    preprocessed_data = preprocess_data(text)
    save_preprocessed_documents(preprocessed_data[0], data_saving_path)
    save_unpreprocessed_documents(preprocessed_data[1], data_saving_path,
                                  preprocessed_data[3])
    
    qt = TopicModelDataPreparation(None, preprocessed_data[3],
                                   show_warning=True)
    training_dataset = qt.fit(text_for_contextual=preprocessed_data[1]['text'].values.tolist(),
                                    text_for_bow=preprocessed_data[0],
                                    custom_embeddings=embeddings)
    
    training_dataset.X_contextual = None
    save_training_dataset(training_dataset, qt, data_saving_path)

    embedding_h5file.close()

if __name__ == "__main__":
    main()