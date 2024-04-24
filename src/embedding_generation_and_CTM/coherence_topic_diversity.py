import pickle
import pandas as pd
from tqdm import tqdm
from contextualized_topic_models.evaluation.measures import CoherenceNPMI, TopicDiversity
from src.embedding_generation_and_CTM.CTM_training import get_grid_search_params


def load_text(data_saving_path):
    path_ = f"{data_saving_path}/preprocessed_documents.parquet"
    preprocessed_documents = pd.read_parquet(path_)
    texts = [i.split() for i in preprocessed_documents['docs'].values]

    del preprocessed_documents
    return texts

def calculate_coherence_and_diversity(topics_saving_path, texts):
    grid_search = get_grid_search_params()
    coherence, topic_diversity = [], []
    for grid in tqdm(grid_search):

        topic_path = f"{topics_saving_path}/topics_{grid['k_numbers']}_" \
            f"{grid['hidden_dimensions'][0]}_{grid['dropout']}_{grid['beta']}.csv"
        topic_list = pd.read_csv(topic_path).values.T

        npmi = CoherenceNPMI(texts=texts, topics=topic_list)
        coherence.append(npmi.score())

        td = TopicDiversity(topic_list)
        topic_diversity.append(td.score(topk=50))

    return coherence, topic_diversity

def save_coherence_and_diversity(coherence, topic_diversity, topics_saving_path):
    with open(f"{topics_saving_path}/coherence_topic_diversity.pkl", 'wb') as f:
        pickle.dump([coherence, topic_diversity], f)

def main():
    topics_saving_path = "/data/processed/CTM/topics"
    data_saving_path = "/data/processed/CTM"

    texts = load_text(data_saving_path)
    coherence, topic_diversity = calculate_coherence_and_diversity(topics_saving_path,
                                                                   texts)
    save_coherence_and_diversity(coherence, topic_diversity, topics_saving_path)

if __name__ == "__main__":
    main()