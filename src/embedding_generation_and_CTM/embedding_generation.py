import os
import h5py
import pandas as pd
from tqdm import tqdm
from src.scripts.tools import Tools
from src.scripts.utils import save_log
from src.scripts.read_data import ReadData
from sentence_transformers import SentenceTransformer


def read_data_and_filter(world_data_path, dataframes_path):
    world_anti_ids = pd.read_parquet(f"{dataframes_path}/world_anti_ids.parquet")
    anti_ids = set(world_anti_ids['id'].values)

    read_data_world = ReadData(world_data_path, ['id', 'text'],
                               filter_tweets=True,
                               custom_filter="df['id'].isin(self.anti_ids)")
    read_data_world.anti_ids = anti_ids
    read_data_world.read_csvs_and_combine_data()
    world_data = read_data_world.data

    text = world_data['text'].values

    return text

def generate_embeddings(tools, text, model, h5file_path):
    if not os.path.exists(h5file_path):
        embedding_h5file = h5py.File(h5file_path, "w")
        dset = embedding_h5file.create_dataset("embeddings",
                                               (text.shape[0], 1024),
                                               chunks=(64, 1024))
        dset.attrs['length'] = 0
    else:
        embedding_h5file = h5py.File(h5file_path, "r+")
        dset = embedding_h5file['embeddings']

    batch_size = 4096
    batches = list(tools.create_chunks(text, 4096))
    for idx, batch in enumerate(tqdm(batches)):

        if idx*batch_size < dset.attrs['length']:
            continue

        word_vectors = model.encode(batch, batch_size=1024)

        dset[dset.attrs['length']:dset.attrs['length']+word_vectors.shape[0]] = word_vectors
        dset.attrs['length'] += word_vectors.shape[0]

    embedding_h5file.close()

def main():
    tools = Tools()

    world_data_path = "data/raw/daily_data_parquet"
    dataframes_path = "data/processed/data_frames"
    text = read_data_and_filter(world_data_path, dataframes_path)
    
    model = SentenceTransformer("digitalepidemiologylab/covid-twitter-bert-v2",
                                device='xla')
    h5file_path = "data/processed/world_anti_embeddings.hdf5"
    generate_embeddings(tools, text, model, h5file_path)
    save_log("embedding_generation")
    

if __name__ == '__main__':
    main()