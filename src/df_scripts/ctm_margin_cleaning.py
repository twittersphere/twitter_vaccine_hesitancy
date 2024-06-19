import numpy as np
import pandas as pd
from src.embedding_generation_and_CTM.CTM_training import load_preprocess_results


def apply_margin(topics_predictions_df, margin=1/8):
    sorted_preds = np.sort(topics_predictions_df, axis=1)
    greater_than_margin = (sorted_preds[:, -1] - sorted_preds[:, -2]) > margin
    new_topics_predictions_df = topics_predictions_df.loc[greater_than_margin, :]

    return new_topics_predictions_df


def main():
    ctm_data_path = "data/processed/CTM"
    probs_saving_path = "data/processed/CTM/probs"
    dataframes_path = "data/processed/data_frames"

    probs_path = f"{probs_saving_path}/probs_8_200_0.2.parquet"
    topics_predictions_df = pd.read_parquet(probs_path)

    new_topics_predictions_df = apply_margin(topics_predictions_df)
    cleaned_probs_path = f"{probs_saving_path}/cleaned_probs_8_200_0.2.parquet"
    new_topics_predictions_df.to_parquet(cleaned_probs_path, index=False)

    # Cleaning world anti tweets
    retained_indices, _ = load_preprocess_results(ctm_data_path)
    del _

    world_anti_ids = pd.read_parquet(
        f"{dataframes_path}/world_anti_ids.parquet")
    cleaned_world_anti_ids = world_anti_ids.iloc[retained_indices, :]
    cleaned_world_anti_ids = cleaned_world_anti_ids.reset_index(drop=True)

    cleaned_indices = np.array(new_topics_predictions_df.index)
    cleaned_world_anti_ids = cleaned_world_anti_ids[cleaned_indices]
    cleaned_ids_path = f"{dataframes_path}/cleaned_world_anti_ids.parquet"
    cleaned_world_anti_ids.to_parquet(cleaned_ids_path, index=False)


if __name__ == '__main__':
    main()
