import os
import pandas as pd
import subprocess


def prepare_attitude_data(dataframes_path):
    subprocess.run(
        f"cp {dataframes_path}/world_data_sentiments_raw.parquet {dataframes_path}/zenodo/attitude_predictions.parquet",
        shell=True)


def prepare_ctm_predictions(dataframes_path, probs_path):
    ctm_probs = pd.read_parquet(
        f"{probs_path}/cleaned_probs_8_200_0.2.parquet")
    cleaned_world_anti_ids = pd.read_parquet(
        f"{dataframes_path}/cleaned_world_anti_ids.parquet")

    combined_data = pd.concat([cleaned_world_anti_ids, ctm_probs], axis=1)

    combined_data.to_parquet(
        f"{dataframes_path}/zenodo/ctm_predictions.parquet", index=False)


if __name__ == "__main__":
    dataframes_path = "data/processed/data_frames"
    os.makedirs(f"{dataframes_path}/zenodo", exist_ok=True)

    prepare_attitude_data(dataframes_path)

    probs_path = "data/processed/CTM/probs"
    prepare_ctm_predictions(dataframes_path, probs_path)
