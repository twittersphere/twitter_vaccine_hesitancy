import pandas as pd


def main():
    df_path = "data/processed/data_frames"
    df = pd.read_parquet(
        f"{df_path}/fisher_exact/state_us_fisher_values.parquet")

    df.to_excel("data/tables/s5_table.xlsx", index=False)


if __name__ == "__main__":
    main()
