import pandas as pd


def main():
    df_path = "data/processed/data_frames"

    coeff = pd.read_parquet(
        f"{df_path}/socio_economic_params_pca_coefficients.parquet")

    coeff.to_excel("data/tables/s6_table.xlsx", index=False)


if __name__ == "__main__":
    main()
