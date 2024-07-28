import pickle
import pandas as pd


def main():
    tables_path = "data/tables"
    dataframes_path = "data/processed/data_frames"
    with open(f"{dataframes_path}/precision_recall_threshold_curve_results.pkl", 'rb') as f:
        result_dict = pickle.load(f)

    precs_and_recs = result_dict['precs_and_recs']

    for idx, label in enumerate(['Neutral', "Positive", "Negative"]):
        precs_and_recs[idx].insert(0, label)

    df = pd.DataFrame(precs_and_recs, columns=["Label", "Precision", "Recall"])
    df.to_excel(f"{tables_path}/s2_table.xlsx", index=False)


if __name__ == '__main__':
    main()
