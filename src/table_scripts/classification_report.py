import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from src.sentiment_analysis.bert_model_fine_tuning import load_model
from src.sentiment_analysis.data_preprocessing import get_train_val_test_dfs


def report_to_df(report):
    report_df = pd.DataFrame({'precision': [report['0']['precision'],
                                            report['1']['precision'],
                                            report['2']['precision'],
                                            '', '', report['macro avg']['precision'],
                                            report['weighted avg']['precision']],
                              'recall': [report['0']['recall'],
                                         report['1']['recall'],
                                         report['2']['recall'],
                                         '', '', report['macro avg']['recall'],
                                         report['weighted avg']['recall']],
                              'f1-score': [report['0']['f1-score'],
                                           report['1']['f1-score'],
                                           report['2']['f1-score'],
                                           '', report['accuracy'],
                                           report['macro avg']['f1-score'],
                                           report['weighted avg']['f1-score']],
                              'support': [report['0']['support'],
                                          report['1']['support'],
                                          report['2']['support'],
                                          '', report['macro avg']['support'],
                                          report['macro avg']['support'],
                                          report['macro avg']['support']]},
                             index=["Neutral", "Positive", "Negative", '',
                                    'accuracy', 'macro avg', 'weighted avg'],
                             dtype='str')
    return report_df


def main():
    tables_path = "data/tables"
    model_dir = "models/sentiment_models"
    model = load_model(model_dir, 'best_model')
    df_train, df_test, df_val = get_train_val_test_dfs()
    test_labels = df_test['labels'].values.astype(int)

    result, model_outputs, wrong_predictions = model.eval_model(df_test)
    y_pred = np.argmax(model_outputs, axis=1)

    report = classification_report(test_labels, y_pred, output_dict=True)
    report_df = report_to_df(report)

    report_df.to_excel(f"{tables_path}/s3_table.xlsx", index=False)


if __name__ == '__main__':
    main()
