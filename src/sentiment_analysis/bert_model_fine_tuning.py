import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.utils import class_weight
from sklearn.model_selection import ParameterGrid
from simpletransformers.classification import ClassificationModel
from src.sentiment_analysis.data_preprocessing import get_train_val_test_dfs
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from src.scripts.utils import save_log

def metric_scores(y_true, y_pred):
    f1_macro = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    return f1_macro, accuracy, balanced_accuracy

def create_auto_model(model_name, model_args, class_weight):
    model = ClassificationModel(
                "bert",
                model_name,
                use_cuda=True,
                num_labels=3,
                weight=class_weight,
                args=model_args,
                cuda_device=0
            )
    return model

def generate_grid_search(weights):
    hyper_parameters = {"learning_rate":[1e-3, 1e-4, 1e-5],
                        "adam_epsilon": [1e-7, 1e-8, 1e-9],
                        "weight":[weights.tolist(), (weights**2).tolist()],
                        "max_seq_length": [128, 64],
                        "weight_decay":[0, 0.01, 0.0001]}
    
    grid_search = list(ParameterGrid(hyper_parameters))

    return grid_search

def save_best_model(model, y_pred, output_dir, model_name):
    with open(f"{output_dir}/{model_name}.db", 'wb') as f:
        pickle.dump(model, f)

    np.save(f"{output_dir}/preds/{model_name}_ypreds.npy", y_pred)

def load_model(output_dir, model_name):
    with open(f"{output_dir}/{model_name}.db", 'rb') as f:
        model = pickle.load(f)

    return model

def train_model(model_name, model_args, grid_search, df_train, df_test,
                df_val, output_dir):
    best_scores = [0, 0, 0]
    all_best = [0, 0, 0]

    for value in tqdm(grid_search):
        for k, v in value.items():
            if k != "weight":
                model_args[k] = v

        model = create_auto_model(model_name, model_args, value['weight'])
        model.train_model(df_train, eval_df=df_val, acc=accuracy_score)

        result, model_outputs, wrong_predictions = model.eval_model(df_test)
        y_pred = np.argmax(model_outputs, axis=1)

        scores = metric_scores(df_test['labels'].values.astype(int), y_pred)
        
        if scores[0]>all_best[0] and scores[1]>all_best[1] and scores[2]>all_best[2]:
            all_best[0] = scores[0]
            all_best[1] = scores[1]
            all_best[2] = scores[2]
            save_best_model(model, y_pred, output_dir, "best_model")

        if scores[0] > best_scores[0]:
            best_scores[0] = scores[0]
            save_best_model(model, y_pred, output_dir, "best_f1")

        if scores[1] > best_scores[1]:
            best_scores[1] = scores[1]
            save_best_model(model, y_pred, output_dir, "best_acc")

        if scores[2] > best_scores[2]:
            best_scores[2] = scores[2]
            save_best_model(model, y_pred, output_dir, "best_balanced_acc")

def main():
    model_args = {
        "use_early_stopping": True,
        "early_stopping_patience": 5,
        "fp16": True,
        "num_train_epochs": 20,
        'overwrite_output_dir': True,
        'learning_rate': 1e-5,
        "save_steps": -1,
        "evaluate_during_training": True,
        "early_stopping_consider_epochs": True,
    }

    output_dir = "models/sentiment_models"
    os.makedirs(output_dir + '/preds', exist_ok=True)

    df_train, df_test, df_val = get_train_val_test_dfs()

    labels = df_train['labels']
    weights = class_weight.compute_class_weight('balanced',
                                                classes=labels.unique().values,
                                                y=labels.values)
    del labels

    grid_search = generate_grid_search(weights)

    model_name = "digitalepidemiologylab/covid-twitter-bert-v2"
    train_model(model_name, model_args, grid_search, df_train, df_test,
                df_val, output_dir)
    
    save_log("bert_model_fine_tuning")
    

    
if __name__ == "__main__":
    main()
    