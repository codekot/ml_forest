import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import click
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_validate
import config

'''Your script should be runnable from the terminal, 
receive some arguments such as the 
path to data, 
model configurations, etc.'''

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default=config.DATA_PATH,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default=config.MODEL_SAVE_PATH,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
# @click.option(
#     "--test-split-ratio",
#     default=0.2,
#     type=click.FloatRange(0, 1, min_open=True, max_open=True),
#     show_default=True,
# )
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--solver",
    default='lbfgs',
    type=str,
    show_default=True
)
@click.option(
    "-f",
    "--fold",
    default=0,
    type=int,
    show_default=True
)

def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    # test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    solver: str,
    fold: int
):
    target_column = "Cover_Type"
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    if use_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    if fold > 0:
        cross_valid(X, y, cv=fold)
    else:
        model = LogisticRegression(solver=solver,
                                   random_state=random_state,
                                   max_iter=max_iter,
                                   C=logreg_c).fit(X, y)
        joblib.dump(model, save_model_path)
        print("Train script finished")

def cross_valid(X, y, cv):
    result = cross_validate(LogisticRegression(),
                   X, y, cv=cv,
                   return_train_score=True,
                   scoring=['accuracy', 'f1_weighted', 'roc_auc_ovr_weighted'])
    print('Accuracy, train data: ', result['train_accuracy'])
    print('Accuracy, test data: ', result['test_accuracy'])
    print('F1 score, train data: ', result['train_f1_weighted'])
    print('F1 scorem test data: ', result['test_f1_weighted'])
    print('ROC AUC, train data: ', result['train_roc_auc_ovr_weighted'])
    print('ROC AUC, test data: ', result['test_roc_auc_ovr_weighted'])
    print("Cross validation finished")


