import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import click
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_validate

'''Your script should be runnable from the terminal, 
receive some arguments such as the 
path to data, 
model configurations, etc.'''

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
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

def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    # test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    solver: str
):
    target_column = "Cover_Type"
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    if use_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    model = LogisticRegression(solver=solver,
                               random_state=random_state,
                               max_iter=max_iter,
                               C=logreg_c).fit(X, y)
    joblib.dump(model, save_model_path)
    print("Train script finished")

def cross_valid():
    print("Cross validation started")
    target_column = "Cover_Type"
    dataset_path = "data/train.csv"
    dataset = pd.read_csv(dataset_path)
    X = dataset.drop(target_column, axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = dataset[target_column]
    result = cross_validate(LogisticRegression(),
                   X, y, cv=7,
                   return_train_score=True,
                   scoring=['accuracy', 'neg_mean_squared_error'])
    print(result)
    print("Cross validation finished")


