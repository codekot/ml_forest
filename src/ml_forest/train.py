import pandas as pd

def train():
    target_column = "Cover_Type"
    dataset = pd.read_csv(r"D:\CODE\RSML\ml_forest\data\train.csv")
    features = dataset.drop(target_column, axis=1)
    target = dataset[target_column]

