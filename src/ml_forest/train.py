import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train():
    target_column = "Cover_Type"
    dataset = pd.read_csv(r"D:\CODE\RSML\ml_forest\data\train.csv")
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    model = LogisticRegression(solver='saga', random_state=42).fit(X, y)
    filename = '.\data\model.sav'
    joblib.dump(model, filename)
    print("Train script finished")


