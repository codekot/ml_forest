import pandas as pd
from pandas_profiling import ProfileReport

def eda():
    df = pd.read_csv(r"D:\CODE\RSML\ml_forest\data\train.csv")
    profile = ProfileReport(df)
    profile.to_file("output.html")

