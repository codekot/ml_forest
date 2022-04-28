import pandas as pd
from pandas_profiling import ProfileReport

def eda():
    df = pd.DataFrame([[1,2,3], ["red", "green", "blue"], [55,11,77],[1000,1200,1500]])
    profile = ProfileReport(df)
    profile.to_file("output.html")

