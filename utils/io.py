import os
import pandas as pd


def save_dataframe_csv(df, path, name):
    df.to_csv(os.path.join(path, name), index=False)

def load_dataframe_csv(path, name):
    return pd.read_csv(os.path.join(path, name))