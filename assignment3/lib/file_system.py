import pandas as pd


def load_dataset(path):
    data_frame = pd.read_csv(path, header=None, names=None)
    return data_frame
