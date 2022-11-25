from sklearn.model_selection import train_test_split
import pandas as pd


def split_df(df, split_size=255, seed=0):
    N = int(len(df) / split_size)
    frames = [df.iloc[i * split_size:(i + 1) * split_size].copy() for i in range(N + 1)]
    train, val_test = train_test_split(frames, test_size=0.4, random_state=seed)
    val, test = train_test_split(val_test, test_size=0.5, random_state=seed)
    train = pd.concat(train)
    val = pd.concat(val)
    test = pd.concat(test)
    return train, val, test
