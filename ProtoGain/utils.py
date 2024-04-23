import torch
import numpy as np
import pandas as pd
import os


def create_csv(data, name: str, header):
    df = pd.DataFrame(data)
    df.to_csv(name + ".csv", index=False, header=header)


def create_dist(size: int, dim: int, name: str):

    X = torch.normal(0.0, 1, (size, dim))
    A = torch.tensor([[1, 2], [-0.1, 0.5]])
    b = torch.tensor([0, 0])
    data = torch.matmul(X, A) + b

    create_csv(data, name)


def create_missing(data, miss_rate: float, name: str, header):

    size = data.shape[0]
    dim = data.shape[1]

    mask = torch.zeros(data.shape)

    for i in range(dim):

        chance = torch.rand(size)
        miss = chance > miss_rate
        mask[:, i] = miss

        missing_data = np.where(mask < 1, np.nan, data)

    name = name + "_{}".format(int(miss_rate * 100))

    create_csv(missing_data, name, header)


def create_output(data, path: str, override: int):

    if override == 1:
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    else:
        if os.path.exists(path):
            df = pd.read_csv(path)
            new_df = pd.DataFrame(data)
            df = pd.concat([df, new_df], axis=1)
            df.columns = range(len(df.columns))
            df.to_csv(path, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
