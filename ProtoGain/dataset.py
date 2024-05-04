import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Data:
    def __init__(self, dataset, hint_rate):

        mask = np.where(np.isnan(dataset), 0.0, 1.0)
        dataset = np.where(mask, dataset, 0.0)
        hint = generate_hint(mask, hint_rate)

        range_scaler = (0, 1)

        self.scaler = MinMaxScaler(feature_range=range_scaler)
        dataset_scaled = self.scaler.fit_transform(dataset)

        self.dataset = torch.from_numpy(dataset)
        self.mask = torch.from_numpy(mask)
        self.hint = torch.from_numpy(hint)
        self.dataset_scaled = torch.from_numpy(dataset_scaled)


def generate_hint(mask, hint_rate):
    hint_mask = generate_mask(mask, 1 - hint_rate)
    hint = mask * hint_mask

    return hint


def generate_mask(data, miss_rate):
    dim = data.shape[1]
    size = data.shape[0]
    A = np.random.uniform(0.0, 1.0, size=(size, dim))
    B = A > miss_rate
    mask = 1.0 * B

    return mask
