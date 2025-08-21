import numpy as np
from omegaconf import DictConfig


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean: np.array, std: np.array):
        self.mean = mean
        self.std = std

    def transform(self, data: np.array):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.array):
        return (data * self.std) + self.mean


def reduce_sample_size(config: DictConfig, *args):
    sz = args[0].shape[0]
    used_sz = int(sz * config.datasz.percentage)
    return [d[:used_sz] for d in args]


def cut_timeseries(ts, labels, sites, groups, cut):
    new_ts, new_labels, new_sites, new_groups = [], [], [], []
    for en, i in enumerate(ts):
        if len(i) >= cut:
            new_ts.append(i[:cut])
            new_labels.append(labels[en])
            new_sites.append(sites[en])
            if len(groups) > 0:
                new_groups.append(groups[en])

    new_ts, new_labels, new_sites, new_groups = np.array(new_ts), np.array(new_labels), np.array(new_sites), np.array(new_groups)
    return new_ts, new_labels, new_sites, new_groups
