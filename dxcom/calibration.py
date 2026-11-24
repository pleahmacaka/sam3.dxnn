import os

import numpy as np
import torch
from torch.utils.data import Dataset


class Tensors(Dataset):

    def __init__(self):
        self.samples = np.load(os.environ["SAM3_CALIB"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return torch.from_numpy(self.samples[index])
