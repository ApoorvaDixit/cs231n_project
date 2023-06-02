from torch.utils.data import Dataset, DataLoader
import sys
import os
import torch
from torch import optim
from time import time
from tqdm import tqdm 

sys.path.append("data/")

import dataset_utils


dataset = dataset_utils.TripletDataset()
dataloader = DataLoader(dataset, batch_size=100)
n_train, n_batches = len(dataloader.dataset), len(dataloader)

for (iterator) in enumerate(dataloader):
    print(idx)
# needs a dataloader