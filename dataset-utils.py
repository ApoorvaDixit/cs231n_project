import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import glob
import os
import numpy as np
from src.data_utils import clip_and_scale_image


def read_metadata(type = 'train', # can be one of ['train', 'test', 'all' ]
                  data_dir = '/home/ubuntu/cs231n_project/land_cover_representation/'):
    metadata = pd.read_csv(os.path.join(data_dir,'metadata.csv'))
    if type == 'all':
        return metadata
    metadata['triplet_id'] = metadata['file_name'].apply(lambda x: int(re.findall(r'\d+', x)[0]) )
    test_metadata = metadata[metadata['split_str']=='test'] 
    if type == 'test':
        return test_metadata
    test_triplet_ids = test_metadata['triplet_id'].tolist()

    # Everything other than test (both train and val) is included in training data for us
    train_metadata = metadata[~metadata['triplet_id'].isin(test_triplet_ids)]
    return train_metadata

def get_tiles(type='train', # can be one of ['train', 'test', 'all' ]
              data_dir = '/home/ubuntu/cs231n_project/land_cover_representation/'):
    metadata = read_metadata(type, data_dir)
    filenames = metadata['file_name']
    data = []

    for file_name in filenames:
        loaded_data = np.load(os.path.join(data_dir,file_name))
        data.append(loaded_data)

    return data

