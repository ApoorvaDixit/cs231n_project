import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter
import torch
import glob
import os
import re
import numpy as np


def read_metadata(type = 'train', # can be one of ['train', 'test', 'all' ]
                  data_dir = '/home/ubuntu/cs231n_project/cs231n_project/land_cover_representation/'):
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
              data_dir = '/home/ubuntu/cs231n_project/cs231n_project/land_cover_representation/'):
    metadata = read_metadata(type, data_dir)
    filenames = metadata['file_name']
    data = []

    for file_name in filenames:
        loaded_data = np.load(os.path.join(data_dir,file_name))
        data.append(loaded_data)

    return data

class TripletDataset(Dataset):
    def __init__(self, type = 'train', data_dir='/home/ubuntu/cs231n_project/cs231n_project/land_cover_representation/'):
        self.tile_dir = data_dir
        metadata = pd.read_csv(os.path.join(data_dir,'metadata.csv'))
        metadata['triplet_id'] = metadata['file_name'].apply(lambda x: int(re.findall(r'\d+', x)[0]) )
        tilenet_metadata = metadata[metadata['split_str']!='test']
        train_triplet_ids = tilenet_metadata['triplet_id'].tolist()
        cnt = Counter(train_triplet_ids)
        self.triplet_ids = [k for k, v in cnt.items() if v == 3]
        print(len(self.triplet_ids))
        
        
    def __len__(self):
        return len(self.triplet_ids)
    
    def __getitem__(self, idx):
        if idx not in self.triplet_ids:
            print("oops")
            return None
        a = np.load(os.path.join(self.tile_dir+'tiles/', '{}anchor.npy'.format(idx)))
        n = np.load(os.path.join(self.tile_dir+'tiles/', '{}neighbor.npy'.format(idx)))
        d = np.load(os.path.join(self.tile_dir+'tiles/', '{}distant.npy'.format(idx)))
        a = np.moveaxis(a, -1, 0)
        n = np.moveaxis(n, -1, 0)
        d = np.moveaxis(d, -1, 0)
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample