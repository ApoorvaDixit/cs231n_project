import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter
import torch
import glob
import os
import re
import numpy as np

### TRANSFORMS ###
def clip_and_scale_image(img, img_type='naip', clip_min=0, clip_max=10000):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    if img_type in ['naip', 'rgb']:
        return img / 255
    elif img_type == 'landsat':
        return np.clip(img, clip_min, clip_max) / (clip_max - clip_min)

class GetBands(object):
    """
    Gets the first X bands of the tile triplet.
    """
    def __init__(self, bands):
        assert bands >= 0, 'Must get at least 1 band'
        self.bands = bands

    def __call__(self, sample):
        a, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
        # Tiles are already in [c, w, h] order
        a, n, d = (a[:self.bands,:,:], n[:self.bands,:,:], d[:self.bands,:,:])
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

class RandomFlipAndRotate(object):
    """
    Does data augmentation during training by randomly flipping (horizontal
    and vertical) and randomly rotating (0, 90, 180, 270 degrees). Keep in mind
    that pytorch samples are CxWxH.
    """
    def __call__(self, sample):
        a, n, d = (sample['anchor'], sample['neighbor'], sample['distant'])
        # Randomly horizontal flip
        if np.random.rand() < 0.5: a = np.flip(a, axis=2).copy()
        if np.random.rand() < 0.5: n = np.flip(n, axis=2).copy()
        if np.random.rand() < 0.5: d = np.flip(d, axis=2).copy()
        # Randomly vertical flip
        if np.random.rand() < 0.5: a = np.flip(a, axis=1).copy()
        if np.random.rand() < 0.5: n = np.flip(n, axis=1).copy()
        if np.random.rand() < 0.5: d = np.flip(d, axis=1).copy()
        # Randomly rotate
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: a = np.rot90(a, k=rotations, axes=(1,2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: n = np.rot90(n, k=rotations, axes=(1,2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: d = np.rot90(d, k=rotations, axes=(1,2)).copy()
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

class ClipAndScale(object):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    def __init__(self, img_type):
        assert img_type in ['naip', 'rgb', 'landsat']
        self.img_type = img_type

    def __call__(self, sample):
        a, n, d = (clip_and_scale_image(sample['anchor'], self.img_type),
                   clip_and_scale_image(sample['neighbor'], self.img_type),
                   clip_and_scale_image(sample['distant'], self.img_type))
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, sample):
        a, n, d = (torch.from_numpy(sample['anchor']).float(),
            torch.from_numpy(sample['neighbor']).float(),
            torch.from_numpy(sample['distant']).float())
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        return sample

### TRANSFORMS ###

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
    def __init__(self, type = 'train', data_dir='/home/ubuntu/cs231n_project/cs231n_project/land_cover_representation/', transform=None):
        self.transform = transform
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
        idx = self.triplet_ids[idx]
        a = np.load(os.path.join(self.tile_dir+'tiles/', '{}anchor.npy'.format(idx)))
        n = np.load(os.path.join(self.tile_dir+'tiles/', '{}neighbor.npy'.format(idx)))
        d = np.load(os.path.join(self.tile_dir+'tiles/', '{}distant.npy'.format(idx)))
        a = np.moveaxis(a, -1, 0)
        n = np.moveaxis(n, -1, 0)
        d = np.moveaxis(d, -1, 0)
        sample = {'anchor': a, 'neighbor': n, 'distant': d}
        sample = self.transform(sample)
        return sample
    
def TripletDataLoader(img_type, bands=4, batch_size=4, shuffle=True, augment=True, num_workers=4):
    transform_list = []
    if img_type in ['landsat', 'naip']: transform_list.append(GetBands(bands))
    transform_list.append(ClipAndScale(img_type))
    if augment: transform_list.append(RandomFlipAndRotate())
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    
    dataset = TripletDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader

class TileDataset():
    def __init__(self, type = 'test', data_dir='/home/ubuntu/cs231n_project/cs231n_project/land_cover_representation/', transform=None, num_tiles_requested = -1):
        # tuples of (filename, label)
        self.num_tiles_requested = num_tiles_requested
        self.tile_dir = data_dir
        metadata = pd.read_csv(os.path.join(data_dir,'metadata.csv'))
        tilenet_metadata = metadata[metadata['split_str']=='test']
        self.test_files = tilenet_metadata['file_name'].tolist()
        self.test_file_labels = tilenet_metadata['y'].tolist()
        
        
    def __len__(self):
        if self.num_tiles_requested != -1:
            return self.num_tiles_requested
        else:
            return len(self.test_files)
    
    def __getitem__(self, idx):
        filename = self.test_files[idx]
        label = self.test_file_labels[idx]
        tile = np.load(os.path.join(self.tile_dir, filename))
        # Get first 4 NAIP channels (5th is CDL mask)
        tile = tile[:,:,:4]
        # Rearrange to PyTorch order
        tile = np.moveaxis(tile, -1, 0)
        tile = np.expand_dims(tile, axis=0)
        # Scale to [0, 1]
        tile = tile / 255
        # Embed tile
        tile = torch.from_numpy(tile).float()
        sample = {'tile': tile, 'label': label}
        return sample
        
    

def TilesClassificationDataLoader(batch_size=4, num_workers=4, num_tiles_requested = -1):
    dataset = TileDataset(num_tiles_requested = num_tiles_requested)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader


def get_label(id):
    labels = pd.read_csv('src/data/labels.csv')
    return labels.iloc[id]['land_cover']

def get_label_id(label_str):
    labels = pd.read_csv('src/data/labels.csv')
    return labels[labels['land_cover']==label_str]['y'].values[0]