from torch.utils.data import Dataset, DataLoader
import sys
import os
import torch
from torch import optim
from time import time
from tqdm import tqdm 

from data.dataset_utils import TripletDataLoader
from model_architectures.resnets.tilenet_18 import make_tilenet
from training import train_triplet_epoch

img_type = 'naip'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cuda = torch.cuda.is_available()
in_channels = 4
z_dim = 512
TileNet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
if cuda: TileNet.cuda()
TileNet.train()

print('TileNet set up complete.')

lr = 1e-3
optimizer = optim.Adam(TileNet.parameters(), lr=lr, betas=(0.5, 0.999))

print('Optimizer set up complete.')

dataloader = TripletDataLoader(img_type, batch_size=64)

print('Dataset set up.')

epochs = 10
margin = 10
l2 = 0.01
print_every = 10000

model_dir = 'models/'
if not os.path.exists(model_dir): os.makedirs(model_dir)

t0 = time()

print('Begin training.................')
for epoch in tqdm(range(0, epochs), desc="epoch loop"):
    (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
        TileNet, cuda, dataloader, optimizer, epoch+1, margin=margin, l2=l2,
        print_every=print_every, t0=t0)
    
model_fn = os.path.join(model_dir, 'TileNet18_epoch{}.ckpt'.format(epochs))
torch.save(TileNet.state_dict(), model_fn)