from torch.utils.data import Dataset, DataLoader
import sys
import os
print(os.getcwd())

import torch
from torch import optim
import time
from tqdm import tqdm 

from data.dataset_utils import TripletDataLoader
from model_architectures.transformer.transformer import make_ViT
from training import train_triplet_epoch

img_type = 'naip'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cuda = torch.cuda.is_available()
image_size = 100
z_dim = 512
vit = make_ViT(image_size = image_size, z_dim=z_dim)
if cuda: vit.cuda()
vit.train()

print('Transformer set up complete.')

lr = 1e-4
optimizer = optim.Adam(vit.parameters(), lr=lr, betas=(0.5, 0.999))

print('Optimizer set up complete.')

# increase batch size to address nan loss value
dataloader = TripletDataLoader(img_type, batch_size=16)

# Original
# dataloader = TripletDataLoader(img_type, batch_size=64)

# Sanity check
# dataloader = TripletDataLoader(img_type, batch_size=1)

print('Dataset set up.')

epochs = 10
margin = 10
l2 = 0.01
print_every = 10000

model_dir = 'models/'
if not os.path.exists(model_dir): os.makedirs(model_dir)

t0 = time.time()
time_struct = time.localtime(t0)
formatted_date = time.strftime("%I:%M %p, %B %d, %Y", time_struct)  # Example: June 02, 2023

print('Begin training.................')
for epoch in tqdm(range(0, epochs), desc="epoch loop"):
    (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
        vit, cuda, dataloader, optimizer, epoch+1, margin=margin, l2=l2,
        print_every=print_every, t0=t0)

model_fn = os.path.join(model_dir, 'ViT_epoch{}.ckpt'.format(epochs))
torch.save(net.state_dict(), model_fn)