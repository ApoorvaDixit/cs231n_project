from torch.utils.data import Dataset, DataLoader
import sys
import os
import torch
from torch import optim
import time
from tqdm import tqdm 

from data.dataset_utils import TripletDataLoader
#from model_architectures.googlenet.googtilenet import make_googtilenet
from model_architectures.googlenet.googtilenet_v3 import make_googtilenet
from training import train_triplet_epoch

img_type = 'naip'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cuda = torch.cuda.is_available()
in_channels = 4
z_dim = 512
net = make_googtilenet(in_channels=in_channels, z_dim=z_dim)
if cuda: net.cuda()
net.train()

print('GoogTiLeNet set up complete.')

lr = 1e-3
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))

print('Optimizer set up complete.')

dataloader = TripletDataLoader(img_type, batch_size=64)

print('Dataset set up.')

epochs = 5
margin = 10
l2 = 0.01
print_every = 10000

model_dir = 'models/'
if not os.path.exists(model_dir): os.makedirs(model_dir)

t0 = time.time()
time_struct = time.localtime(t0)
formatted_datetime = time.strftime("%I:%M %p, %B %d, %Y", time_struct)  # Example: June 02, 2023

print(f'Begin training at {formatted_datetime}................')
for epoch in tqdm(range(0, epochs), desc="epoch loop"):
    (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
        net, cuda, dataloader, optimizer, epoch+1, margin=margin, l2=l2,
        print_every=print_every, t0=t0)
    if (epoch % 5) == 0 and epoch>0:
        model_fn = os.path.join(model_dir, 'GoogTiLeNet_v3_epoch{}.ckpt'.format(epoch))
        torch.save(net.state_dict(), model_fn)

time_struct = time.localtime(time.time())
formatted_datetime = time.strftime("%I:%M %p, %B %d, %Y", time_struct)  # Example: June 02, 2023
print(f'Finished training at {formatted_datetime}................')

