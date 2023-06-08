from torch.utils.data import Dataset, DataLoader
import sys
import os
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

import time
from tqdm import tqdm 

from data.dataset_utils import TripletDataLoader
from utils import get_timestr
from model_architectures.googlenet.googtilenet import make_googtilenet
model_name = 'GoogTiLeNet'
# from model_architectures.googlenet.googtilenet_v3 import make_googtilenet
# model_name = 'GoogTiLeNet_v3_xav'
# from model_architectures.googlenet.googtilenet_v3_trim import make_googtilenet
# model_name = 'GoogTiLeNet_v3_trim'
from training import train_triplet_epoch

img_type = 'naip'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cuda = torch.cuda.is_available()
in_channels = 4
z_dim = 512
net = make_googtilenet(in_channels=in_channels, z_dim=z_dim)
if cuda: net.cuda()
net.train()


# Load parameters
# model_fn = 'models/GoogTiLeNet_v3_epoch5.ckpt'
# checkpoint = torch.load(model_fn)
# net.load_state_dict(checkpoint)


print('GoogTiLeNet set up complete.')

lr = 1e-2
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))#, weight_decay=0.01)
scheduler = None
# scheduler = MultiStepLR(optimizer, 
#                         milestones=[1,2,3, 4], # List of epoch indices
#                         gamma =0.5)

print('Optimizer set up complete.')

dataloader = TripletDataLoader(img_type, batch_size=64)

print('Dataset set up.')

epochs = 15
epoch_start = 0
margin = 10
l2 = 0.01
print_every = 10000
max_grad_norm = None #1.0

model_dir = 'models/'
if not os.path.exists(model_dir): os.makedirs(model_dir)

t0 = time.time()
print(f'Begin training at {get_timestr()}................')
for epoch in tqdm(range(epoch_start, epochs), desc="epoch loop"):
    (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
        net, cuda, dataloader, optimizer, epoch+1, margin=margin, l2=l2,
        print_every=print_every, t0=t0, max_grad_norm=max_grad_norm, scheduler=scheduler)
    
    model_fn = os.path.join(model_dir, f'{model_name}_epoch{epoch+1}_cosine.ckpt')
    torch.save(net.state_dict(), model_fn)

print(f'Finished training at {get_timestr()}................')

