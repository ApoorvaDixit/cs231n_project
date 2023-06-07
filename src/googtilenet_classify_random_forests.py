import numpy as np
import os
import torch
from time import time
from torch.autograd import Variable
from data.dataset_utils import TilesClassificationDataLoader
from model_architectures.googlenet.googtilenet import make_googtilenet
from tqdm import tqdm 

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-cp", "--checkpoint", help="Relative path to checkpoint file. For example, models/GoogTiLeNet_epoch10.ckpt")

args = parser.parse_args()

cuda = torch.cuda.is_available()
in_channels = 4
z_dim = 512
net = make_googtilenet(in_channels=in_channels, z_dim=z_dim)
if cuda: 
    net = net.cuda()

# Load parameters
checkpoint = torch.load(args.checkpoint)
net.load_state_dict(checkpoint)
net.eval()



n_tiles = 8000
X = np.zeros((n_tiles, z_dim))
y = np.zeros(n_tiles)

dataloader = TilesClassificationDataLoader(batch_size=1, num_tiles_requested = n_tiles)

for i, sample in enumerate(tqdm(dataloader)):
    tile = sample['tile']
    tile = torch.squeeze(tile, dim=0)
    label = sample['label']
    y[i] = label
    tile = Variable(tile)
    if cuda: 
        tile = tile.cuda()
    z = net.forward(tile)
    if cuda: z = z.cpu()
    z = z.data.numpy()

    z.squeeze(0)
    X[i, :] = z

y = LabelEncoder().fit_transform(y)
# print(set(y))

n_trials = 100
accs = np.zeros((n_trials,))
for i in range(n_trials):
    # Splitting data and training RF classifer
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    rf = RandomForestClassifier()
    rf.fit(X_tr, y_tr)
    accs[i] = rf.score(X_te, y_te)
    print(f'Trial {i} has accuracy {accs[i]}')
print('Mean accuracy: {:0.4f}'.format(accs.mean()))
print('Standard deviation: {:0.4f}'.format(accs.std()))

# Results with googlenetv1, 10 epochs + aux outputs disabled, 27k labeled tiles in random forest classifier test split 0.2
# Mean accuracy: 0.6580
# Standard deviation: 0.0058