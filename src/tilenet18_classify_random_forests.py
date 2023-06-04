import numpy as np
import os
import torch
from time import time
from torch.autograd import Variable
from data.dataset_utils import TilesClassificationDataLoader
from model_architectures.resnets.tilenet import make_tilenet_18
from tqdm import tqdm 

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

cuda = torch.cuda.is_available()
in_channels = 4
z_dim = 512
tilenet = make_tilenet_18(in_channels=in_channels, z_dim=z_dim)
if cuda: 
    tilenet = tilenet.cuda()

# Load parameters
model_fn = 'models/TileNet18_epoch10.ckpt'
checkpoint = torch.load(model_fn)
tilenet.load_state_dict(checkpoint)
tilenet.eval()

dataloader = TilesClassificationDataLoader(batch_size=1)

n_tiles = 27972
X = np.zeros((n_tiles, z_dim))
y = np.zeros(n_tiles)


for i, sample in enumerate(tqdm(dataloader)):
    tile = sample['tile']
    tile = torch.squeeze(tile, dim=0)
    label = sample['label']
    y[i] = label
    tile = Variable(tile)
    if cuda: 
        tile = tile.cuda()
    z = tilenet.encode(tile)
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
print('Mean accuracy: {:0.4f}'.format(accs.mean()))
print('Standard deviation: {:0.4f}'.format(accs.std()))

# Mean accuracy: 0.6767
# Standard deviation: 0.0060