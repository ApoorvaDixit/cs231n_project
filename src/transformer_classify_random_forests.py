import numpy as np
import os
import torch
from time import time
from torch.autograd import Variable
from data.dataset_utils import TilesClassificationDataLoader
from model_architectures.transformer.transformer import make_ViT
from tqdm import tqdm 

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

cuda = torch.cuda.is_available()
image_size = 100
z_dim = 512
vit = make_ViT(image_size = image_size, z_dim=z_dim)
if cuda: 
    vit = vit.cuda()

# Load parameters
model_fn = 'models/ViT_epoch5.ckpt'
checkpoint = torch.load(model_fn)
vit.load_state_dict(checkpoint)
vit.eval()

dataloader = TilesClassificationDataLoader(batch_size=1)

n_tiles = 8000
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
    z = vit.forward(tile)
    if cuda: z = z.cpu()
    z = z.data.numpy()

    z.squeeze(0)
    z = z.reshape(z.shape[1], z.shape[2])
    z = z.mean()

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