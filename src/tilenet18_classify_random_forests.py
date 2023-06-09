import numpy as np
import torch
from torch.autograd import Variable
from data.dataset_utils import TilesClassificationDataLoader
from model_architectures.resnets.tilenet import make_tilenet_18
from utils import get_timestr
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
tilenet = make_tilenet_18(in_channels=in_channels, z_dim=z_dim)
if cuda: 
    tilenet = tilenet.cuda()

# Load parameters
checkpoint = torch.load(args.checkpoint)
tilenet.load_state_dict(checkpoint)
tilenet.eval()


n_tiles = 8000
test_size = 0.2
X = np.zeros((n_tiles, z_dim))
y = np.zeros(n_tiles)

dataloader = TilesClassificationDataLoader(batch_size=1, num_tiles_requested = n_tiles)

print(f'Begin at {get_timestr()}, with test_size {test_size}................')
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

print(f'{get_timestr()} | finished embedding')
y = LabelEncoder().fit_transform(y)
print(f'{get_timestr()} | finished encoding labels')
# print(set(y))

n_trials = 100
accs = np.zeros((n_trials,))
for i in range(n_trials):
    # Splitting data and training RF classifer
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size)
    rf = RandomForestClassifier()
    rf.fit(X_tr, y_tr)
    accs[i] = rf.score(X_te, y_te)
print('Mean accuracy: {:0.4f}'.format(accs.mean()))
print('Standard deviation: {:0.4f}'.format(accs.std()))

print(f'End at {get_timestr()}................')

# Result with 10 epochs in tilenet, 27k labeled tiles in random forest classifier test split 0.2
# Mean accuracy: 0.6767
# Standard deviation: 0.0060





# Result with 10 epochs in tilenet, 8k labeled tiles in random forest classifier test split 0.2
# Mean accuracy: 0.6383
# Standard deviation: 0.0120

# Result with 15 epochs in tilenet, 8k labeled tiles in random forest classifier test split 0.2
# Mean accuracy: 0.6542
# Standard deviation: 0.0110

# Result with 10 epochs in tilenet, 8k labeled tiles in random forest classifier test split 0.5
# Mean accuracy: 0.6226
# Standard deviation: 0.0065

# Result with 15 epochs in tilenet, 8k labeled tiles in random forest classifier test split 0.5
# Mean accuracy: 0.6383
# Standard deviation: 0.0056
