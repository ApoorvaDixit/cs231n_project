import numpy as np
import os
import torch
from time import time
from torch.autograd import Variable
from data.dataset_utils import TilesClassificationDataLoader
from model_architectures.googlenet.googtilenet import make_googtilenet
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

print(f'Begin at {get_timestr()}................')
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

print(f'End at {get_timestr()}................')

# Results with googlenetv1, 5 epochs + aux outputs disabled, 27k labeled tiles in random forest classifier test split 0.2
# Mean accuracy: 0.6261
# Standard deviation: 0.0056

# Results with googlenetv1, 10 epochs + aux outputs disabled, 27k labeled tiles in random forest classifier test split 0.2
# Mean accuracy: 0.6580
# Standard deviation: 0.0058

# Results with googlenetv1, 10 epochs + aux outputs disabled, 8k labeled tiles in random forest classifier test split 0.2


# Results with googlenetv1, 15 epochs + aux outputs disabled, 8k labeled tiles in random forest classifier test split 0.2
# Trial 0 has accuracy 0.62
# Trial 1 has accuracy 0.63625
# Trial 2 has accuracy 0.650625
# Trial 3 has accuracy 0.633125
# Trial 4 has accuracy 0.623125
# Trial 5 has accuracy 0.65625
# Trial 6 has accuracy 0.64125
# Trial 7 has accuracy 0.6475
# Trial 8 has accuracy 0.625
# Trial 9 has accuracy 0.63625
# Trial 10 has accuracy 0.6275
# Trial 11 has accuracy 0.638125
# Trial 12 has accuracy 0.64125
# Trial 13 has accuracy 0.638125
# Trial 14 has accuracy 0.64125
# Trial 15 has accuracy 0.641875
# Trial 16 has accuracy 0.635
# Trial 17 has accuracy 0.63
# Trial 18 has accuracy 0.631875
# Trial 19 has accuracy 0.629375
# Trial 20 has accuracy 0.626875
# Trial 21 has accuracy 0.63
# Trial 22 has accuracy 0.625
# Trial 23 has accuracy 0.638125
# Trial 24 has accuracy 0.656875
# Trial 25 has accuracy 0.631875
# Trial 26 has accuracy 0.640625
# Trial 27 has accuracy 0.631875
# Trial 28 has accuracy 0.651875
# Trial 29 has accuracy 0.64375
# Trial 30 has accuracy 0.651875
# Trial 31 has accuracy 0.628125
# Trial 32 has accuracy 0.644375
# Trial 33 has accuracy 0.649375
# Trial 34 has accuracy 0.64125
# Trial 35 has accuracy 0.630625
# Trial 36 has accuracy 0.62
# Trial 37 has accuracy 0.634375
# Trial 38 has accuracy 0.631875
# Trial 39 has accuracy 0.6375
# Trial 40 has accuracy 0.638125
# Trial 41 has accuracy 0.62625
# Trial 42 has accuracy 0.648125
# Trial 43 has accuracy 0.623125
# Trial 44 has accuracy 0.64125
# Trial 45 has accuracy 0.63625
# Trial 46 has accuracy 0.625625
# Trial 47 has accuracy 0.645625
# Trial 48 has accuracy 0.635625
# Trial 49 has accuracy 0.640625
# Trial 50 has accuracy 0.61
# Trial 51 has accuracy 0.654375
# Trial 52 has accuracy 0.63375
# Trial 53 has accuracy 0.650625
# Trial 54 has accuracy 0.63125
# Trial 55 has accuracy 0.635
# Trial 56 has accuracy 0.645625
# Trial 57 has accuracy 0.6375
# Trial 58 has accuracy 0.63375
# Trial 59 has accuracy 0.62125
# Trial 60 has accuracy 0.6275
# Trial 61 has accuracy 0.63875
# Trial 62 has accuracy 0.641875
# Trial 63 has accuracy 0.62875
# Trial 64 has accuracy 0.606875
# Trial 65 has accuracy 0.640625
# Trial 66 has accuracy 0.62875
# Trial 67 has accuracy 0.63625
# Trial 68 has accuracy 0.644375
# Trial 69 has accuracy 0.64125
# Trial 70 has accuracy 0.638125
# Trial 71 has accuracy 0.65
# Trial 72 has accuracy 0.62125
# Trial 73 has accuracy 0.628125
# Trial 74 has accuracy 0.630625
# Trial 75 has accuracy 0.63125
# Trial 76 has accuracy 0.6375
# Trial 77 has accuracy 0.646875
# Trial 78 has accuracy 0.6325
# Trial 79 has accuracy 0.635625
# Trial 80 has accuracy 0.629375
# Trial 81 has accuracy 0.653125
# Trial 82 has accuracy 0.646875
# Trial 83 has accuracy 0.65375
# Trial 84 has accuracy 0.649375
# Trial 85 has accuracy 0.6325
# Trial 86 has accuracy 0.6325
# Trial 87 has accuracy 0.624375
# Trial 88 has accuracy 0.63
# Trial 89 has accuracy 0.62625
# Trial 90 has accuracy 0.644375
# Trial 91 has accuracy 0.62125
# Trial 92 has accuracy 0.63125
# Trial 93 has accuracy 0.635
# Trial 94 has accuracy 0.643125
# Trial 95 has accuracy 0.6225
# Trial 96 has accuracy 0.633125
# Trial 97 has accuracy 0.628125
# Trial 98 has accuracy 0.61875
# Trial 99 has accuracy 0.624375
# Mean accuracy: 0.6355
# Standard deviation: 0.0099


# Results with googlenetv1, 10 epochs + aux outputs disabled, 8k labeled tiles in random forest classifier test split 0.5
# Trial 0 has accuracy 0.607
# Trial 1 has accuracy 0.60975
# Trial 2 has accuracy 0.608
# Trial 3 has accuracy 0.60225
# Trial 4 has accuracy 0.60475
# Trial 5 has accuracy 0.61125
# Trial 6 has accuracy 0.61575
# Trial 7 has accuracy 0.599
# Trial 8 has accuracy 0.6145
# Trial 9 has accuracy 0.6125
# Trial 10 has accuracy 0.6005
# Trial 11 has accuracy 0.60625
# Trial 12 has accuracy 0.5985
# Trial 13 has accuracy 0.60775
# Trial 14 has accuracy 0.6085
# Trial 15 has accuracy 0.6095
# Trial 16 has accuracy 0.60975
# Trial 17 has accuracy 0.60125
# Trial 18 has accuracy 0.6075
# Trial 19 has accuracy 0.618
# Trial 20 has accuracy 0.60625
# Trial 21 has accuracy 0.6055
# Trial 22 has accuracy 0.6095
# Trial 23 has accuracy 0.61875
# Trial 24 has accuracy 0.61
# Trial 25 has accuracy 0.60775
# Trial 26 has accuracy 0.617
# Trial 27 has accuracy 0.606
# Trial 28 has accuracy 0.6045
# Trial 29 has accuracy 0.60825
# Trial 30 has accuracy 0.61025
# Trial 31 has accuracy 0.61075
# Trial 32 has accuracy 0.60275
# Trial 33 has accuracy 0.6065
# Trial 34 has accuracy 0.605
# Trial 35 has accuracy 0.6115
# Trial 36 has accuracy 0.614
# Trial 37 has accuracy 0.60825
# Trial 38 has accuracy 0.612
# Trial 39 has accuracy 0.59975
# Trial 40 has accuracy 0.60475
# Trial 41 has accuracy 0.61
# Trial 42 has accuracy 0.60775
# Trial 43 has accuracy 0.6045
# Trial 44 has accuracy 0.60225
# Trial 45 has accuracy 0.61025
# Trial 46 has accuracy 0.61125
# Trial 47 has accuracy 0.60325
# Trial 48 has accuracy 0.5995
# Trial 49 has accuracy 0.603
# Trial 50 has accuracy 0.616
# Trial 51 has accuracy 0.61975
# Trial 52 has accuracy 0.603
# Trial 53 has accuracy 0.613
# Trial 54 has accuracy 0.6115
# Trial 55 has accuracy 0.602
# Trial 56 has accuracy 0.598
# Trial 57 has accuracy 0.60275
# Trial 58 has accuracy 0.60775
# Trial 59 has accuracy 0.60875
# Trial 60 has accuracy 0.6
# Trial 61 has accuracy 0.611
# Trial 62 has accuracy 0.612
# Trial 63 has accuracy 0.61075
# Trial 64 has accuracy 0.6065
# Trial 65 has accuracy 0.6085
# Trial 66 has accuracy 0.599
# Trial 67 has accuracy 0.6145
# Trial 68 has accuracy 0.612
# Trial 69 has accuracy 0.60425
# Trial 70 has accuracy 0.6065
# Trial 71 has accuracy 0.6055
# Trial 72 has accuracy 0.6005
# Trial 73 has accuracy 0.61525
# Trial 74 has accuracy 0.60675
# Trial 75 has accuracy 0.6145
# Trial 76 has accuracy 0.61375
# Trial 77 has accuracy 0.61475
# Trial 78 has accuracy 0.6165
# Trial 79 has accuracy 0.61375
# Trial 80 has accuracy 0.6075
# Trial 81 has accuracy 0.615
# Trial 82 has accuracy 0.604
# Trial 83 has accuracy 0.604
# Trial 84 has accuracy 0.6015
# Trial 85 has accuracy 0.6135
# Trial 86 has accuracy 0.61425
# Trial 87 has accuracy 0.61425
# Trial 88 has accuracy 0.616
# Trial 89 has accuracy 0.602
# Trial 90 has accuracy 0.613
# Trial 91 has accuracy 0.61
# Trial 92 has accuracy 0.608
# Trial 93 has accuracy 0.6065
# Trial 94 has accuracy 0.61425
# Trial 95 has accuracy 0.6085
# Trial 96 has accuracy 0.60975
# Trial 97 has accuracy 0.62175
# Trial 98 has accuracy 0.60525
# Trial 99 has accuracy 0.60125
# Mean accuracy: 0.6084
# Standard deviation: 0.0053

# Results with googlenetv1, 15 epochs + aux outputs disabled, 8k labeled tiles in random forest classifier test split 0.5
# Trial 0 has accuracy 0.61825
# Trial 1 has accuracy 0.6175
# Trial 2 has accuracy 0.6195
# Trial 3 has accuracy 0.61975
# Trial 4 has accuracy 0.60825
# Trial 5 has accuracy 0.61425
# Trial 6 has accuracy 0.62425
# Trial 7 has accuracy 0.6155
# Trial 8 has accuracy 0.616
# Trial 9 has accuracy 0.62675
# Trial 10 has accuracy 0.623
# Trial 11 has accuracy 0.628
# Trial 12 has accuracy 0.62175
# Trial 13 has accuracy 0.61575
# Trial 14 has accuracy 0.614
# Trial 15 has accuracy 0.61125
# Trial 16 has accuracy 0.618
# Trial 17 has accuracy 0.618
# Trial 18 has accuracy 0.62875
# Trial 19 has accuracy 0.62375
# Trial 20 has accuracy 0.614
# Trial 21 has accuracy 0.624
# Trial 22 has accuracy 0.621
# Trial 23 has accuracy 0.62325
# Trial 24 has accuracy 0.61775
# Trial 25 has accuracy 0.61225
# Trial 26 has accuracy 0.633
# Trial 27 has accuracy 0.62325
# Trial 28 has accuracy 0.61725
# Trial 29 has accuracy 0.61525
# Trial 30 has accuracy 0.63075
# Trial 31 has accuracy 0.625
# Trial 32 has accuracy 0.62
# Trial 33 has accuracy 0.625
# Trial 34 has accuracy 0.61375
# Trial 35 has accuracy 0.62325
# Trial 36 has accuracy 0.6225
# Trial 37 has accuracy 0.61625
# Trial 38 has accuracy 0.618
# Trial 39 has accuracy 0.61675
# Trial 40 has accuracy 0.61175
# Trial 41 has accuracy 0.6225
# Trial 42 has accuracy 0.6205
# Trial 43 has accuracy 0.6305
# Trial 44 has accuracy 0.62925
# Trial 45 has accuracy 0.621
# Trial 46 has accuracy 0.61375
# Trial 47 has accuracy 0.6235
# Trial 48 has accuracy 0.6195
# Trial 49 has accuracy 0.62175
# Trial 50 has accuracy 0.62
# Trial 51 has accuracy 0.622
# Trial 52 has accuracy 0.61675
# Trial 53 has accuracy 0.624
# Trial 54 has accuracy 0.62375
# Trial 55 has accuracy 0.6075
# Trial 56 has accuracy 0.6205
# Trial 57 has accuracy 0.60925
# Trial 58 has accuracy 0.60925
# Trial 59 has accuracy 0.6165
# Trial 60 has accuracy 0.619
# Trial 61 has accuracy 0.6175
# Trial 62 has accuracy 0.62275
# Trial 63 has accuracy 0.6205
# Trial 64 has accuracy 0.62125
# Trial 65 has accuracy 0.61525
# Trial 66 has accuracy 0.61825
# Trial 67 has accuracy 0.6155
# Trial 68 has accuracy 0.61925
# Trial 69 has accuracy 0.62625
# Trial 70 has accuracy 0.62025
# Trial 71 has accuracy 0.62225
# Trial 72 has accuracy 0.62525
# Trial 73 has accuracy 0.624
# Trial 74 has accuracy 0.617
# Trial 75 has accuracy 0.62275
# Trial 76 has accuracy 0.6175
# Trial 77 has accuracy 0.6165
# Trial 78 has accuracy 0.6245
# Trial 79 has accuracy 0.615
# Trial 80 has accuracy 0.62225
# Trial 81 has accuracy 0.61525
# Trial 82 has accuracy 0.61525
# Trial 83 has accuracy 0.6145
# Trial 84 has accuracy 0.618
# Trial 85 has accuracy 0.61825
# Trial 86 has accuracy 0.623
# Trial 87 has accuracy 0.62525
# Trial 88 has accuracy 0.608
# Trial 89 has accuracy 0.6165
# Trial 90 has accuracy 0.61225
# Trial 91 has accuracy 0.62375
# Trial 92 has accuracy 0.634
# Trial 93 has accuracy 0.627
# Trial 94 has accuracy 0.6275
# Trial 95 has accuracy 0.61625
# Trial 96 has accuracy 0.62825
# Trial 97 has accuracy 0.625
# Trial 98 has accuracy 0.61225
# Trial 99 has accuracy 0.6045
# Mean accuracy: 0.6197
# Standard deviation: 0.0057