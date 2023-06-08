import torch
import os
from tqdm import tqdm 
from data.dataset_utils import TilesClassificationDataLoader
from model_architectures.resnets.tilenet import make_tilenet_18
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-cp", "--checkpoint", help="Relative path to checkpoint file. For example, models/GoogTiLeNet_epoch10.ckpt")

args = parser.parse_args()

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
         
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
         
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
        
img_type = 'naip'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cuda = torch.cuda.is_available()
in_channels = 4
z_dim = 512
tilenet = make_tilenet_18(in_channels=in_channels, z_dim=z_dim)
if cuda: tilenet.cuda()

checkpoint = torch.load(args.checkpoint)
tilenet.load_state_dict(checkpoint)
tilenet.eval()

print('TileNet50 set up complete.')

lr = 1e-3
optimizer = optim.Adam(tilenet.parameters(), lr=lr, betas=(0.5, 0.999))

print('Optimizer set up complete.')

dataloader = TilesClassificationDataLoader(batch_size=1)

n_tiles = 8000
X = np.zeros((n_tiles, z_dim))
y = np.zeros(n_tiles)


for i, sample in enumerate(tqdm(dataloader)):
    if i == n_tiles:
        break
    
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
    

num_classes=66

# 60-20-20 split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
X_tr, X_val, y_tr, y_val  = train_test_split(X_tr, y_tr, test_size=0.25) # 0.25x0.8 = 0.2

num_epochs = 1
train_size = y_tr.size
print(train_size)
hidden_dim = 256

print(z_dim)
print(num_classes)
model = TwoLayerFC(z_dim, hidden_dim, num_classes)
model.train()

for e in range(num_epochs):
    for i in range(train_size//10):
        x = torch.tensor(X_tr[i*10:(i+1)*10, :]).to(dtype=torch.float32)
        y = torch.tensor(y_tr[i*10:(i+1)*10]).type(torch.LongTensor)
        scores = model(x)
        loss = F.cross_entropy(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
                
# evaluate
print("Checking accuracy.")
num_correct = 0
num_samples = 0
model.eval()

with torch.no_grad():
    for i in range(y_te.size//10):
        x = torch.tensor(X_te[i*10:(i+1)*10, :]).to(dtype=torch.float32)
        y = torch.tensor(y_te[i*10:(i+1)*10]).type(torch.LongTensor)
        scores = model(x)
        _, preds = scores.max(1)
        print(y)
        print(preds)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        