import torch
import os
from tqdm import tqdm 
from data.dataset_utils import TilesClassificationDataLoader
from model_architectures.googlenet.googtilenet import make_googtilenet
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

class ThreeLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
         
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
         

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
        
        
def check_accuracy(model, X, Y):
    with torch.no_grad():
        x = torch.tensor(X).to(dtype=torch.float32)
        y = torch.tensor(Y).type(torch.LongTensor)
        scores = model(x)
        _, preds = scores.max(1)
        num_correct = (preds == y).sum()
        num_samples = preds.size(0)
    return float(num_correct) / num_samples
    

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cuda = torch.cuda.is_available()
in_channels = 4
z_dim = 512

tilenet = make_googtilenet(in_channels=in_channels, z_dim=z_dim)
if cuda: tilenet.cuda()

checkpoint = torch.load(args.checkpoint)
tilenet.load_state_dict(checkpoint)
tilenet.eval()

print(f'{args.checkpoint} set up complete.')

n_tiles = 8000
dataloader = TilesClassificationDataLoader(batch_size=1, num_tiles_requested=n_tiles)
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
    z = tilenet.forward(tile)
    if cuda: z = z.cpu()
    z = z.data.numpy()

    z.squeeze(0)
    X[i, :] = z
    

num_classes=66

# 60-20-20 split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
X_tr, X_val, y_tr, y_val  = train_test_split(X_tr, y_tr, test_size=0.25) # 0.25x0.8 = 0.2
train_size = y_tr.size

num_epochs = 50
hidden_dims = [(256,128), (128,64), (64,32)]

best_acc = 0
best_model = None
best_hidden = None
for (hidden_dim1,hidden_dim2) in hidden_dims:
    print('Training model for hidden dim ', (hidden_dim1,hidden_dim2))
    model = ThreeLayerFC(z_dim, hidden_dim1, hidden_dim2, num_classes)
    model.train()

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    for e in range(num_epochs):
        for i in range(train_size//10):
            x = torch.tensor(X_tr[i*10:(i+1)*10, :]).to(dtype=torch.float32)
            y = torch.tensor(y_tr[i*10:(i+1)*10]).type(torch.LongTensor)
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler:
                scheduler.step(loss)
    
    model.eval()
    acc = check_accuracy(model, X_val, y_val)
    print(f'{100*acc:.2f} correct on val set\n' )

    if acc>best_acc:
        best_acc = acc
        best_model = model
        best_hidden = (hidden_dim1,hidden_dim2)


acc = check_accuracy(best_model, X_val, y_val)
print(f'{best_hidden} is best hidden dimension - verifying {100*acc:.2f} is its accuracy on val set')

model.eval()
acc = check_accuracy(model, X_te, y_te)
print(f'{best_hidden} is best hidden dimension - {100*acc:.2f} is its accuracy on test set')

    


        