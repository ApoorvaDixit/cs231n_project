import torch
from tqdm import tqdm 
from data.dataset_utils import TilesClassificationDataLoader

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
         
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
         
    def forward(self, x):
        return self.fc2(F.relu(self.bn(self.fc1(x))))
        
img_type = 'naip'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cuda = torch.cuda.is_available()
in_channels = 4
z_dim = 512
TileNet = make_tilenet_50(in_channels=in_channels, z_dim=z_dim)
if cuda: TileNet.cuda()
TileNet.train()

print('TileNet50 set up complete.')

lr = 1e-3
optimizer = optim.Adam(TileNet.parameters(), lr=lr, betas=(0.5, 0.999))

print('Optimizer set up complete.')

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
    

num_classes=np.unique(y).size()

# 60-20-20 split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25) # 0.25x0.8 = 0.2

num_epochs = 1
train_size = y_train.size()
hidden_dim = 128

model = TwoLayerFC(z_dim, hidden_dim, num_classes)

for e in range(num_epochs):
    for i in range(train_size//100):
        x = X_tr[i:i+100, :]
        y = y_tr[i:i+100]
        scores = model(x)
        
        loss = F.cross_entropy(scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % print_every == 0:
                print('Iteration %d, loss = %.4f' % (i, loss.item()))
                
                
