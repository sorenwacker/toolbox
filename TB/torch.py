import pandas as pd 
import numpy as np 

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from matplotlib.pyplot import scatter, title, show

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2, layers=[64, 32], add_sigmoid=False, dropout=0):
        super(AutoEncoder,self).__init__()
        
        layout = [input_dim] + layers + [latent_dim]
        layout = [(i,j) for i, j in zip(layout[:-1], layout[1:])]

        encoder = []

        for i,j in layout:
            encoder.append(nn.Dropout(dropout))
            encoder.append(nn.Linear(i, j))
            encoder.append(nn.ReLU())
        encoder.pop()

        if add_sigmoid:
            encoder.append(nn.Sigmoid())

        layout.reverse()

        decoder = []
        for j,i in layout:
            decoder.append(nn.Linear(i, j))
            decoder.append(nn.ReLU())
        decoder.pop()


        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.snapshots = []

        for i in encoder:
            print(i)
        for i in decoder:
            print(i)
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, X_train, epochs, batch_size=8, 
            num_workers=4, shuffle=True, labels=None,
            lr=1e-4):
        
        if isinstance(X_train, pd.DataFrame):
            ndx = X_train.index
            X_train = X_train.values
            
        X_train = torch.tensor(X_train).float()

        dataloader = DataLoader(X_train, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                num_workers=batch_size)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
     
        criterion = nn.MSELoss()

        for i in tqdm(range(1, epochs+1)):
            for data in dataloader:
                optimizer.zero_grad()
                # compute reconstructions
                outputs = self(data)
                # compute training reconstruction loss
                train_loss = criterion(outputs, data)
                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                # compute the epoch training loss
                assert  train_loss is not np.NaN
            
            if (i) % 10 == 0:    
                result = pd.DataFrame( self.encoder(X_train).detach().numpy(), index=ndx ).add_prefix('AE_')
                result['Epoch'] = i
                self.snapshots.append( result )
                scatter(result['AE_0'], result['AE_1'], c=labels)
                title(f'Epoch {i}, loss={train_loss:2.2f}')
                show()
                
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            ndx = X.index
            X = X.values
        else:
            ndx = None
        X = torch.tensor(X).float()
        enc = self.encoder(X)
        enc = pd.DataFrame( enc ).add_prefix('AE_')
        if ndx is not None:
            enc.index = ndx
        return enc 