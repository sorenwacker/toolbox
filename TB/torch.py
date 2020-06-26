import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2))
            
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, X.input_dim)
            )
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# STOP

import re

with open(__file__, 'r') as this_file:
    for line in this_file.readlines():
        if re.search('STOP', line):
            break
        print(line, end="")    