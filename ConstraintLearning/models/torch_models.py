import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TabularDataset(Dataset):
    def __init__(self, X, y):
        """
        Characterizes a Dataset for PyTorch
        """
        
        self.n = X.shape[0]
        
        self.y = y.astype(np.float32).values.reshape(-1, 1)
        self.X = X.astype(np.float32).values
    
    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n
    
    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.X[idx]]


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_size, drop):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.drop = drop
        
        # First Layer
        first_lin_layer = nn.Linear(self.input_size, self.hidden_size)
        
        # Hidden Layers
        self.lin_layers = nn.ModuleList(
            [first_lin_layer]
            + [
                nn.Linear(self.hidden_size, self.hidden_size)
                for i in range(self.hidden_layers - 1)
            ]
        )
        
        # Output Layer
        self.output_layer = nn.Linear(self.hidden_size, output_size)
        
        # Dropout Layers
        self.droput_layers = nn.ModuleList(
            [nn.Dropout(self.drop) for layer in self.lin_layers]
        )

        # Batch Normalization
        # self.batch_norm_layers = nn.ModuleList(
        #     [nn.BatchNorm1d(self.hidden_size) for layer in self.lin_layers]
        # )

        
    def forward(self, x):
        for i, (lin, d) in enumerate(zip(self.lin_layers, self.droput_layers)):
            x = lin(x)
            # x = self.batch_norm_layers[i](x)
            x = F.relu(x)
            x = d(x)
        x = self.output_layer(x)
        return x
