import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class FeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers # A list containing the number of nodes in each hidden layer
        self.drop = drop
        
        # Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.input_size, self.hidden_layers[0])]
            + [
                nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]) for i in range(len(self.hidden_layers) - 1)
            ]
            + [nn.Linear(self.hidden_layers[-1], self.output_size)]
        )
        
        # Dropout layers (only for hidden layers)
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(self.drop) for _ in range(len(self.layers) - 1)]
        )

    def forward(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x




def mape_loss(y_pred, y_true):
    """
    Calculates the Mean Absolute Percentage Error.
    Args:
        y_pred: The predicted values from the model.
        y_true: The ground truth values.
    Returns:
        A PyTorch tensor with the MAPE loss.
    """
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    # Ensure tensors are on the same device and dtype
    y_true = y_true.to(y_pred.device, dtype=y_pred.dtype)
    
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape