import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    X_train_tensor = torch.load(os.path.join(current_dir, 'X_train_tensor.pt'), weights_only=True, map_location=torch.device('cpu'))
    edge_index = torch.load(os.path.join(current_dir, 'edge_index.pt'), weights_only=True, map_location=torch.device('cpu'))
    
    if not isinstance(X_train_tensor, torch.Tensor):
        raise TypeError("X_train_tensor is not a valid torch tensor")
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError("edge_index is not a valid torch tensor")
        
except (FileNotFoundError, RuntimeError, TypeError) as e:
    print(f"Error loading tensor files from {current_dir}")
    print(f"Error details: {str(e)}")
    raise

class GATWithDimensionalityReduction(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, reduce_dim, num_heads=4):
        super(GATWithDimensionalityReduction, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, edge_dim=1)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels * 2, heads=num_heads, edge_dim=1)
        self.conv3 = GATConv(hidden_channels * 2 * num_heads, hidden_channels * 4, heads=num_heads, edge_dim=1)
        self.conv4 = GATConv(hidden_channels * 4 * num_heads, hidden_channels * 8, heads=num_heads, edge_dim=1)

        self.dim_reduce = nn.Linear(hidden_channels * 8 * num_heads, reduce_dim)

        self.fc1 = nn.Linear(reduce_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))

        x = self.dim_reduce(x)
        x = F.elu(x)

        x = F.elu(self.fc1(x))
        x = self.fc2(x)

        return x

torch.manual_seed(42)

in_channels = X_train_tensor.shape[1]
hidden_channels = 8
out_channels = 16
reduce_dim = 32
num_heads = 4