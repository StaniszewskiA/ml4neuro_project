import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GATv2Conv

class GATv2Lightning(nn.Module):
    def __init__(self, in_features, n_gat_layers, hidden_dim, n_heads, dropout, slope, 
                 pooling_method, activation, norm_method, n_classes, lr, weight_decay):
        super(GATv2Lightning, self).__init__()

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATv2Conv(in_channels=in_features, out_channels=hidden_dim, heads=n_heads, dropout=dropout))
        
        for _ in range(n_gat_layers - 1):
            self.gat_layers.append(GATv2Conv(in_channels=hidden_dim * n_heads, out_channels=hidden_dim, heads=n_heads, dropout=dropout))

        self.fc_out = nn.Linear(hidden_dim * n_heads, n_classes)
        
        self.dropout = dropout
        self.activation = activation
        self.norm_method = norm_method

    def forward(self, x, edge_index):
        print(f'Input x shape: {x.shape}')
        print(f'Input edge_index shape: {edge_index.shape}')
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)
            print(f'After GAT layer {i+1}, x shape: {x.shape}')
            if self.activation == 'relu':
                x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
            print(f'After ReLU and Dropout, x shape: {x.shape}')

        x = self.fc_out(x)
        print(f'After fc_out, x shape: {x.shape}')
        return x
