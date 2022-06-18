import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
import torch.optim as optim

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
          xs = []
          x = self.convs[i](x, edge_index)
          if i != self.num_layers - 1:
              x = F.normalize(x)
              x = F.relu(x)
          xs.append(x)
          x_all = torch.cat(xs, dim=0)
          if i == self.num_layers-1:
            layer_embeddings = x_all
        return layer_embeddings

    def inference(self, x_all, loader):
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in loader:
                edge_index, _, size = adj
                total_edges += edge_index.size(1)
                x = x_all[n_id]
                x = self.convs[i](x, edge_index)
                if i != self.num_layers - 1:
                    x = F.normalize(x)
                    x = F.relu(x)
                xs.append(x)
            x_all = torch.cat(xs, dim=0)
            if i == self.num_layers-1:                
                layer_embeddings = x_all

        return layer_embeddings