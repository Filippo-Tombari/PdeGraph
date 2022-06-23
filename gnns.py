import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
import torch.optim as optim
from torch_geometric_temporal.nn.recurrent import A3TGCN,GConvGRU


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

    def forward(self, x, adjs, device):
        for i, (edge_index, _, size) in enumerate(adjs):
          edge_index = edge_index.to(device)
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

    def inference(self, x_all, loader, device):
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in loader:
                edge_index, _, size = adj
                edge_index = edge_index.to(device)
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


class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=32, 
                           periods=periods)
        # Equals single-shot prediction
        #self.recurrent = GConvGRU(node_features, filters, 3)
        self.linear = torch.nn.Linear(32, node_features)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h

class RecurrentGNN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGNN, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, 3)
        self.linear = torch.nn.Linear(filters, node_features)

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h

