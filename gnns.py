import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, LayerNorm, LeakyReLU, Flatten, Unflatten
from tqdm import tqdm
from torch_geometric.nn import SAGEConv,MessagePassing
from torch_geometric.loader import NeighborSampler
import torch.optim as optim
from torch_geometric_temporal.nn.recurrent import A3TGCN,GConvGRU, GConvLSTM


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(SAGE, self).__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.linear = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.linear.append(torch.nn.Linear(hidden_channels,10))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(10, hidden_channels))
            self.linear.append(torch.nn.Linear(hidden_channels,10))
        self.convs.append(SAGEConv(10, 10))
        self.linear.append(torch.nn.Linear(10,out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs, device):
        for i, (edge_index, _, size) in enumerate(adjs):
          edge_index = edge_index.to(device)
          xs = []
          if i != self.num_layers - 1:
            x = self.convs[i](x, edge_index)
            #x = F.normalize(x)
            x = F.leaky_relu(x,negative_slope=0.1)
          else:
            x = self.linear(x)
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
                if i != self.num_layers - 1:
                  #x = F.normalize(x)
                  x = self.convs[i](x, edge_index)
                  x = F.leaky_relu(x,negative_slope=0.1)
                  x = self.linear[i](x)
                  x = x = F.leaky_relu(x,negative_slope=0.1)
                else:
                  x = self.convs[i](x, edge_index)
                  x = self.linear[i](x)
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
    def __init__(self, node_features, hidden_channels, filters):
        super(RecurrentGNN, self).__init__()
        self.recurrent = torch.nn.ModuleList()
        self.linear = torch.nn.ModuleList()
        self.recurrent.append(GConvLSTM(node_features, hidden_channels, filters))
        self.linear.append(torch.nn.Linear(hidden_channels, int(hidden_channels/2)))

        self.recurrent.append(GConvLSTM(int(hidden_channels/2), int(hidden_channels/2), filters))
        self.linear.append(torch.nn.Linear(int(hidden_channels/2), int(hidden_channels/4)))

        self.recurrent.append(GConvLSTM(int(hidden_channels/4), int(hidden_channels/4), filters))
        self.linear.append(torch.nn.Linear(int(hidden_channels/4), node_features))
        #self.recurrent_start = GConvLSTM(node_features, hidden_channels, filters)
        #self.linear.append(torch.nn.Linear(hidden_channels, hidden_channels))
        #self.recurrent_end = GConvLSTM(hidden_channels, hidden_channels, filters)


    def forward(self, x, edge_index):
      h = x
      for i in range(len(self.recurrent)-1):
        h = self.recurrent[i](h, edge_index)
        h = F.leaky_relu(h[0],negative_slope=0.1)
        h = self.linear[i](h)
        h = F.leaky_relu(h,negative_slope=0.1)

      h = self.recurrent[-1](h, edge_index)
      h = F.leaky_relu(h[0],negative_slope=0.1)
      h = self.linear[-1](h)
      return h


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels = 1, last = False, **kwargs):
        super().__init__(aggr = "sum") 
        self.mlp = Seq(Linear(in_channels, hidden_channels//2),
                       LeakyReLU(0.1),
                       Linear(hidden_channels//2, out_channels),
                       LeakyReLU(0.1))
        self.mlp_edge = Seq(Linear(hidden_channels, hidden_channels//2),
                            LeakyReLU(0.1),
                            Linear(hidden_channels//2,hidden_channels),
                            LeakyReLU(0.1))
        self.last = last
    
    def forward(self, x, edge_attr, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        #node_feat, edge_feat = x
        if self.last:
          edge_attr = self.mlp_edge(edge_attr)
        
        x_new = self.propagate(edge_index, x=x,edge_attr = edge_attr)
        #x_new += x
        #if self.last:
          #return x_new
          
        return x_new 

    def message(self, x_i, edge_attr):
        # x_i has shape [E, in_channels]
        # edge_weights has shape [E, in_channels]
        tmp = torch.cat([x_i, edge_attr], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
    

class Encoder(torch.nn.Module):
  def __init__(self, in_node, in_edge, hidden_channels, mesh_nodes):
        super(Encoder, self).__init__()

        self.mlp_node = Seq(Linear(in_node,hidden_channels//2),
                            LeakyReLU(0.1),
                            Linear(hidden_channels//2,hidden_channels),
                            LeakyReLU(0.1))
        
        self.mlp_edge = Seq(Linear(in_edge,hidden_channels//2),
                            LeakyReLU(0.1),
                            Linear(hidden_channels//2,hidden_channels),
                            LeakyReLU(0.1))
        self.gnn = Seq(EdgeConv(hidden_channels*2,hidden_channels,out_channels = hidden_channels),
                       EdgeConv(hidden_channels*2,hidden_channels,out_channels = hidden_channels, last = True))
        self.mlp_graph = Seq(Flatten(0,1),
                             Linear(mesh_nodes*hidden_channels,hidden_channels),
                             LeakyReLU(0.1))
        
  def forward(self,x,edge_index):
    node_feat, edge_feat = x
    node_feat = self.mlp_node(node_feat)
    edge_feat = self.mlp_edge(edge_feat)
    for nn in self.gnn:
      node_feat = nn.forward(node_feat,edge_feat,edge_index)

    node_feat = self.mlp_graph(node_feat)

    return (node_feat,edge_feat)


class Decoder(torch.nn.Module):
  def __init__(self, in_channels, in_edge, hidden_channels, mesh_nodes):
        super(Decoder, self).__init__()

        self.mlp_node = Seq(Linear(hidden_channels,hidden_channels//2),
                            LeakyReLU(0.1),
                            Linear(hidden_channels//2,1))
        
        self.mlp_edge = Seq(Linear(hidden_channels,hidden_channels//2),
                            LeakyReLU(0.1))
                            #Linear(hidden_channels//2,hidden_channels),
                            #LeakyReLU(0.1))
        
        self.gnn = Seq(EdgeConv(hidden_channels,hidden_channels, out_channels = hidden_channels//2),
                       EdgeConv(hidden_channels,hidden_channels, out_channels = hidden_channels))
        
        self.mlp_graph = Seq(Linear(in_channels,mesh_nodes*hidden_channels),
                             LeakyReLU(0.1),
                             Unflatten(0,(mesh_nodes,hidden_channels)))
        
  def forward(self,x,edge_index):
    node_feat, edge_feat = x
    node_feat = self.mlp_graph(node_feat)
    '''
    edge_feat = self.mlp_edge(edge_feat)
    for nn in self.gnn:
      node_feat = nn.forward(node_feat,edge_feat,edge_index)
    '''
    node_feat = self.mlp_node(node_feat)
   

    return node_feat


class Sequential(torch.nn.Sequential):
    def forward(self, x,edge_index):
        for module in self._modules.values():
          x = module.forward(x,edge_index)
        return x