import torch
import torch.nn as nn
from torch_scatter import scatter_add



class MLP(nn.Module):
    '''Class for creating a Multi-Layer Perceptron
          Attributes
            layers      (List)      A list of layers transforms a tensor x into f(Wx + b), where
                                    f is SiLU activation function, W is the weight matrix and b the bias tensor.


    '''
    def __init__(self, num_layers,in_channels,hidden_channels,out_channels):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels,hidden_channels))
        self.layers.append(nn.SiLU())
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels,hidden_channels))
            self.layers.append(nn.SiLU())
        self.layers.append(nn.Linear(hidden_channels, out_channels))

               
    def forward(self, x):
        '''

        :param x: torch.float() Tensor
        :return: torch.float() Tensor
        '''
        for layer in self.layers:
            x = layer(x)
        return x



class EdgeModel(torch.nn.Module):
    '''Class for creating a model for the edges
        Attributes
            edge_mlp      (object)    A MLP object that tranforms edge features

    '''
    def __init__(self, mlp_layers, hidden_channels):
        super(EdgeModel, self).__init__()
        self.edge_mlp = MLP(mlp_layers,3*hidden_channels,hidden_channels,hidden_channels)

    def forward(self, src, dest, edge_attr):
        '''

        :param src:         torch.float() Tensor with shape [batch_size-1,n_edges,hidden_channels]. Node features
                            corresponding to source nodes of the edges.
        :param dest:        torch.float() Tensor with shape [batch_size-1,n_edges,hidden_channels]. Node features
                            corresponding to destination nodes of the edges.
        :param edge_attr:   torch.float() Tensor with shape [batch_size-1,n_edges,hidden_channels].
        :return:            torch.float() Tensor with shape [batch_size-1,n_edges,hidden_channels].
        '''
        out = torch.cat([edge_attr, src, dest], dim=2)
        out = self.edge_mlp(out)
        return out


class NodeModel(torch.nn.Module):
    '''Class for creating a model for the nodes
        Attributes
            node_mlp      (object)    A MLP object that combines and tranforms node and edge features
    '''
    def __init__(self, mlp_layers, hidden_channels):
        super(NodeModel, self).__init__()
        self.node_mlp = MLP(mlp_layers,2*hidden_channels,hidden_channels,hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        '''

        :param x:           torch.float() Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        :param edge_index:  torch.long() Tensor with shape [2,n_edges].
        :param edge_attr:   torch.float() Tensor with shape [batch_size-1,n_edges,hidden_channels].
        :return:            torch.float() Tensor with shape [batch_size-1,n_nodes,hidden_channels]
        '''
        src, dest = edge_index
        out = scatter_add(edge_attr, dest, dim=1, dim_size=x.size(1))

        out = torch.cat([x, out], dim=2)
        out = self.node_mlp(out)
        return out



class MPLayer(torch.nn.Module):
    '''Class for creating a single message passing layer
    Attributes
            edge_model      (object)    A edge_model object that transforms the current edge_features
            node_model      (object)    A node_model object that combines node_features and edge_features and
                                        transforms them.


    '''

    def __init__(self, edge_model=None, node_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, x, edge_index, edge_attr):
        '''Performs a message passing forward pass

        :param x:           torch.float() Tensor with shape [batch_size-1,n_nodes,hidden_channels].
        :param edge_index:  torch.long() Tensor with shape [2,n_edges].
        :param edge_attr:   torch.float() Tensor with shape [batch_size-1,n_edges,hidden_channels]
        :return:            (torch.float() Tensor with shape [batch_size-1,n_nodes,hidden_channels],
                            torch.float() Tensor with shape [batch_size-1,n_nodes,hidden_channels])
        '''

        src = edge_index[0]
        dest = edge_index[1]

        edge_attr = self.edge_model(x[:,src], x[:,dest], edge_attr)
        x = self.node_model(x, edge_index, edge_attr)

        return x, edge_attr



class GNN(torch.nn.Module):
    """Class for creating a Graph Neural Network
            Attributes
                encoder_node    (object)    A MLP object that encodes node input features.
                encoder_edge    (object)    A MLP object that encodes edge input features.
                processor       (List)      A list of MPLayer objects of length mp_steps that propagate
                                            the messages across the mesh nodes.
                decoder         (Object)    A MLP object that decodes the output features.

    """
    def __init__(self, in_node, in_edge, hidden_channels,out_channels, mlp_layers, mp_steps):
        super(GNN, self).__init__()


        # Encoder MLPs
        self.encoder_node = MLP(mlp_layers,in_node,hidden_channels,hidden_channels)
        self.encoder_edge = MLP(mlp_layers,in_edge,hidden_channels,hidden_channels)
        # Processor MLPs
        self.processor = nn.ModuleList()
        for _ in range(mp_steps):
            node_model = NodeModel(mlp_layers,hidden_channels)
            edge_model = EdgeModel(mlp_layers,hidden_channels)
            GraphNet = MPLayer(node_model=node_model, edge_model=edge_model)
            self.processor.append(GraphNet)
        # Decoder MLP
        self.decoder = MLP(mlp_layers,hidden_channels,hidden_channels,out_channels)

    def forward(self, x,edge_index,edge_attr):
        '''Performs a forward pass across the GNN

        :param x:           torch.float() Tensor with shape [batch_size-1,n_nodes,in_node]. It is the input features
                            tensor.
        :param edge_index:  torch.long() Tensor with shape [2,n_edges]. It is the edge connectivity matrix
                            of the mesh, where edge_index[0] returns the source nodes and edge_index[1]
                            returns the destination nodes.
        :param edge_attr:   torch.float() Tensor with shape [batch_size-1,n_edges,in_edge]. It is the matrix containing
                            the edge fetaures for each edge in the mesh
        :return:            torch.float() Tensor with shape [batch_size-1,n_nodes,in_node]. It is the output
                            features tensor.
        '''
        #Decode

        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)

        #Process
        for GraphNet in self.processor:
            x_res, edge_attr_res = GraphNet(x, edge_index, edge_attr)
            x += x_res
            edge_attr += edge_attr_res

        #Decode

        x = self.decoder(x)

        return x




    


