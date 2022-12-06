import numpy as np
import torch
import dolfin
import random
import json
import os



def buildconnectivity(mesh):
    '''builds the connectivity matrix of a mesh
    :param mesh:    dolfin.cpp.mesh.Mesh object.
    :return:        torch.Tensor with shape [2,n_edges].
    '''
    mesh.init(mesh.topology().dim()-1,0)
    edge_to_vertex = mesh.topology()(mesh.topology().dim()-1,0)
    for edge in range(mesh.num_edges()):
        if edge == 0:
          edge_index = np.expand_dims(edge_to_vertex(edge),axis=0)
        else:
          edge_index = np.append(edge_index,np.expand_dims(edge_to_vertex(edge),axis=0),axis=0)
    return edge_index.astype('int32')


def initialize_weights(edge_index, mesh):
    '''initialize edge attributes
    :param edge_index:  torch.Tensor with shape [2,n_edges].
    :param mesh:        dolfin.cpp.mesh.Mesh object.
    :return:            torch.fTensor with shape [n_edges,3]
    '''
    edge_weights = torch.zeros((edge_index.shape[1],3))
    for k in range(edge_index.shape[1]):
        i = edge_index[0,k]
        j = edge_index[1,k]
        edge_weights[k,0:2] = torch.from_numpy(mesh.coordinates()[i,:] - mesh.coordinates()[j,:])
        edge_weights[k,2] = np.linalg.norm(mesh.coordinates()[i,:] - mesh.coordinates()[j,:],ord = 2)

    return edge_weights

def load_data(indices,json_data,device,mydir):
    '''Creates a dataset from a json file selecting the keys contained in indices
    :param indices:     list of strings containing the keys to choose
    :param json_data:   json file that contains all the simulations. Every key contains dictionary with the following keys:
                        -'mesh': name of the mesh corresponding to the key simulation
                        -'traj': numpy.ndarray containing the trajectories of the key simulation
    :param device:      either 'cuda' or 'cpu'
    :param mydir:       directory in which the meshes and the json file are contained
    :return:            {'mesh': dolfin.cpp.mesh.Mesh, 'edge_index': torch.Tensor, 'edge_weights': torch.Tensor,
                        'trajs': torch.Tensor, "n_b_nodes": int}
    '''
    meshes = []
    edge_indices = []
    edge_weights = []
    n_b_nodes = []
    trajs = []
    dt = 0.02

    for i in indices:
        mesh = dolfin.cpp.mesh.Mesh(mydir + json_data[i]['mesh'] + ".xml")
        edge_index = torch.t(torch.from_numpy(buildconnectivity(mesh)).long()).to(device)
        meshes.append(mesh)
        edge_indices.append(edge_index)
        edge_weights.append(initialize_weights(edge_index, mesh).to(device))

        # get boundary nodes
        bmesh = dolfin.BoundaryMesh(mesh, "exterior", True)
        n_b_nodes.append(bmesh.coordinates().shape[0])

        traj = torch.Tensor(json_data[i]['traj']).float().unsqueeze(dim=2).to(device)
        # Create a dummy feature to indicate the boundary nodes
        # (they are listed first in the coordinates array)
        bindex = torch.zeros(traj.shape).to(device)
        bindex[:, :n_b_nodes[-1], 0] = 1
        # Create a tensor containg the timesteps for each trajectory
        dt_tensor = torch.stack([torch.full((traj.shape[1], 1), dt * j) for j in range(traj.shape[0])]).to(device)
        traj = torch.cat((traj, dt_tensor, bindex), 2)
        trajs.append(traj)

    data = {'mesh': meshes, 'edge_index': edge_indices, 'edge_weights': edge_weights, 'trajs': trajs,
            "n_b_nodes": n_b_nodes}
    return data


def create_dataset(device, train_size=60,valid_size=20):
    ''' Creates training, validation and test set'''
    mydir = os.getcwd() + f'/files/'

    with open(mydir+'data.json', 'r') as f:
        json_data = json.loads(f.read())

    random.seed(10)
    indices = list(json_data.keys())
    random.shuffle(indices)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size+valid_size]
    test_indices = indices[train_size+valid_size:]

    train_data = load_data(train_indices,json_data,device,mydir)
    valid_data = load_data(valid_indices, json_data, device,mydir)
    test_data = load_data(test_indices, json_data, device,mydir)
    return train_data,valid_data, test_data



