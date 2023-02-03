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
    :return:            torch.Tensor with shape [n_edges,3].
    '''
    edge_weights = torch.zeros((edge_index.shape[1],3))
    for k in range(edge_index.shape[1]):
        i = edge_index[0,k]
        j = edge_index[1,k]
        edge_weights[k,0:2] = torch.from_numpy(mesh.coordinates()[i,:] - mesh.coordinates()[j,:])
        edge_weights[k,2] = np.linalg.norm(mesh.coordinates()[i,:] - mesh.coordinates()[j,:],ord = 2)

    return edge_weights

def get_bd(mesh_coords,bmesh_coords):
    '''Get the indices corresponding to boundary nodes
    :param      mesh_coords: numpy array with shape [mesh_nodes,2]
    :param      bmesh_coords: numpy array with shape [boundary_mesh_nodes,2]
    :return:    list with length boundary_mesh_nodes
    '''
    indices = []
    for i in range(mesh_coords.shape[0]):
      x1,x2 = mesh_coords[i]
      for j in range(bmesh_coords.shape[0]):
        y1,y2 = bmesh_coords[j]
        if (x1==y1 and x2==y2 and y1!=1.):
          indices.append(i)
          break
    return indices

def load_data(indices,json_data,device,mydir,problem):
    '''Creates a dataset from a json file selecting the keys contained in indices
    :param indices:     list of strings containing the keys to choose.
    :param json_data:   json file that contains all the simulations. Every key contains dictionary with the following keys:
                        -'mesh': name of the mesh corresponding to the key simulation.
                        -'traj': numpy.ndarray containing the trajectories of the key simulation.
    :param device:      either 'cuda' or 'cpu'.
    :param mydir:       directory in which the meshes and the json file are contained.
    :param problem:     name of the problem we want to solve.
    :return:            {'mesh': dolfin.cpp.mesh.Mesh, 'edge_index': torch.Tensor, 'edge_weights': torch.Tensor,
                        'trajs': torch.Tensor, "n_b_nodes": int}.
    '''
    meshes = []
    edge_indices = []
    edge_weights = []
    b_nodes = []
    in_nodes = []
    trajs = []
    if problem == 'AD':
        dt = 0.02
    else:
        dt = 0.01

    for i in indices:
        mesh = dolfin.cpp.mesh.Mesh(mydir + json_data[i]['mesh'] + ".xml")
        edge_index = torch.t(torch.from_numpy(buildconnectivity(mesh)).long()).to(device)
        meshes.append(mesh)
        edge_indices.append(edge_index)
        edge_weights.append(initialize_weights(edge_index, mesh).to(device))

        # get boundary nodes
        bmesh = dolfin.BoundaryMesh(mesh, "exterior", True)

        traj = torch.Tensor(json_data[i]['traj']).float().unsqueeze(dim=2).to(device)
        # Create a dummy feature to indicate the boundary nodes
        bindex = torch.zeros(traj.shape).to(device)
        # get boundary nodes
        b_nodes.append(get_bd(mesh.coordinates(), bmesh.coordinates()))
        in_nodes.append(list(set(range(mesh.coordinates().shape[0])).symmetric_difference(set(b_nodes[-1]))))
        bindex[:, b_nodes[-1], 0] = 1


        # Create a tensor containg the timesteps for each trajectory
        dt_tensor = torch.stack([torch.full((traj.shape[1], 1), dt * j) for j in range(traj.shape[0])]).to(device)
        traj = torch.cat((traj, dt_tensor, bindex), 2)
        trajs.append(traj)

    data = {'mesh': meshes, 'edge_index': edge_indices, 'edge_weights': edge_weights, 'trajs': trajs,
            "b_nodes": b_nodes, 'in_nodes': in_nodes}
    return data


def create_dataset(device, problem, train_size):
    '''
    Creates training, validation and test set

    :param  device:     either 'cuda' or 'cpu'.
    :param  problem:    name of the problem we want to solve. It is the same name of the directory
                        in which the data are placed
    :param train_size:  size of the training set
    :return:            tuple of dictionaries (see load_data)
    '''
    mydir = os.getcwd() + f'/data/{problem}/'

    with open(mydir+'data.json', 'r') as f:
        json_data = json.loads(f.read())

    random.seed(10)
    indices = list(json_data.keys())
    random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_data = load_data(train_indices,json_data,device,mydir,problem)
    test_data = load_data(test_indices, json_data, device,mydir,problem)
    return train_data, test_data