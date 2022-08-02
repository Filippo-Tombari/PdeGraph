import numpy as np
import torch
from torch_geometric.nn import norm


def buildconnectivity(mesh):
  cells = mesh.cells
  mesh.init(mesh.topology().dim()-1,0)
  edge_to_vertex = mesh.topology()(mesh.topology().dim()-1,0)
  for edge in range(mesh.num_edges()):
    if edge == 0:
      edge_index = np.expand_dims(edge_to_vertex(edge),axis=0)
    else:
      edge_index = np.append(edge_index,np.expand_dims(edge_to_vertex(edge),axis=0),axis=0 )
  return edge_index

def create_adj(cells,num_vertices):
  rows = cells.shape[0]
  cols = cells.shape[1]
  adj = np.zeros((num_vertices,num_vertices))
  for i in range(rows):
    for j in range(cols-1):
      adj[cells[i][j]][cells[i][j+1]] = 1
      adj[cells[i][j+1]][cells[i][j]] = 1
  adj = adj + np.eye(num_vertices)
  return adj

def initialize_weights(edge_index, mesh):
  edge_weights = torch.zeros((edge_index.shape[1],3))
  for k in range(edge_index.shape[1]):
    i = edge_index[0,k]
    j = edge_index[1,k]
    edge_weights[k,0:2] = torch.from_numpy(mesh.coordinates()[i,:] - mesh.coordinates()[j,:])
    edge_weights[k,2] = np.linalg.norm(mesh.coordinates()[i,:] - mesh.coordinates()[j,:],ord = 2)

  return edge_weights

def train(net, fts_train, edge_index, edge_weights, loss, optimizer, model_chk_path, epochs, dt, early_stopping = 0):
  es = 0
  mse_min = 1000
  t = 1 # current epoch
  done = False
  while not done:
        train_loss = 0
        # training
        def closure():
          optimizer.zero_grad()
          # forward pass
          integrating = torch.stack([net.forward((u,edge_weights), edge_index) for u in fts_train[:]], axis = 0) 
          #integrating = torch.stack([gnn.inference(u, train_loader, device) for u in fts_train[:]], axis = 0)
          train_out = fts_train[[0]] + dt*integrating.cumsum(axis = 0) #u(t1) = u(t0) + int{t0,t1}phi(t)dt
          #train_out = [gnn.forward(u, edge_index_train, edge_weights_train)for u in fts_train[:]] 
          #train_out = torch.stack(train_out, axis = 0)
          train_loss = loss(train_out[:-1,:,0],fts_train[1:,:,0])
          # backpropagation
          train_loss.backward()
          return train_loss
        optimizer.step(closure)
        #scheduler.step(train_loss)
        with torch.no_grad():
          integrating = torch.stack([net.forward((u,edge_weights), edge_index) for u in fts_train[:]], axis = 0) 
          #integrating = torch.stack([gnn.inference(u, train_loader, device) for u in fts_train[:]], axis = 0)
          train_out = fts_train[[0]] + dt*integrating.cumsum(axis = 0) #u(t1) = u(t0) + int{t0,t1}phi(t)dt
          #train_out = [gnn.forward(u, edge_index_train, edge_weights_train) for u in fts_train[:]]
          #train_out = torch.stack(train_out, axis = 0)
          train_loss = loss(train_out[:-1,:,0],fts_train[1:,:,0])
        # print rollout number and MSE for training and validation set at each epoch
      
        print(f"Rollout {t:1f}: MSE_train {train_loss.item() :6.6f}" )
        if train_loss < mse_min:
          mse_min = train_loss
          torch.save(net, model_chk_path)
          print('Saving model checkpoint')
          if early_stopping != 0:
            es = 0
        else:
          if early_stopping != 0:
            es += 1
        #stop the training after reaching the number of epochs
        t += 1
        if (t > epochs or (early_stopping!=0 and es == early_stopping)): 
          done = True


def forecast(u0, model, fts_train, edge_index, edge_attr, steps, dt):
  res = [u0]
  with torch.no_grad():
    for i in range(steps):
      res[-1][:,1] = fts_train[0,:,1]
      out = model.forward((res[-1],edge_attr),edge_index)
      res.append(res[-1]+dt*out)

  return torch.stack(res, axis = 0)
