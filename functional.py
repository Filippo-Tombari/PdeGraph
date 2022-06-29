import torch
import numpy as np
import dolfin

class Sparse(torch.nn.Module):
    def __init__(self, mask):
        super(Sparse, self).__init__()
        self.loc = np.nonzero(mask)
        self.in_d, self.out_d = mask.shape
        self.weight = torch.nn.Parameter(torch.zeros(len(self.loc[0])))
        self.bias = torch.nn.Parameter(torch.zeros(self.out_d))
        self.device = torch.device("cpu")
        
    def cuda(self):
        with torch.no_grad():
          self.weight = torch.nn.Parameter(self.weight.cuda())
          self.bias = torch.nn.Parameter(self.bias.cuda())
          self.device = torch.device("cuda:0")
    
    def forward(self, x):
        return self.bias + x.mm(self.W())
    
    def W(self):
        if torch.cuda.is_available():
          self.cuda()
        W = torch.zeros(self.in_d, self.out_d, dtype = self.weight.dtype).to(self.device)
        W[self.loc] = self.weight
        return W


class Operator(Sparse):
    def __init__(self, matrix):
        matrix[np.abs(matrix)<1e-10] = 0
        super(Operator, self).__init__(matrix)
        self.weight = torch.nn.Parameter(torch.tensor(matrix[np.nonzero(matrix)]).to(self.device))
        self.requires_grad_(False)
        
    def cuda(self):
        super(Operator, self).cuda()
        self.requires_grad_(False)


class Bilinear(Operator):
    def __init__(self, mesh, operator, obj = 'CG', degree = 1):
        W = dolfin.function.functionspace.FunctionSpace(mesh, obj, degree)
        if(degree == 1):
            perm = np.ndarray.astype(dolfin.cpp.fem.vertex_to_dof_map(W), 'int')
        else:
            perm = np.arange(mesh.num_cells())
        v1, v2 = dolfin.function.argument.TrialFunction(W), dolfin.function.argument.TestFunction(W)
        M = dolfin.fem.assembling.assemble(operator(v1, v2)).array()[:, perm][perm, :]
        super(Bilinear, self).__init__(M)
        
    def forward(self, x):
        return x[0].mm(self.W().mm(x[1].T))  
        
class Norm(Bilinear):
    def forward(self, x):
        return (x.mm(self.W())*x).sum(axis = -1).sqrt()   
        
class L2(Norm):
    def __init__(self, mesh, obj = 'CG', degree = 1):
        def operator(u,v):
            return u*v*dolfin.dx
        super(L2, self).__init__(mesh, operator, obj, degree)

def asfunction(vector, mesh):
  if(isinstance(vector, torch.Tensor)):
    return asfunction(vector.detach().cpu().numpy(), mesh)
  else:
    V = dolfin.function.functionspace.FunctionSpace(mesh, 'CG', 1)
    uv = dolfin.function.function.Function(V) 
    uv.vector()[:] = vector[dolfin.cpp.fem.dof_to_vertex_map(V)]
  return uv

def asfield(matrix, mesh):
  ux, uy = matrix.T
  uvx, uvy = asfunction(ux, mesh), asfunction(uy, mesh)
  return dolfin.fem.projection.project(uvx*dolfin.function.constant.Constant((1.0, 0.0))+ uvy*dolfin.function.constant.Constant((0.0, 1.0)),
                                       dolfin.function.functionspace.VectorFunctionSpace(mesh, 'CG', degree = 1, dim = 2))

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

  
plot = dolfin.common.plotting.plot

