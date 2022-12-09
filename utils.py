import argparse
import torch
import dolfin

def str2bool(v):
    # Code from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
