import argparse
import torch
import dolfin

def str2bool(v):
    # Code from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    ''' Converts a string to a boolean value
    :param v: string
    :return: boolean
    '''

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def asfunction(vector, mesh):
    ''' Views a vector as a Dolfin function
    :param vector: torch tensor or numpy array
    :param mesh: dolfin.cpp.mesh.Mesh object
    :return: dolfin.function.function.Function object
    '''
    if(isinstance(vector, torch.Tensor)):
        return asfunction(vector.detach().cpu().numpy(), mesh)
    else:
        V = dolfin.function.functionspace.FunctionSpace(mesh, 'CG', 1)
        uv = dolfin.function.function.Function(V)
        uv.vector()[:] = vector[dolfin.cpp.fem.dof_to_vertex_map(V)]
    return uv

