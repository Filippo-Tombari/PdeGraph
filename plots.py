import os
import imageio
import matplotlib.pyplot as plt
import dolfin
from utils import asfunction

plot = dolfin.common.plotting.plot

def savegif(drawframe, frames, name, remove = True):
    ''' Creates and saves a gif'''
    filenames = []
    for i in range(frames):
        # plot frame
        drawframe(i)

        # create file name and append it to a list
        filename = f'{i}.png'
        filenames.append(filename)

        # save frame
        plt.savefig(filename)
        plt.close()
    # build gif
    with imageio.get_writer(name + '.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    if(remove):
        for filename in set(filenames):
            os.remove(filename)

def trajectorytogif(traj, dt, name, mesh):
    ''' Transforms a simulation trajectory into a gif
    :param traj:    torch.Tensor representing the trajectory.
    :param dt:      dt of the simulation.
    :param name:    name of the .gif file.
    :param mesh:    dolfin.cpp.mesh.Mesh object.
    :return:
    '''
    def drawframe(i):
        colorbar = plot(asfunction(traj[i,:,0], mesh), vmin = traj.min(), vmax = traj.max())
        plt.colorbar(colorbar, shrink = 0.75)
        plt.title("T = %.2f" % (dt*i))
        plt.axis("off")
    savegif(drawframe, frames = len(traj), name = name)
