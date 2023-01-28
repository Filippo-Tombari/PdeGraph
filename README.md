# PdeGraph
Graph Neural Network solver for parameter dependent PDEs
### Getting started
The following instructions refer to a Linux environment. The installation of Anaconda is recommended, e.g. with the commands

`wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh`

`bash Anaconda3-2022.10-Linux-x86_64.sh`

then you can create a virtual environment with

`conda create -n ENV_NAME python=3.8.5`

and then activate it with 

`conda activate ENV_NAME`

You can deactivate the virtual environment at any time running

`conda dectivate`

All the following python packages will be installed inside the virtual environment in order to avoid conflicts with other projects by running

`conda install -c conda-forge mshr`

`pip install ipython numpy matplotlib torch imageio matplotlib`

`pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+${CUDA}.html`

In the last command you have to replace `${CUDA}` with `cu116`,`cu117` or `cpu` according to your system.

Finally you can clone this repository running 

