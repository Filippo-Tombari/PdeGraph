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

`pip install ipython numpy matplotlib torch imageio`

`pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+${CUDA}.html`

In the last command you have to replace `${CUDA}` with `cu116`,`cu117` or `cpu` according to your system.

Finally you can clone this repository running

`git clone https://github.com/Filippo-Tombari/PdeGraph.git`

and then move inside the folder with

`cd PdeGraph`

### Usage

The main file is `main.py` and it can be run with the command 

`python main.py`    

Moreover it is possible to specify the value of some parameters by adding them to the command line as follows

`python main.py --parameter_name parameter_value`

The following table shows the available parameters and their default values.

| Parameter           | Description                         | Default value |
|---------------------|-------------------------------------|---------------|
| `--example`         | example name: `AD`,`Stokes`         | `AD`          |
| `--dset_dir`        | dataset directory                   | `data`        |
| `--train_model`     | train or test                       | `False`       |
| `--dt`              | time step                           | `0.02`        |
| `--mlp_layers`      | number of hidden layers per MLP     | `2`           |
| `--hidden_channels` | ndimension of hidden units          | `32`          |
| `--mp_steps`        | number of message passing steps     | `12`          |
| `--seed`            | random seed                         | `10`          |
| `--batch_size`      | batch size                          | `25`          |
| `--epochs`          | number of training epochs           | `1500`        |
| `--lr`              | learning rate                       | `0.001`       |
| `--noise_var`       | training noise variance             | `1e-6`        |
| `--milestones`      | learning rate scheduler milestones  | `[500,1000]`  |
| `--w1`              | weight for loss 1                   | `1.0`         |
| `--w2`              | weight for loss 2                   | `0.0`         |
| `--save_plot`       | Save test simulation gif            | `True`        |

### Examples

The following examples are available:

- `AD`: Advection-Diffusion equation on a square domain with circular obstacle
- `Stokes`: Advection-Diffusion equation with Stokes advectiion field on a rectanguar domain with a bump

Data for the examples can be generated following the instructions in the `data` folder.

Otherwise the zip files contained the data used in the pretrained simulations can be downloaded at the following
link : https://polimi365-my.sharepoint.com/:f:/g/personal/10569815_polimi_it/EpvbKhXqEudIteCIkY50GMkB72VD0a7U4SZtpQFTdUDZFA?e=tuBs4M

After downloading the zip files, they have to be extracted in the `data` folder.