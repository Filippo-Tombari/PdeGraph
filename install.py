import os
from IPython.display import clear_output
import torch

def pytorchgeo():
  try:
     import torch_geometric
  except ImportError:
    os.system("pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-%s.html" % torch.__version__)
    os.system("pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-%s.html" % torch.__version__)
    os.system("pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-%s.html" % torch.__version__)
    os.system("pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-%s.html" % torch.__version__)
    os.system("pip install torch-geometric") 
    os.system("pip install ogb")
    os.system("pip install umap")
    clear_output()
  print("Pytorch geometric installed.")


def fenics():
  try:
    import dolfin
  except ImportError:
    os.system("wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub | sudo apt-key add -")
    os.system('wget "https://fem-on-colab.github.io/releases/fenics-install.sh" -O "/tmp/fenics-install.sh" && bash "/tmp/fenics-install.sh"')
    clear_output()
  print("FEniCS installed.")
