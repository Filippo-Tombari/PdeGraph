import os
import argparse
import numpy as np
import torch
from learn_model import Learner
from utils import str2bool

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.save_plot and not os.path.exists('images'):
        os.makedirs('images')

    learner = Learner(args)

    if args.train_model: learner.train()
    
    learner.forecast(args.save_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Neural Networks solver for PDEs')

    # Problem
    parser.add_argument('--example', default='AD', type=str, help='example name')
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--train_model', default=False, type=str2bool, help='train or test')
    parser.add_argument('--train_size', default=80, type=int, help='training set size')

    # Net Parameters
    parser.add_argument('in_node', action='store_const', const=3)
    parser.add_argument('in_edge', action='store_const', const=3)
    parser.add_argument('out_channels', action='store_const', const=1)
    parser.add_argument('--mlp_layers', default=2, type=int, help='number of hidden layers per MLP')
    parser.add_argument('--hidden_channels', default=32, type=int, help='dimension of hidden units')
    parser.add_argument('--mp_steps', default=12, type=int, help='number of message passing steps')

    # Training Parameters
    parser.add_argument('--seed', default=10, type=int, help='random seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--noise_var', default=1e-6, type=float, help='training noise variance')
    parser.add_argument('--batch_size', default=25, type=int, help='training batch size')
    parser.add_argument('--epochs', default=1500, type=int, help='training iterations')
    parser.add_argument('--milestones', default=[500, 1000], nargs='+', type=int, help='learning rate scheduler milestones')
    parser.add_argument('--w1', default=1., type=float, help='weight for loss 1')
    parser.add_argument('--w2', default=0., type=int, help='weight for loss 2')

    # Save Parameters
    parser.add_argument('--save_plot', default=True, type=str2bool, help='Save test simulation gif')

    args = parser.parse_args()
    if args.example != 'AD' and args.example != 'Stokes':
        raise ValueError('Example name should be AD or Stokes')
    main(args)



