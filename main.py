import argparse
import numpy as np
import torch
from learn_model import Learner
from utils import str2bool

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    learner = Learner(args)

    if args.train_model: learner.train()
    
    learner.forecast(args.save_plot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Neural Networks solver for PDEs')

    parser.add_argument('--train_model', default=False, type=str2bool, help='train or test')

    # Net Parameters
    parser.add_argument('--mlp_layers', default=2, type=int, help='number of hidden layers per MLP')
    parser.add_argument('--hidden_channels', default=32, type=int, help='dimension of hidden units')
    parser.add_argument('--mp_steps', default=10, type=int, help='number of message passing steps')

    # Training Parameters
    parser.add_argument('--seed', default=10, type=int, help='random seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--noise_var', default=1e-6, type=float, help='training noise variance')
    parser.add_argument('--batch_size', default=25, type=int, help='training batch size')
    parser.add_argument('--epochs', default=2000, type=int, help='training iterations')
    parser.add_argument('--milestones', default=[500, 1000], nargs='+', type=int, help='learning rate scheduler milestones')

    # Save Parameters
    parser.add_argument('--save_plot', default=True, type=str2bool, help='Save test simulation gif')

    args = parser.parse_args()

    main(args)



