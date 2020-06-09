from __future__ import print_function
import os, sys
sys.path.append('./datasets')
sys.path.append('./model')
sys.path.append('./utils')
sys.path.append('./gcn')

import pdb
import argparse
import numpy as np
import torch
from torch.autograd import Variable

from model.build_gen import *
from datasets.dataset_read import dataset_read
from solver import Solver


# training settings
parser = argparse.ArgumentParser(description='Training for LtC-MSDA')
parser.add_argument('--net', type=str, default='lenet', metavar='N',
                    help='backbone of the generator, lenet, resnet50, resnet101')
parser.add_argument('--load_checkpoint', type=str, default=None, metavar='N',
                    help='the checkpoint to load from')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='manually set seed')
parser.add_argument('--target', type=str, default='mnistm', metavar='N',
                    help='target domain dataset')
parser.add_argument('--sigma', type=float, default=0.005, metavar='N',
                    help='the variance parameter for Gaussian function')

args = parser.parse_args()

# define task-specific parameters
args.use_target = True
args.nfeat = 2048
args.nclasses = 10
args.ndomain = 5

print (args)

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def main():
    # define the test solver
    solver = Solver(args, target=args.target)

    # test on target domain
    solver.test(0)

if __name__ == '__main__':
    main()
    os.system('watch nvidia-smi')