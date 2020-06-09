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
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')
parser.add_argument('--use_target', action='store_true', default=False,
                    help='whether to use target domain')
parser.add_argument('--record_folder', type=str, default='record', metavar='N',
                    help='record folder')
parser.add_argument('--net', type=str, default='lenet', metavar='N',
                    help='backbone of the generator, lenet, resnet50, resnet101')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='direction to store checkpoints')
parser.add_argument('--load_checkpoint', type=str, default=None, metavar='N',
                    help='the checkpoint to load from')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate')
parser.add_argument('--max_epoch', type=int, default=500, metavar='N',
                    help='the number of training epoch')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N',
                    help='which optimizer to use')
parser.add_argument('--save_epoch', type=int, default=20, metavar='N',
                    help='when to save the model')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='manually set seed')
parser.add_argument('--target', type=str, default='mnistm', metavar='N',
                    help='target domain dataset')
parser.add_argument('--entropy_thr', type=float, default=0.5, metavar='N',
                    help='the threshold for the entropy of prediction')
parser.add_argument('--sigma', type=float, default=0.005, metavar='N',
                    help='the variance parameter for Gaussian function')
parser.add_argument('--beta', type=float, default=0.7, metavar='N',
                    help='the decay ratio for moving average')
parser.add_argument('--Lambda_global', type=float, default=20, metavar='N',
                    help='the trade-off parameter of losses')
parser.add_argument('--Lambda_local', type=float, default=0.01, metavar='N',
                    help='the trade-off parameter of losses')

args = parser.parse_args()

# define task-specific parameters
args.nfeat = 2048
args.nclasses = 10
args.ndomain = 5

print (args)

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def main():
    # define the training solver
    solver = Solver(args, target=args.target, learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, checkpoint_dir=args.checkpoint_dir, save_epoch=args.save_epoch)

    # define recording files
    record_num = 0
    record_train = '%s/%s_%s.txt' % (
        args.record_folder, args.target, record_num)
    record_test = '%s/%s_%s_test.txt' % (
        args.record_folder, args.target, record_num)
    while os.path.exists(record_train):
        record_num += 1
        record_train = '%s/%s_%s.txt' % (
            args.record_folder, args.target, record_num)
        record_test = '%s/%s_%s_test.txt' % (
            args.record_folder, args.target, record_num)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.record_folder):
        os.mkdir(args.record_folder)

    # train the model
    for t in range(args.max_epoch):
        print('Epoch: ', t)

        # setting: Multi-Source Domain Adaptation
        if args.use_target:
            num = solver.train_gcn_adapt(t, record_file=record_train)
        # setting: Domain Generalization
        else:
            num = solver.train_gcn_baseline(t, record_file=record_train)

        # test on target domain
        solver.test(t, record_file=record_test, save_model=args.save_model)

if __name__ == '__main__':
    main()
    os.system('watch nvidia-smi')