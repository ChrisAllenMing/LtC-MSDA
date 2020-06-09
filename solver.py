from __future__ import print_function
import os, sys
sys.path.append('./gcn')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.build_gen import *
from datasets.dataset_read import dataset_read
from gcn.models import GCN

# The solver for training and testing LtC-MSDA
class Solver(object):
    def __init__(self, args, batch_size=128,
                 target='mnistm', learning_rate=0.0002, interval=10, optimizer='adam',
                 checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.target = target
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.interval = interval
        self.lr = learning_rate
        self.best_correct = 0
        self.args = args
        if self.args.use_target:
            self.ndomain = self.args.ndomain
        else:
            self.ndomain = self.args.ndomain - 1

        # load source and target domains
        self.datasets, self.dataset_test, self.dataset_size = dataset_read(target, self.batch_size)
        self.niter = self.dataset_size / self.batch_size
        print('Dataset loaded!')

        # define the feature extractor and GCN-based classifier
        self.G = Generator(self.args.net)
        self.GCN = GCN(nfeat=args.nfeat, nclasses=args.nclasses)
        self.G.cuda()
        self.GCN.cuda()
        print('Model initialized!')

        if self.args.load_checkpoint is not None:
            self.state = torch.load(self.args.load_checkpoint)
            self.G.load_state_dict(self.state['G'])
            self.GCN.load_state_dict(self.state['GCN'])
            print('Model load from: ', self.args.load_checkpoint)

        # initialize statistics (prototypes and adjacency matrix)
        if self.args.load_checkpoint is None:
            self.mean = torch.zeros(args.nclasses * self.ndomain, args.nfeat).cuda()
            self.adj = torch.zeros(args.nclasses * self.ndomain, args.nclasses * self.ndomain).cuda()
            print('Statistics initialized!')
        else:
            self.mean = self.state['mean'].cuda()
            self.adj = self.state['adj'].cuda()
            print('Statistics loaded!')

        # define the optimizer
        self.set_optimizer(which_opt=optimizer, lr=self.lr)
        print('Optimizer defined!')

    # optimizer definition
    def set_optimizer(self, which_opt='sgd', lr=0.001, momentum=0.9):
        if which_opt == 'sgd':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)
            self.opt_gcn = optim.SGD(self.GCN.parameters(),
                                     lr=lr, weight_decay=0.0005,
                                     momentum=momentum)
        elif which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)
            self.opt_gcn = optim.Adam(self.GCN.parameters(),
                                      lr=lr, weight_decay=0.0005)

    # empty gradients
    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_gcn.zero_grad()

    # compute the discrepancy between two probabilities
    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    # compute the Euclidean distance between two tensors
    def euclid_dist(self, x, y):
        x_sq = (x ** 2).mean(-1)
        x_sq_ = torch.stack([x_sq] * y.size(0), dim = 1)
        y_sq = (y ** 2).mean(-1)
        y_sq_ = torch.stack([y_sq] * x.size(0), dim = 0)
        xy = torch.mm(x, y.t()) / x.size(-1)
        dist = x_sq_ + y_sq_ - 2 * xy

        return dist

    # construct the extended adjacency matrix
    def construct_adj(self, feats):
        dist = self.euclid_dist(self.mean, feats)
        sim = torch.exp(-dist / (2 * self.args.sigma ** 2))
        E = torch.eye(feats.shape[0]).float().cuda()

        A = torch.cat([self.adj, sim], dim = 1)
        B = torch.cat([sim.t(), E], dim = 1)
        gcn_adj = torch.cat([A, B], dim = 0)

        return gcn_adj

    # assign pseudo labels to target samples
    def pseudo_label(self, logit, feat):
        pred = F.softmax(logit, dim=1)
        entropy = (-pred * torch.log(pred)).sum(-1)
        label = torch.argmax(logit, dim=-1).long()

        mask = (entropy < self.args.entropy_thr).float()
        index = torch.nonzero(mask).squeeze(-1)
        feat_ = torch.index_select(feat, 0, index)
        label_ = torch.index_select(label, 0, index)

        return feat_, label_

    # update prototypes and adjacency matrix
    def update_statistics(self, feats, labels, epsilon=1e-5):
        curr_mean = list()
        num_labels = 0

        for domain_idx in range(self.ndomain):
            tmp_feat = feats[domain_idx]
            tmp_label = labels[domain_idx]
            num_labels += tmp_label.shape[0]

            if tmp_label.shape[0] == 0:
                curr_mean.append(torch.zeros((self.args.nclasses, self.args.nfeat)).cuda())
            else:
                onehot_label = torch.zeros((tmp_label.shape[0], self.args.nclasses)).scatter_(1, tmp_label.unsqueeze(
                    -1).cpu(), 1).float().cuda()
                domain_feature = tmp_feat.unsqueeze(1) * onehot_label.unsqueeze(-1)
                tmp_mean = domain_feature.sum(0) / (onehot_label.unsqueeze(-1).sum(0) + epsilon)

                curr_mean.append(tmp_mean)

        curr_mean = torch.cat(curr_mean, dim = 0)
        curr_mask = (curr_mean.sum(-1) != 0).float().unsqueeze(-1)
        self.mean = self.mean.detach() * (1 - curr_mask) + (
                    self.mean.detach() * self.args.beta + curr_mean * (1 - self.args.beta)) * curr_mask
        curr_dist = self.euclid_dist(self.mean, self.mean)
        self.adj = torch.exp(-curr_dist / (2 * self.args.sigma ** 2))

        # compute local relation alignment loss
        loss_local = ((((curr_mean - self.mean) * curr_mask) ** 2).mean(-1)).sum() / num_labels

        return loss_local

    # compute global relation alignment loss
    def adj_loss(self):
        adj_loss = 0

        for i in range(self.ndomain):
            for j in range(self.ndomain):
                adj_ii = self.adj[i * self.args.nclasses:(i + 1) * self.args.nclasses,
                         i * self.args.nclasses:(i + 1) * self.args.nclasses]
                adj_jj = self.adj[j * self.args.nclasses:(j + 1) * self.args.nclasses,
                         j * self.args.nclasses:(j + 1) * self.args.nclasses]
                adj_ij = self.adj[i * self.args.nclasses:(i + 1) * self.args.nclasses,
                         j * self.args.nclasses:(j + 1) * self.args.nclasses]

                adj_loss += ((adj_ii - adj_jj) ** 2).mean()
                adj_loss += ((adj_ij - adj_ii) ** 2).mean()
                adj_loss += ((adj_ij - adj_jj) ** 2).mean()

        adj_loss /= (self.ndomain * (self.ndomain - 1) / 2 * 3)

        return adj_loss

    # per epoch training in a Domain Generalization setting
    def train_gcn_baseline(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.GCN.train()

        for batch_idx, data in enumerate(self.datasets):
            # get the source batches
            img_s = list()
            label_s = list()
            stop_iter = False
            for domain_idx in range(self.ndomain):
                tmp_img = data['S' + str(domain_idx + 1)].cuda()
                tmp_label = data['S' + str(domain_idx + 1) + '_label'].long().cuda()
                img_s.append(tmp_img)
                label_s.append(tmp_label)

                if tmp_img.size()[0] < self.batch_size:
                    stop_iter = True

            if stop_iter:
                break

            self.reset_grad()

            # get feature embeddings
            feats = list()
            for domain_idx in range(self.ndomain):
                tmp_img = img_s[domain_idx]
                tmp_feat = self.G(tmp_img)
                feats.append(tmp_feat)

            # Update the global mean and adjacency matrix
            loss_local = self.update_statistics(feats, label_s)
            feats = torch.cat(feats, dim=0)
            labels = torch.cat(label_s, dim=0)

            # add query samples to the domain graph
            gcn_feats = torch.cat([self.mean, feats], dim=0)
            gcn_adj = self.construct_adj(feats)

            # output classification logit with GCN
            gcn_logit = self.GCN(gcn_feats, gcn_adj)

            # define GCN classification losses
            domain_logit = gcn_logit[:self.mean.shape[0], :]
            domain_label = torch.cat([torch.arange(self.args.nclasses)] * self.ndomain, dim=0)
            domain_label = domain_label.long().cuda()
            loss_cls_dom = criterion(domain_logit, domain_label)

            query_logit = gcn_logit[self.mean.shape[0]:, :]
            loss_cls_src = criterion(query_logit, labels)

            loss_cls = loss_cls_src + loss_cls_dom

            # define relation alignment losses
            loss_global = self.adj_loss() * self.args.Lambda_global
            loss_local = loss_local * self.args.Lambda_local
            loss_relation = loss_local + loss_global

            loss = loss_cls + loss_relation

            # back-propagation
            loss.backward()
            self.opt_gcn.step()
            self.opt_g.step()

            # record training information
            if epoch == 0 and batch_idx == 0:
                record = open(record_file, 'a')
                record.write(str(self.args))
                record.close()

            if batch_idx % self.interval == 0:
                print(
                    'Train Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                    '\tLoss_global: {:.5f}\tLoss_local: {:.5f}'.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter,
                        loss_cls_dom.item(), loss_cls_src.item(), loss_global.item(), loss_local.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write(
                        '\nTrain Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                        '\tLoss_global: {:.5f}\tLoss_local: {:.5f}'.format(
                            epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter,
                            loss_cls_dom.item(), loss_cls_src.item(), loss_global.item(), loss_local.item()))
                    record.close()

        return batch_idx

    # per epoch training in a Multi-Source Domain Adaptation setting
    def train_gcn_adapt(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.GCN.train()

        for batch_idx, data in enumerate(self.datasets):
            # get the source batches
            img_s = list()
            label_s = list()
            stop_iter = False
            for domain_idx in range(self.ndomain - 1):
                tmp_img = data['S' + str(domain_idx + 1)].cuda()
                tmp_label = data['S' + str(domain_idx + 1) + '_label'].long().cuda()
                img_s.append(tmp_img)
                label_s.append(tmp_label)

                if tmp_img.size()[0] < self.batch_size:
                    stop_iter = True

            if stop_iter:
                break

            # get the target batch
            img_t = data['T'].cuda()
            if img_t.size()[0] < self.batch_size:
                break

            self.reset_grad()

            # get feature embeddings
            feat_list = list()
            for domain_idx in range(self.ndomain - 1):
                tmp_img = img_s[domain_idx]
                tmp_feat = self.G(tmp_img)
                feat_list.append(tmp_feat)

            feat_t = self.G(img_t)
            feat_list.append(feat_t)
            feats = torch.cat(feat_list, dim=0)
            labels = torch.cat(label_s, dim=0)

            # add query samples to the domain graph
            gcn_feats = torch.cat([self.mean, feats], dim=0)
            gcn_adj = self.construct_adj(feats)

            # output classification logit with GCN
            gcn_logit = self.GCN(gcn_feats, gcn_adj)

            # predict the psuedo labels for target domain
            feat_t_, label_t_ = self.pseudo_label(gcn_logit[-feat_t.shape[0]:, :], feat_t)
            feat_list.pop()
            feat_list.append(feat_t_)
            label_s.append(label_t_)

            # update the statistics for source and target domains
            loss_local = self.update_statistics(feat_list, label_s)

            # define GCN classification losses
            domain_logit = gcn_logit[:self.mean.shape[0], :]
            domain_label = torch.cat([torch.arange(self.args.nclasses)] * self.ndomain, dim=0)
            domain_label = domain_label.long().cuda()
            loss_cls_dom = criterion(domain_logit, domain_label)

            query_logit = gcn_logit[self.mean.shape[0]:, :]
            loss_cls_src = criterion(query_logit[:-feat_t.shape[0]], labels)

            target_logit = query_logit[-feat_t.shape[0]:]
            target_prob = F.softmax(target_logit, dim=1)
            loss_cls_tgt = (-target_prob * torch.log(target_prob + 1e-8)).mean()

            loss_cls = loss_cls_dom + loss_cls_src + loss_cls_tgt

            # define relation alignment losses
            loss_global = self.adj_loss() * self.args.Lambda_global
            loss_local = loss_local * self.args.Lambda_local
            loss_relation = loss_local + loss_global

            loss = loss_cls + loss_relation

            # back-propagation
            loss.backward(retain_graph = True)
            self.opt_gcn.step()
            self.opt_g.step()

            # record training information
            if epoch ==0 and batch_idx==0:
                record = open(record_file, 'a')
                record.write(str(self.args)+'\n')
                record.close()

            if batch_idx % self.interval == 0:
                print(
                    'Train Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                    '\tLoss_cls_target: {:.5f}\tLoss_global: {:.5f}\tLoss_local: {:.5f}'.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter,
                        loss_cls_dom.item(), loss_cls_src.item(), loss_cls_tgt.item(),
                        loss_global.item(), loss_local.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write(
                        '\nTrain Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                        '\tLoss_cls_target: {:.5f}\tLoss_global: {:.5f}\tLoss_local: {:.5f}'.format(
                            epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter,
                            loss_cls_dom.item(), loss_cls_src.item(), loss_cls_tgt.item(),
                            loss_global.item(), loss_local.item()))
                    record.close()

        return batch_idx

    # per epoch test on target domain
    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.GCN.eval()

        test_loss = 0
        correct = 0
        size = 0

        for batch_idx, data in enumerate(self.dataset_test):
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.long().cuda()

            feat = self.G(img)

            gcn_feats = torch.cat([self.mean, feat], dim=0)
            gcn_adj = self.construct_adj(feat)
            gcn_logit = self.GCN(gcn_feats, gcn_adj)
            output = gcn_logit[self.mean.shape[0]:, :]

            test_loss += -F.nll_loss(output, label).item()
            pred = output.max(1)[1]
            k = label.size()[0]
            correct += pred.eq(label).cpu().sum()
            size += k

        test_loss = test_loss / size

        if correct > self.best_correct:
            self.best_correct = correct
            if save_model:
                best_state = {'G': self.G.state_dict(), 'GCN': self.GCN.state_dict(), 'mean': self.mean.cpu(),
                              'adj': self.adj.cpu(), 'epoch': epoch}
                torch.save(best_state, os.path.join(self.checkpoint_dir, 'best_model.pth'))

        # save checkpoint
        if save_model and epoch % self.save_epoch == 0:
            state = {'G': self.G.state_dict(), 'GCN': self.GCN.state_dict(), 'mean': self.mean.cpu(),
                     'adj': self.adj.cpu()}
            torch.save(state, os.path.join(self.checkpoint_dir, 'epoch_' + str(epoch) + '.pth'))

        # record test information
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Best Accuracy: {}/{} ({:.4f}%)  \n'.format(
                test_loss, correct, size, 100. * float(correct) / size, self.best_correct, size,
                                          100. * float(self.best_correct) / size))

        if record_file:
            if epoch == 0:
                record = open(record_file, 'a')
                record.write(str(self.args))
                record.close()

            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write(
                '\nEpoch {:>3} Average loss: {:.5f}, Accuracy: {:.5f}, Best Accuracy: {:.5f}'.format(
                    epoch, test_loss, 100. * float(correct) / size, 100. * float(self.best_correct) / size))
            record.close()