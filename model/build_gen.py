import svhn2mnist
import usps
import syn2gtrsb
#import syndig2svhn

import lenet
import resnet
import alexnet

def Generator(net):

    if net == 'lenet':
        return lenet.Feature()
    if net == 'resnet18':
        return resnet.resnet18(pretrained=True)
    if net == 'resnet50':
        return resnet.resnet50(pretrained=True)
    if net == 'resnet101':
        return resnet.resnet101(pretrained=True)
    if net == 'alexnet':
        return alexnet.alexnet(pretrained=True)



def Classifier(net, feat, nclass):

    if net == 'lenet':
        return lenet.Predictor()
    if net == 'resnet18' or net == 'resnet50' or net == 'resnet101':
        return resnet.Predictor(feat, nclass)
    if net == 'alexnet':
        return alexnet._netC_alexnet(nclasses=nclass)


