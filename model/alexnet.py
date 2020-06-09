import torch, utils
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import math

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

"""
Generator network
"""
class _netG(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netG, self).__init__()
        
        self.ndim = opt.ndf*4
        self.ngf = opt.ngf
        self.nz = opt.nz
        self.gpu = opt.gpu
        self.nclasses = nclasses
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.ndim + self.nz + nclasses + 1, self.ngf*8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):   
        batchSize = input.size()[0]
        input = input.view(-1, self.ndim+self.nclasses+1, 1, 1)
        noise = torch.FloatTensor(batchSize, self.nz, 1, 1).normal_(0, 1)    
        if self.gpu>=0:
            noise = noise.cuda()
        noisev = Variable(noise)
        output = self.main(torch.cat((input, noisev),1))

        return output

"""
Discriminator network
"""
class _netD(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netD, self).__init__()
        
        self.ndf = opt.ndf
        self.feature = nn.Sequential(
            nn.Conv2d(3, self.ndf, 3, 1, 1),            
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(self.ndf, self.ndf*2, 3, 1, 1),         
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),        

            nn.Conv2d(self.ndf*2, self.ndf*4, 3, 1, 1),           
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(self.ndf*4, self.ndf*2, 3, 1, 1),           
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4,4)           
        )

        self.classifier_c = nn.Sequential(nn.Linear(self.ndf*2, nclasses))              
        self.classifier_s = nn.Sequential(
        						nn.Linear(self.ndf, 1),
        						nn.Sigmoid())
        self.classifier_t = nn.Sequential(nn.Linear(self.ndf*2, self.ndf))

    def forward(self, input):       
        output = self.feature(input)
        output_c = self.classifier_c(output.view(-1, self.ndf * 2))
        output_t = self.classifier_t(output.view(-1, self.ndf * 2))
        output_s = self.classifier_s(output_t)
        output_s = output_s.view(-1)
        return output_s, output_c, output_t

"""
Feature extraction network
"""
class _netF(nn.Module):
    def __init__(self, opt):
        super(_netF, self).__init__()

        
        self.ndf = opt.ndf
        self.nz = opt.nz
        self.gpu = opt.gpu

        self.feature = nn.Sequential(
            nn.Conv2d(3, self.ndf, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(self.ndf, self.ndf, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
                    
            nn.Conv2d(self.ndf, self.ndf*2, 5, 1,0),
            nn.ReLU(inplace=True)
        )

        self.mean = nn.Sequential(nn.Linear(self.ndf*2, self.ndf*2))              
        self.std = nn.Sequential(nn.Linear(self.ndf*2, self.ndf*2))    

    def forward(self, input):  
        batchSize = input.size()[0] 
        output = self.feature(input)

        mean_vector = self.mean(output.view(-1, 2*self.ndf))
        std_vector = self.std(output.view(-1, 2*self.ndf))
        if self.gpu>=0:
            mean_vector = mean_vector.cuda()
            std_vector = std_vector.cuda()

        return output.view(-1, 2*self.ndf), mean_vector, std_vector

"""
Classifier network
"""
class _netC(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netC, self).__init__()
        self.ndf = opt.ndf
        self.main = nn.Sequential(          
            nn.Linear(4*self.ndf, 2*self.ndf),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.ndf, nclasses),                         
        )
        self.soft = nn.Sequential(nn.Sigmoid())

    def forward(self, input):      
        output_logit = self.main(input)
        output = self.soft(output_logit)
        return output_logit, output

"""
Feature extractor: Alexnet
"""
class _netF_alexnet(nn.Module):
    def __init__(self, groups_ = 1):
        super(_netF_alexnet, self).__init__()

        # self.ndf = opt.ndf
        # self.gpu = opt.gpu

        self.conv = nn.Sequential()
        self.conv.add_module('conv1',nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=0))
        self.conv.add_module('relu1',nn.ReLU(inplace=True))
        #self.conv.add_module('bn1_s1',nn.BatchNorm2d(96))
        # self.conv.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.conv.add_module('pool1',nn.MaxPool2d(kernel_size=3, stride=2))

        
        self.conv.add_module('conv2',nn.Conv2d(64, 192, kernel_size=5, padding=2, groups=groups_))
        self.conv.add_module('relu2',nn.ReLU(inplace=True))
        #self.conv.add_module('bn2_s1',nn.BatchNorm2d(256))
        # self.conv.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.conv.add_module('pool2',nn.MaxPool2d(kernel_size=3, stride=2))

        
        self.conv.add_module('conv3',nn.Conv2d(192, 384, kernel_size=3, padding=1))
        self.conv.add_module('relu3',nn.ReLU(inplace=True))
        # self.conv.add_module('bn3_s1',nn.BatchNorm2d(384))
        
        self.conv.add_module('conv4',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=groups_))
        # self.conv.add_module('bn4_s1',nn.BatchNorm2d(384))
        self.conv.add_module('relu4',nn.ReLU(inplace=True))
        
        self.conv.add_module('conv5',nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=groups_))
        # self.conv.add_module('bn5_s1',nn.BatchNorm2d(256))
        self.conv.add_module('relu5',nn.ReLU(inplace=True))
        self.conv.add_module('pool5',nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6',nn.Linear(256*6*6, 4096))
        self.fc6.add_module('relu6',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6',nn.Dropout(p=0.5))
        
        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7',nn.Linear(4096,4096))
        self.fc7.add_module('relu7',nn.ReLU(inplace=True))
        self.fc7.add_module('drop7',nn.Dropout(p=0.5))
        
        # self.mean = nn.Sequential()
        # self.mean.add_module('fc_mean',nn.Linear(4096, self.ndf*2))
        #
        # self.std = nn.Sequential()
        # self.std.add_module('fc_std', nn.Linear(4096, self.ndf*2))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)

        tmp_dict = {}
        for k,v in pretrained_dict.items():
            if (k == "state_dict"):
                tmp_dict = v

        # get pretrained parameters
        param_list = []
        for k,v in pretrained_dict.items():
            if ('features' in k or '1' in k or '4' in k):
                param_list.append(v)

        key_list = []
        for k in model_dict:
            if ('conv' in k or 'fc6' in k or 'fc7' in k):
                key_list.append(k)

        # get parameter dict
        param_dict = {}
        for i in range(len(key_list)):
            print (key_list[i])
            param_dict[key_list[i]] = param_list[i] 
        
        model_dict.update(param_dict)
        self.load_state_dict(model_dict)

    def forward(self, inputs):

        output = self.conv(inputs)
        embedding = output.view(output.size(0), 6*6*256)

        output = self.fc6(embedding)
        output = self.fc7(output)

        # mean_vector = self.mean(output)
        # std_vector = self.std(output)
        # if self.gpu>=0:
        #     mean_vector = mean_vector.cuda()
        #     std_vector = std_vector.cuda()

        # return output, mean_vector, std_vector

        return output

"""
Classifier: Alexnet
"""
class _netC_alexnet(nn.Module):
    def __init__(self, nclasses):
        super(_netC_alexnet, self).__init__()
        
        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8',nn.Linear(4096, nclasses))

        self.soft = nn.Sequential(nn.Sigmoid())

    def forward(self, embedding):      
        output_logit = self.classifier(embedding)
        # output = self.soft(output_logit)
        return output_logit

"""
Generator: Alexnet
"""
class _netG_alexnet(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netG_alexnet, self).__init__()
        
        # self.ndim = opt.ndf*4
        self.ndim = 4096
        self.ngf = opt.ngf
        self.nz = opt.nz
        self.gpu = opt.gpu
        self.nclasses = nclasses
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.ndim + self.nz + nclasses + 1, self.ngf*8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*2, self.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):   
        batchSize = input.size()[0]
        input = input.view(-1, self.ndim+self.nclasses+1, 1, 1)
        noise = torch.FloatTensor(batchSize, self.nz, 1, 1).normal_(0, 1)    
        if self.gpu>=0:
            noise = noise.cuda()
        noisev = Variable(noise)
        output = self.main(torch.cat((input, noisev),1))

        return output

"""
Discriminator: Alexnet
"""
class _netD_alexnet(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netD_alexnet, self).__init__()
        
        self.ndf = opt.ndf

        self.feature = nn.Sequential(
            nn.Conv2d(3, self.ndf, 3, 1, 1),            
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(self.ndf, self.ndf*2, 3, 1, 1),         
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(self.ndf*2, self.ndf*2, 3, 1, 1),         
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(self.ndf*2, self.ndf*4, 3, 1, 1),           
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(self.ndf*4, self.ndf*2, 3, 1, 1),           
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(4,4)
        )

        self.fc= nn.Sequential(
            nn.Linear(self.ndf*2, 500),
            nn.Linear(500, 500)
        )

        self.classifier_c = nn.Sequential(nn.Linear(500, nclasses))              
        self.classifier_s = nn.Sequential(
                                nn.Linear(500, 1), 
                                nn.Sigmoid())              

    def forward(self, input):       
        conv_output = self.feature(input)
        fc_output = self.fc(conv_output.view(-1, self.ndf*2))
        output_s = self.classifier_s(fc_output)
        output_s = output_s.view(-1)
        output_c = self.classifier_c(fc_output)
        return output_s, output_c

"""
Discriminator: Alexnet
"""
class _netD_alexnet_v2(nn.Module):
    def __init__(self, opt, nclasses):
        super(_netD_alexnet_v2, self).__init__()
        
        self.ndf = opt.ndf

        self.feature = nn.Sequential(
            nn.Conv2d(3, self.ndf*2, 5, 1),            
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(self.ndf*2, self.ndf*2, 5, 1),         
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(self.ndf*2, self.ndf*2, 5, 1, 1),           
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(self.ndf*2, self.ndf*2, 5, 1),           
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc= nn.Sequential(
            nn.Linear(self.ndf*2, 500),
            nn.Linear(500, 500)
        )

        self.classifier_c = nn.Sequential(nn.Linear(500, nclasses))              
        self.classifier_s = nn.Sequential(
                                nn.Linear(500, 1), 
                                nn.Sigmoid())              

    def forward(self, input):       
        conv_output = self.feature(input)
        fc_output = self.fc(conv_output.view(-1, self.ndf*2))
        output_s = self.classifier_s(fc_output)
        output_s = output_s.view(-1)
        output_c = self.classifier_c(fc_output)
        return output_s, output_c

"""
Generator: WGAN
"""
class _netG_wgan(nn.Module):

    def __init__(self, opt, nclasses):
        super(_netG_wgan, self).__init__()

        self.ndim = opt.ndf*4
        self.imageSize = opt.imageSize
        self.nz = opt.nz
        self.gpu = opt.gpu
        self.nclasses = nclasses

        self.fc = nn.Sequential(
            nn.Linear(self.ndim + self.nz + self.nclasses + 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.imageSize // 4) * (self.imageSize // 4)),
            nn.BatchNorm1d(128 * (self.imageSize // 4) * (self.imageSize // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, inputs):
        batchSize = inputs.size()[0]
        inputs = inputs.view(-1, self.ndim+self.nclasses+1)
        noise = torch.FloatTensor(batchSize, self.nz).normal_(0, 1)    
        if self.gpu>=0:
            noise = noise.cuda()
        noisev = Variable(noise)
        inputs_cat = torch.cat((inputs, noisev),1)
        
        x = self.fc(inputs_cat)
        x = x.view(-1, 128, (self.imageSize // 4), (self.imageSize // 4))
        x = self.deconv(x)

        return x

"""
Discriminator: WGAN
"""
class _netD_wgan(nn.Module):

    def __init__(self, opt, nclasses):
        super(_netD_wgan, self).__init__()

        self.input_dim = 3
        self.imageSize = opt.imageSize

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * (self.imageSize // 4) * (self.imageSize // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )

        self.classifier_c = nn.Sequential(nn.Linear(1024, nclasses))              
        self.classifier_s = nn.Sequential(nn.Linear(1024, 1))

    def forward(self, inputs):
        x = self.conv(inputs)
        x = x.view(-1, 128 * (self.imageSize // 4) * (self.imageSize // 4))
        x = self.fc(x)

        output_c = self.classifier_c(x)
        output_s = self.classifier_s(x)

        return output_s, output_c

"""
Generator: WGAN
"""
class _netG_wgan_alexnet(nn.Module):

    def __init__(self, opt, nclasses):
        super(_netG_wgan_alexnet, self).__init__()

        self.ndim = 4096
        self.imageSize = opt.dSize
        self.nz = opt.nz
        self.gpu = opt.gpu
        self.nclasses = nclasses

        self.fc = nn.Sequential(
            nn.Linear(self.ndim + self.nz + self.nclasses + 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.imageSize // 8) * (self.imageSize // 8)),
            nn.BatchNorm1d(128 * (self.imageSize // 8) * (self.imageSize // 8)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, inputs):
        batchSize = inputs.size()[0]
        inputs = inputs.view(-1, self.ndim+self.nclasses+1)
        noise = torch.FloatTensor(batchSize, self.nz).normal_(0, 1)    
        if self.gpu>=0:
            noise = noise.cuda()
        noisev = Variable(noise)
        inputs_cat = torch.cat((inputs, noisev),1)
        
        x = self.fc(inputs_cat)
        x = x.view(-1, 128, (self.imageSize // 8), (self.imageSize // 8))
        x = self.deconv(x)

        return x

"""
Discriminator: WGAN
"""
class _netD_wgan_alexnet(nn.Module):

    def __init__(self, opt, nclasses):
        super(_netD_wgan_alexnet, self).__init__()

        self.input_dim = 3
        self.imageSize = opt.dSize

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * (self.imageSize // 8) * (self.imageSize // 8), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )

        self.classifier_c = nn.Sequential(nn.Linear(1024, nclasses))              
        self.classifier_s = nn.Sequential(nn.Linear(1024, 1))

    def forward(self, inputs):
        x = self.conv(inputs)
        x = x.view(-1, 256 * (self.imageSize // 8) * (self.imageSize // 8))
        x = self.fc(x)

        output_c = self.classifier_c(x)
        output_s = self.classifier_s(x)

        return output_s, output_c

"""
Feature extractor: VGGNet
"""
class _netF_vggnet(nn.Module):
    def __init__(self, opt):
        super(_netF_vggnet, self).__init__()

        self.ndf = opt.ndf
        self.gpu = opt.gpu

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        
        self.mean = nn.Sequential()
        self.mean.add_module('fc_mean',nn.Linear(4096, self.ndf*2))

        self.std = nn.Sequential()
        self.std.add_module('fc_std', nn.Linear(4096, self.ndf*2))

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)

        tmp_dict = {}
        for k,v in pretrained_dict.items():
            if (k == "state_dict"):
                tmp_dict = v

        # get pretrained parameters
        param_list = []
        for k,v in pretrained_dict.items():
            if not ('classifier.6' in k):
                param_list.append(v)

        key_list = []
        for k in model_dict:
            if ('feature' in k or 'classifier' in k):
                key_list.append(k)

        # get parameter dict
        param_dict = {}
        for i in range(len(key_list)):
            print (key_list[i])
            param_dict[key_list[i]] = param_list[i] 
        
        model_dict.update(param_dict)
        self.load_state_dict(model_dict)

    def forward(self, inputs):
        B,C,H,W = inputs.size()
        output = self.feature(inputs)
        embedding = output.view(B, 7 * 7 * 512)

        output = self.classifier(embedding)

        mean_vector = self.mean(output)
        std_vector = self.std(output)
        if self.gpu>=0:
            mean_vector = mean_vector.cuda()
            std_vector = std_vector.cuda()

        return output, mean_vector, std_vector

"""
Classifier: VGGNet
"""
class _netC_vggnet(nn.Module):
    def __init__(self, opt, nclasses): 
        super(_netC_vggnet, self).__init__()
        
        self.classifier = nn.Sequential(nn.Linear(4096, nclasses))

        self.soft = nn.Sequential(nn.Sigmoid())

    def forward(self, embedding):      
        output_logit = self.classifier(embedding)
        output = self.soft(output_logit)
        return output_logit, output

"""
Bottleneck for resnet
"""
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

"""
Encoder: Resnet
"""
class _netF_resnet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(_netF_resnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def load(self, checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)

        tmp_dict = {}
        for k,v in pretrained_dict.items():
            if (k == "state_dict"):
                tmp_dict = v

        # get pretrained parameters
        param_list = []
        for k,v in pretrained_dict.items():
            if not ('fc' in k):
                param_list.append(v)

        key_list = []
        for k in model_dict:
            if ('layer' in k or 'bn' in k or 'conv' in k):
                key_list.append(k)

        # get parameter dict
        param_dict = {}
        for i in range(len(key_list)):
            print (key_list[i])
            param_dict[key_list[i]] = param_list[i] 
        
        model_dict.update(param_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        embedding = x.view(x.size(0), -1)

        return embedding

"""
Classifier: Resnet
"""
class _netC_resnet(nn.Module):
    def __init__(self, opt, nclasses): 
        super(_netC_resnet, self).__init__()
        
        self.classifier = nn.Sequential(nn.Linear(2048, nclasses))

        self.soft = nn.Sequential(nn.Sigmoid())

    def forward(self, embedding):
        output_logit = self.classifier(embedding)
        output = self.soft(output_logit)

        return output_logit, output

def alexnet(pretrained=False):
  model = _netF_alexnet()
  print("backbone:alexnet")
  # if pretrained:
  #   model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  if pretrained == True:
    model_path = 'data/pretrained_model/alexnet_pretrain.pth'
    print("Loading pretrained weights from %s" %(model_path))
    model.load(model_path)
    # model.fc = torch.nn.Linear(2048, 1024)
  return model