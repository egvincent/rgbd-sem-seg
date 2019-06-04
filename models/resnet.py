"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

from utils.helpers import maybe_download
from utils.layer_factory import conv1x1, conv3x3, CRPBlock

data_info = {
    7 : 'Person',
    21: 'VOC',
    40: 'NYU',
    60: 'Context'
    }

models_urls = {
    '50_person'  : 'https://cloudstor.aarnet.edu.au/plus/s/mLA7NxVSPjNL7Oo/download',
    '101_person' : 'https://cloudstor.aarnet.edu.au/plus/s/f1tGGpwdCnYS3xu/download',
    '152_person' : 'https://cloudstor.aarnet.edu.au/plus/s/Ql64rWqiTvWGAA0/download',

    '50_voc'     : 'https://cloudstor.aarnet.edu.au/plus/s/2E1KrdF2Rfc5khB/download',
    '101_voc'    : 'https://cloudstor.aarnet.edu.au/plus/s/CPRKWiaCIDRdOwF/download',
    '152_voc'    : 'https://cloudstor.aarnet.edu.au/plus/s/2w8bFOd45JtPqbD/download',

    '50_nyu'     : 'https://cloudstor.aarnet.edu.au/plus/s/gE8dnQmHr9svpfu/download',
    '101_nyu'    : 'https://cloudstor.aarnet.edu.au/plus/s/VnsaSUHNZkuIqeB/download',
    '152_nyu'    : 'https://cloudstor.aarnet.edu.au/plus/s/EkPQzB2KtrrDnKf/download',

    '101_context': 'https://cloudstor.aarnet.edu.au/plus/s/hqmplxWOBbOYYjN/download',
    '152_context': 'https://cloudstor.aarnet.edu.au/plus/s/O84NszlYlsu00fW/download',

    '50_imagenet' : 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101_imagenet': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    '152_imagenet': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

stages_suffixes = {0 : '_conv',
                   1 : '_conv_relu_varout_dimred'}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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


### This class was made to abstract the RefineNet code from the original
### ResNet (originally ResNetLW) module into its own module. Further, I have
### made the following modifications to convert from LW-RefineNet to RefineNet:
###     - conv1x1 changed to conv3x3 in all lines ending in _varout_dimred, 
###       which are part of the multi-resolution fusion blocks 
###     - RCU blocks added
###     - heavy reorganization, renaming, etc
### I would like to rename the blocks in this module but I think the names are important for
###   other parts of the code
class RefineNet(nn.Module):

    # inplanes_hr/lr are the number of channels of the high resolution input and
    #   low resolution input respectively. if inplanes_hr is omitted, no fusion takes
    #   place: this is useful for layer 4 in our case, to prep for fusing it with layer 3.
    # the inputs will be passed through:
    #   - if HR is provided: 2 RCU blocks for HR only
    #     otherwise, 2 RCU blocks for LR only
    #   - fusion of HR and LR (if HR is provided)
    #   - CRP block + 1 RCU block for result of fusion (or LR if no HR provided)
    def __init__(self, inplanes_lr, inplanes_hr=None):
        super(RefineNet, self).__init__()
        
        # first set of RCU blocks
        if inplanes_hr == None:
            self.inplanes = inplanes_lr
            self.rcu1_lr = self._make_layer(BasicBlock, planes=inplanes_lr, blocks=2)
        else:
            self.inplanes = inplanes_hr
            self.rcu1_hr = self._make_layer(BasicBlock, planes=inplanes_hr, blocks=2)  

        # fusion
        if inplanes_hr != None:
            self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3(inplanes_lr, inplanes_hr, bias=False)
            self.adapt_stage2_b2_joint_varout_dimred = conv3x3(inplanes_hr, inplanes_hr, bias=False)

        # CRP and RCU for fusion result
        outplanes = inplanes_hr if inplanes_hr != None else inplanes_lr
        self.mflow_conv_g1_pool = self._make_crp(outplanes, outplanes, 4)
        self.inplanes = outplanes
        self.rcu2 = self._make_layer(BasicBlock, planes=outplanes, blocks=1)

    # _make_layer and _make_crp copied from the ResNetLW class
    
    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)

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

    def forward(self, x_lr, x_hr=None):

        # first set of RCU blocks
        if x_hr is None:
            x_lr = self.rcu1_lr(x_lr)
        else:
            x_hr = self.rcu1_hr(x_hr)

        # fusion (or not)
        x = x_lr
        if x_hr is not None:
            x_lr = self.mflow_conv_g1_b3_joint_varout_dimred(x_lr)
            x_lr = nn.Upsample(size=x_hr.size()[2:], mode='bilinear', align_corners=True)(x_lr)
            
            x_hr = self.adapt_stage2_b2_joint_varout_dimred(x_hr)
            
            x = x_lr + x_hr
            x = F.relu(x)

        # CRP and RCU for fusion result
        x = self.mflow_conv_g1_pool(x)  # CRP
        x = self.rcu2(x)
        
        out = x
        return out


### added for RDFNet implementation
class MMFNet(nn.Module):
    
    # in_channels will be 512 for MMFNet-4 and 256 for others, as per RDFNet paper
    def __init__(self, in_channels):
        super(MMFNet, self).__init__()
        
        self.do = nn.Dropout(p=0.5)

        # these next blocks are the same for both RGB and HHA because 
        #  the input volumes are exactly the same dimensions (including # channels)
        # conv3 is also used for the convolution before and after fusion

        # pre-fusion RGB blocks
        self.conv1_rgb = conv1x1(in_planes=in_channels, out_planes=in_channels)
        self.RCUs_rgb = nn.Sequential(  # 2 RCU blocks
            BasicBlock(inplanes=in_channels, planes=in_channels),  # expansion=1, no downsampling
            BasicBlock(inplanes=in_channels, planes=in_channels)   # expansion=1, no downsampling
        )
        self.conv3_rgb = conv3x3(in_planes=in_channels, out_planes=in_channels)

        # pre-fusion HHA blocks
        self.conv1_hha = conv1x1(in_planes=in_channels, out_planes=in_channels)
        self.RCUs_hha = nn.Sequential(  # 2 RCU blocks
            BasicBlock(inplanes=in_channels, planes=in_channels),  # expansion=1, no downsampling
            BasicBlock(inplanes=in_channels, planes=in_channels)   # expansion=1, no downsampling
        )
        self.conv3_hha = conv3x3(in_planes=in_channels, out_planes=in_channels)

        # post-fusion block
        #self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1)
        #self.conv3 = conv3x3(in_planes=in_channels, out_planes=in_channels)
        self.crp = CRPBlock(in_planes=in_channels, out_planes=in_channels, n_stages=1)

    def forward(self, x_rgb, x_depth):
        # pre-fusion RGB
        x_rgb = self.do(x_rgb)
        x_rgb = self.conv1_rgb(x_rgb)
        x_rgb = self.RCUs_rgb(x_rgb)
        x_rgb = self.conv3_rgb(x_rgb)

        # pre-fusion HHA
        x_depth = self.do(x_depth)
        x_depth = self.conv1_hha(x_depth)
        x_depth = self.RCUs_hha(x_depth)
        x_depth = self.conv3_hha(x_depth)

        # fusion
        x = x_rgb + x_depth

        # post-fusion
        x = F.relu(x)
        #residual = x
        #x = self.maxpool(x)
        #x = self.conv3(x)
        #out = x + residual
        out = self.crp(x)

        return out


### This module now represents the RDFNet
### 
### changes and additions to the original version of this class:
###  - reorganization and commenting for clarity
###  - RefineNet component isolated into its own RefineNet module: see above
###  - to convert from RefineNet to RDFNet:
###     - depth track added (all blocks ending in "_depth")
###     - MMFNet added to fuse RGB and depth tracks
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=40):#21):
        super(ResNet, self).__init__()

        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_depth = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_depth = nn.BatchNorm2d(64)
        #self.relu = nn.ReLU(inplace=True) -- switched to F.relu to avoid errors
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        # backbone
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.inplanes = 64
        self.layer1_depth = self._make_layer(block, 64, layers[0])
        self.layer2_depth = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_depth = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_depth = self._make_layer(block, 512, layers[3], stride=2)

        # dimensionality reduction right off the backbone
        # note that for some reason, x4 uses the outl1 block, x3 uses outl2, etc
        # I would like to rename these as well, but I think the names are used for some other stuff
        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)  # really l4
        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)  # really l3
        self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)   # really l2
        self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)   # really l1
        self.p_ims1d2_outl1_dimred_depth = conv1x1(2048, 512, bias=False)  # really l4
        self.p_ims1d2_outl2_dimred_depth = conv1x1(1024, 256, bias=False)  # really l3
        self.p_ims1d2_outl3_dimred_depth = conv1x1(512, 256, bias=False)   # really l2
        self.p_ims1d2_outl4_dimred_depth = conv1x1(256, 256, bias=False)   # really l1
        
        # MMFNets
        self.MMFNet_l4 = MMFNet(in_channels=512)
        self.MMFNet_l3 = MMFNet(in_channels=256)
        self.MMFNet_l2 = MMFNet(in_channels=256)
        self.MMFNet_l1 = MMFNet(in_channels=256)

        # RefineNets
        self.RefineNet_l4 = RefineNet(inplanes_lr=512)  # no fusion step
        self.RefineNet_l4_l3 = RefineNet(inplanes_lr=512, inplanes_hr=256)
        self.RefineNet_l3_l2 = RefineNet(inplanes_lr=256, inplanes_hr=256)
        self.RefineNet_l2_l1 = RefineNet(inplanes_lr=256, inplanes_hr=256)

        # CLF convolutional step applied to layer 1 output to get class predictions
        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1,
                                  padding=1, bias=True) 
        

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)

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

    def forward(self, x, x_depth):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x_depth = self.conv1_depth(x_depth)
        x_depth = self.bn1_depth(x_depth)
        x_depth = F.relu(x_depth)
        x_depth = self.maxpool(x_depth)

        # backbone
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l1_depth = self.layer1(x_depth)
        l2_depth = self.layer2(l1_depth)
        l3_depth = self.layer3(l2_depth)
        l4_depth = self.layer4(l3_depth)

        # not sure why this is only for l3 and l4, but it's in the original code
        # so I added the same thing added for depth
        l4 = self.do(l4)
        l3 = self.do(l3)
        l4_depth = self.do(l4_depth)
        l3_depth = self.do(l3_depth)

        # dimensionality reduction right off the backbone
        x4 = F.relu(self.p_ims1d2_outl1_dimred(l4))
        x3 = F.relu(self.p_ims1d2_outl2_dimred(l3))
        x2 = F.relu(self.p_ims1d2_outl3_dimred(l2))
        x1 = F.relu(self.p_ims1d2_outl4_dimred(l1))
        x4_depth = F.relu(self.p_ims1d2_outl1_dimred_depth(l4_depth))
        x3_depth = F.relu(self.p_ims1d2_outl2_dimred_depth(l3_depth))
        x2_depth = F.relu(self.p_ims1d2_outl3_dimred_depth(l2_depth))
        x1_depth = F.relu(self.p_ims1d2_outl4_dimred_depth(l1_depth))

        # MMFNets
        x4 = self.MMFNet_l4(x4, x4_depth)
        x3 = self.MMFNet_l3(x3, x3_depth)
        x2 = self.MMFNet_l2(x2, x2_depth)
        x1 = self.MMFNet_l1(x1, x1_depth)

        # RefineNets
        x4 = self.RefineNet_l4(x4)
        x3 = self.RefineNet_l4_l3(x4, x3)
        x2 = self.RefineNet_l3_l2(x3, x2)
        x1 = self.RefineNet_l2_l1(x2, x1)

        # CLF convolutional step to x1 to get class predictions
        out = self.clf_conv(x1)
        return out


### NOTE: for the following functions, I have changed the default value of pretrained
###  from True to False. I might change it back later

def rf_lw50(num_classes, imagenet=False, pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = '50_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = '50_' + dataset.lower()
            key = 'rf_lw' + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model

def rf_lw101(num_classes, imagenet=False, pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = '101_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = '101_' + dataset.lower()
            key = 'rf_lw' + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model

def rf_lw152(num_classes, imagenet=False, pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = '152_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = '152_' + dataset.lower()
            key = 'rf_lw' + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model
