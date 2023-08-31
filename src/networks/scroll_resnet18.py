from distutils.util import change_root
import torch.nn as nn
import math

from .slimmable_ops import SwitchableBatchNorm2d
from .slimmable_ops import SlimmableConv2d, SlimmableLinear
from widths.config import FLAGS

import pdb

def slimconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return SlimmableConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        channels_in = [int(inplanes * width_mult) for width_mult in FLAGS.width_mult_list]
        channels_out = [int(planes * width_mult) for width_mult in FLAGS.width_mult_list]

        self.conv1 = slimconv3x3(channels_in, channels_out, stride)
        self.bn1 = SwitchableBatchNorm2d(channels_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = slimconv3x3(channels_out, channels_out)
        self.bn2 = SwitchableBatchNorm2d(channels_out)
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

class Scroll_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(Scroll_ResNet, self).__init__()
        chann_head_in = [3 for width_mult in FLAGS.width_mult_list]
        chann_head_out = [int(64 * width_mult) for width_mult in FLAGS.width_mult_list]
        self.conv1 = SlimmableConv2d(chann_head_in, chann_head_out, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.bn1 = SwitchableBatchNorm2d(chann_head_out)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # last classifier layer (head) with as many outputs as classes
        chann_tail_in = [int(512 * block.expansion * width_mult) for width_mult in FLAGS.width_mult_list]
        chann_tail_out = [num_classes for width_mult in FLAGS.width_mult_list]
        self.fc = SlimmableLinear(chann_tail_in, chann_tail_out)
        # self.last_dim = self.fc.in_features
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        chann_d_in = [int(self.inplanes * width_mult) for width_mult in FLAGS.width_mult_list]
        chann_d_out = [int(planes * block.expansion * width_mult) for width_mult in FLAGS.width_mult_list]

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SlimmableConv2d(chann_d_in, chann_d_out,
                          kernel_size=1, stride=stride, bias=False),
                SwitchableBatchNorm2d(chann_d_out),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def scroll_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Scroll_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model
