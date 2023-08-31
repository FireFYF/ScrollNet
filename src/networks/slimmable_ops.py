import torch
import torch.nn as nn
import random
from widths.config import FLAGS

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(FLAGS.width_mult_list)
        self.ignore_model_profiling = True
        self.scroll = 0.0

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(FLAGS.width_mult_list)
        self.scroll = 0.0
        self.inverse = 1 # 0 is up, 1 is down

    def cyc_scroll(self, scroll_num1, scroll_num2):
        h, w, _, _ = self.weight.shape
        matrix = torch.cat((self.weight[(h-scroll_num1):,:], self.weight[:(h-scroll_num1),:]), dim=0)
        weight = torch.cat((matrix[:,(h-scroll_num2):], matrix[:,:(h-scroll_num2)]), dim=1)

        return weight 

    def cyc_scroll_bias(self, scroll_num):
        L = self.bias
        bias = torch.cat((self.bias[(L-scroll_num):], self.bias[:(L-scroll_num)]), dim=0)

        return bias 

    def cyc_scroll_inverse(self, scroll_num1, scroll_num2):
        h, w, _, _ = self.weight.shape
        matrix = torch.cat((self.weight[scroll_num1:,:], self.weight[:scroll_num1,:]), dim=0)
        weight = torch.cat((matrix[:,scroll_num2:], matrix[:,:scroll_num2]), dim=1)

        return weight 

    def cyc_scroll_bias_inverse(self, scroll_num):
        bias = torch.cat((self.bias[scroll_num:], self.bias[:scroll_num]), dim=0)

        return bias 

    def forward(self, input):
        self.scroll = self.scroll % len(FLAGS.width_mult_list) # cycle scrolling
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]
        self.groups = self.groups_list[idx]
        scroll_num1 = int(self.scroll*(self.out_channels_list[1]-self.out_channels_list[0]))
        scroll_num2 = int(self.scroll*(self.in_channels_list[1]-self.in_channels_list[0]))

        if self.inverse==0:
            weight = self.cyc_scroll(scroll_num1, scroll_num2)
        elif self.inverse==1:
            weight = self.cyc_scroll_inverse(scroll_num1, scroll_num2)
        weight = weight[:self.out_channels, :self.in_channels, :, :]            
            
        if self.bias is not None:
            if self.inverse==0:
                bias = self.cyc_scroll_bias(scroll_num1)
            elif self.inverse==1:
                bias = self.cyc_scroll_bias_inverse(scroll_num1)
            bias = bias[:self.out_channels]
        else:
            bias = self.bias

        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(FLAGS.width_mult_list)
        self.scroll = 0.0

    def cyc_scroll_inverse(self, scroll_num1, scroll_num2):      
        matrix = torch.cat((self.weight[scroll_num1:,:], self.weight[:scroll_num1,:]), dim=0)
        weight = torch.cat((matrix[:,scroll_num2:], matrix[:,:scroll_num2]), dim=1)
        return weight 

    def cyc_scroll_bias_inverse(self, scroll_num):
        bias = torch.cat((self.bias[scroll_num:], self.bias[:scroll_num]), dim=0)
        return bias

    def forward(self, input):

        self.scroll = self.scroll % len(FLAGS.width_mult_list) # cycle scrolling

        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        scroll_num1 = int(self.scroll*(self.out_features_list[1]-self.out_features_list[0]))
        scroll_num2 = int(self.scroll*(self.in_features_list[1]-self.in_features_list[0]))

        weight = self.cyc_scroll_inverse(scroll_num1, scroll_num2)
        weight = weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.cyc_scroll_bias_inverse(scroll_num1)
            bias = bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)

def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v