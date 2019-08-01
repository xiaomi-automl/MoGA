import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def stem(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        # nn.ReLU6(inplace=True)
        Hswish()
    )


def separable_conv(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_before_pooling(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        # nn.ReLU6(inplace=True)
        Hswish()
    )


def conv_head(inp, oup):
    return nn.Sequential(
            nn.Conv2d(inp, oup, 1, bias=False),
            Hswish(inplace=True),
            nn.Dropout2d(0.2)
    )


def classifier(inp, nclass):
    return nn.Linear(inp, nclass)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, act, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, 1, 0, bias=True),
            act
        )
        self.fc = nn.Sequential(
            nn.Conv2d(channel // reduction, channel, 1, 1, 0, bias=True),
            Hsigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.fc(y)
        return torch.mul(x, y)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, expand_ratio, act, se):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.act = act
        self.se = se
        padding = kernel_size // 2
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        if self.se:
            self.mid_se = SEModule(hidden_dim, act)
        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):
        inputs = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        if self.se:
            x = self.mid_se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.use_res_connect:
            return inputs + x
        else:
            return x


class MoGaB(nn.Module):
    def __init__(self, n_class=1000, input_size=224):
        super(MoGaB, self).__init__()
        assert input_size % 32 == 0
        mb_config = [
            # expansion, out_channel, kernel_size, stride, act(0 RE 1 Hs), se
            [3, 24, 3, 2, 0, 0],
            [3, 24, 3, 1, 0, 0],
            [6, 40, 7, 2, 0, 0],
            [3, 40, 3, 1, 0, 0],
            [6, 40, 5, 1, 0, 0],
            [6, 80, 3, 2, 1, 1],
            [6, 80, 5, 1, 1, 1],
            [3, 80, 3, 1, 1, 0],
            [6, 80, 7, 1, 1, 1],
            [6, 112, 7, 1, 1, 0],
            [3, 112, 5, 1, 1, 0],
            [6, 160, 7, 2, 1, 1],
            [6, 160, 7, 1, 1, 1],
            [6, 160, 3, 1, 1, 1],
        ]

        first_filter = 16
        second_filter = 16
        second_last_filter = 960
        last_channel = 1280

        self.last_channel = last_channel
        self.stem = stem(3, first_filter, 2)
        self.separable_conv = separable_conv(first_filter, second_filter)
        self.mb_module = list()
        input_channel = second_filter
        for t, c, k, s, a, se in mb_config:
            output_channel = c
            act = nn.ReLU(inplace=True) if a==0 else Hswish(inplace=True)
            self.mb_module.append(InvertedResidual(input_channel, output_channel, k, s, expand_ratio=t, act=act, se=se!=0))
            input_channel = output_channel
        self.mb_module = nn.Sequential(*self.mb_module)
        self.conv_before_pooling = conv_before_pooling(input_channel, second_last_filter)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_head = conv_head(second_last_filter, last_channel)
        self.classifier = classifier(last_channel, n_class)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.separable_conv(x)
        x = self.mb_module(x)
        x = self.conv_before_pooling(x)
        x = self.global_pooling(x)
        x = self.conv_head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()
