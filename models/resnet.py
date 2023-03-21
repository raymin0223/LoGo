import torch
import torch.nn as nn

__all__ = ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 
           'resnet152', 'wide_resnet50_2', 'wide_resnet101_2']

    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    

class Block(nn.Module):
    __constants__ = ['downsample']
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
                 base_width=64, dilation=1, norm_layer=None, expansion=1, block_type='basic'):
        super(Block, self).__init__()
        if block_type not in ['basic', 'bottleneck']:
            raise ValueError('Block_Type only supports basic and bottleneck')
        self.block_type = block_type
        self.expansion = expansion
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type == 'basic':
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        width = int(planes * (base_width / 64.)) * groups
        # Both conv3*3 with stride and self.downsample layers downsample the input when stride != 1
        if block_type == 'basic':
            self.conv1 = conv3x3(inplanes, width, stride)
            self.conv2 = conv3x3(width, width)
            
        if block_type == 'bottleneck':
            self.conv1 = conv1x1(inplanes, width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            
        self.bn1 = norm_layer(width)
        self.bn2 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        if self.block_type == 'bottleneck':
            out = self.relu(out)
            
            out = self.conv3(out)
            out = self.bn3(out)
            
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out
    
    
class ResNet(nn.Module):
    BasicBlock_arch = ['resnet10', 'resnet18', 'resnet34']
    Bottleneck_arch = ['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 
                      'wide_resnet50_2', 'wide_resnet101_2']

    def __init__(self, arch, repeats, in_channels=3, num_classes=100, zero_init_residual=True,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        if arch in self.BasicBlock_arch:
            self.expansion = 1
            self.block_type = 'basic'
        elif arch in self.Bottleneck_arch:
            self.expansion = 4
            self.block_type = 'bottleneck'
        else:
            raise NotImplementedError('%s arch is not supported in ResNet' % arch)
            
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.planes = [64, 128, 256, 512] # first plane is for input channel
        self.strides = [1, 2, 2, 2]
        
        self.block_layers = self._make_layer(self.planes, repeats, self.strides)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * self.expansion, num_classes)
        self.linear.bias.data.fill_(0)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Block):
                    if self.block_type == 'basic':
                        nn.init.constant_(m.bn2.weight, 0)
                    if self.block_type == 'bottleneck':
                        nn.init.constant_(m.bn3.weight, 0)
        
    def _make_layer(self, planes, repeats, strides):
        assert len(planes) == len(repeats) == len(strides) == 4, 'Number of Block should be 4'
        
        block_layers = []
        norm_layer = self._norm_layer
        for i in range(4):
            plane = planes[i]
            repeat = repeats[i]
            stride = strides[i]
            
            downsample = None
            if stride != 1 or self.inplanes != plane * self.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, plane * self.expansion, stride),
                    norm_layer(plane * self.expansion),
                )

            layers = []
            layers.append(nn.Sequential(*[Block(self.inplanes, plane, stride, downsample, self.groups,
                                                self.base_width, self.dilation, norm_layer, self.expansion, 
                                                self.block_type)]))
            self.inplanes = plane * self.expansion
            for _ in range(1, repeat):
                layers.append(nn.Sequential(*[Block(self.inplanes, plane, groups=self.groups,
                                                          base_width=self.base_width, dilation=self.dilation,
                                                          norm_layer=norm_layer, expansion=self.expansion, 
                                                          block_type=self.block_type)]))
            block_layers.append(nn.Sequential(*layers))
            
        return nn.Sequential(*block_layers)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.block_layers(x)
        
        features = self.avgpool(x)
        features = features.view(x.size(0), -1)
        
        logits = self.linear(features)

        return logits, features
    
    def get_embedding_dim(self):
        return 512 * self.expansion
    
    
def _resnet(arch, repeats, **kwargs):
    model = ResNet(arch, repeats, **kwargs)
    return model


def resnet10(**kwargs):
    return _resnet('resnet10', [1, 1, 1, 1], **kwargs)


def resnet18(**kwargs):
    return _resnet('resnet18', [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return _resnet('resnet34', [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return _resnet('resnet50', [3, 4, 6, 3], **kwargs)


def resnet101( **kwargs):
    return _resnet('resnet101', [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return _resnet('resnet152', [3, 8, 36, 3], **kwargs)


def wide_resnet50_2(**kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', [3, 4, 6, 3], **kwargs)


def wide_resnet101_2(**kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', [3, 4, 23, 3], **kwargs)

