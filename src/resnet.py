import torch
import torch.nn as nn

class Basicblock(nn.Module):
    """Builds the residual structure for ResNet-18 and ResNet-34.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the convolution (the first residual block reduces the image size by half).
        downsample (Callable, optional): Downsampling function (used for the first residual block to reduce the size of x by half for addition).
    """
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        
        return out
    

class Bottleneck(nn.Module):
    """Builds the residual structure for ResNet-50 and ResNet-101.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the convolution (the first residual block reduces the image size by half).
        downsample (Callable, optional): Downsampling function (used for the first residual block to reduce the size of x by half for addition).
        expansion (int): Expansion factor for the number of channels.
    """
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    
    def __init__(self, block, block_nums, num_classes=1000, include_top=True):
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block=block, channel=64, block_num=block_nums[0])
        self.layer2 = self._make_layer(block=block, channel=128, block_num=block_nums[1], stride=2)
        self.layer3 = self._make_layer(block=block, channel=256, block_num=block_nums[2], stride=2)
        self.layer4 = self._make_layer(block=block, channel=512, block_num=block_nums[3], stride=2)
        
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_features=512*block.expansion, out_features=num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # First approach: For ResNet-50/101/152, the first layer of conv2_x needs to increase the channels 
        # by a factor of 4, while keeping the height and width the same.
        # Second approach: For conv3/4/5_x, the first layer needs to change both 
        # the number of channels and the height and width.
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channel, channel, stride=stride, downsample=downsample))
        self.in_channel = channel * block.expansion
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))
    
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        if self.include_top:
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
        
        return out


def resnet18(num_classes=1000, include_top=True):
    return ResNet(block=Basicblock, block_nums=[2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    return ResNet(block=Basicblock, block_nums=[3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True):
    return ResNet(block=Bottleneck, block_nums=[3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    return ResNet(block=Bottleneck, block_nums=[3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnet152(num_classes=1000, include_top=True):
    return ResNet(block=Bottleneck, block_nums=[3, 8, 36, 3], num_classes=num_classes, include_top=include_top)