import torch
import torch.nn as nn
from pytorch_model_summary import summary

momentum_val = 0.1

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2,bias=False)
        self.bn1 = norm_layer(planes, momentum=momentum_val)
        self.elu = nn.ELU(inplace=True)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2,bias=False)
        self.bn2 = norm_layer(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.elu(out)

        return out


class Resnet1d(nn.Module):
    def __init__(self, n_classes, input_ch, input_time, batch_norm=True,
                 batch_norm_alpha=0.1):
        super(Resnet1d, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        self.n_ch4 = 200
        self.n_outputs = 256 
        self.num_hidden = 512 

        self.dilation = 1
        self.groups = 1
        self.base_width = input_ch
        norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.inplanes = 32
        self.conv1 = nn.Conv1d(input_ch, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.elu = nn.ELU(inplace=True)

        self.maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.maxpool3 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        

        block = BasicBlock

        layers = [2,2,2,2]
        kernel_sizes = [3, 3, 3, 3]
        self.layer1 = self._make_layer(block, 32, kernel_sizes[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, kernel_sizes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, kernel_sizes[2],layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, kernel_sizes[2], layers[2], stride=2)

    def _make_layer(self, block, planes, kernel_size, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes,planes * block.expansion, kernel_size=1, stride=stride,bias=False),
                norm_layer(planes * block.expansion, momentum=momentum_val),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size,stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = x.squeeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool1(x)
        x = self.layer3(x)
        x = self.maxpool2(x)
        x = self.layer4(x)

        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
    def get_embeddings(self, x, layer_num=None):
        
        if layer_num is None:
            x = x.squeeze(1)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.elu(x)

        if layer_num is None or layer_num <= 1:
            x = self.layer1(x)
            x0 = x

        if layer_num is None or layer_num <= 2:
            x = self.layer2(x)
            x = self.maxpool1(x)
            x1 = x  # [64,35]

        if layer_num is None or layer_num <= 3:
            x = self.layer3(x)
            x = self.maxpool2(x)
            x2 = x  # [128,4]

        if layer_num is None or layer_num <= 4:
            x = self.layer4(x)
            x3 = x  # [256,2]

        return_values = []

        if layer_num is None or layer_num <= 1:
            return_values.append(x0)
        if layer_num is None or layer_num <= 2:
            return_values.append(x1)
        if layer_num is None or layer_num <= 3:
            return_values.append(x2)
        if layer_num is None or layer_num <= 4:
            return_values.append(x3)

        return tuple(return_values)

       
if __name__ == '__main__':
    model = Resnet1d(3, 44, 1125)
    print(summary(model, torch.zeros((1, 1, 44, 1125)), show_input=False))