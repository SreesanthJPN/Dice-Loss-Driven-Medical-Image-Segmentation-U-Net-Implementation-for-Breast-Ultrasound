import torch
from torch import nn

class conv_block(nn.Module):

    def __init__(self, in_dims, n_filters):
        super(conv_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = in_dims, out_channels = n_filters, padding = 1, kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(n_filters)

        self.conv2 = nn.Conv2d(in_channels = n_filters, out_channels = n_filters, padding = 1, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(n_filters)

        self.relu = nn.ReLU()

    
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.conv = conv_block(in_dims = in_channels, n_filters = out_channels)
        self.pool = nn.MaxPool2d((2,2))
    
    def forward(self, x):

        x = self.conv(x)
        p = self.pool(x)

        return x, p
    
class Decoder(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()

        self.upscale = nn.ConvTranspose2d(in_channels = in_ch, out_channels = out_ch, stride = 2, padding = 0, kernel_size = 2)
        self.conv = conv_block(in_dims = in_ch, n_filters = out_ch)

    def forward(self, x, skip):

        x = self.upscale(x)
        x = torch.cat([x, skip], axis = 1)
        x = self.conv(x)
        return x
    