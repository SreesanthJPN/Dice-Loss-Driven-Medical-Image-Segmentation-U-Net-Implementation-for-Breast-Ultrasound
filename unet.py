import torch 
from torch import nn
from blocks import Encoder, Decoder, conv_block


class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()
        self.en1 = Encoder(3, 64)
        self.en2 = Encoder(64, 128)
        self.en3 = Encoder(128, 256)
        self.en4 = Encoder(256, 512)

        self.bottle_neck = conv_block(512, 1024)

        self.dc1 = Decoder(1024, 512)
        self.dc2 = Decoder(512, 256)
        self.dc3 = Decoder(256, 128)
        self.dc4 = Decoder(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size = 1, padding = 0)


    def forward(self, input):

        s1, p1 = self.en1(input)
        s2, p2 = self.en2(p1)
        s3, p3 = self.en3(p2)
        s4, p4 = self.en4(p3)

        b = self.bottle_neck(p4)

        d1 = self.dc1(b, s4)
        d2 = self.dc2(d1, s3)
        d3 = self.dc3(d2, s2)
        d4 = self.dc4(d3, s1)

        out = self.out(d4)

        return out