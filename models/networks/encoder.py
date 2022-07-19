import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        if opt.encode_mask:
            # re-load layer1 and layer6
            self.layer1 = norm_layer(nn.Conv2d(opt.semantic_nc, ndf, kw, stride=2, padding=pw))
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 16, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def execute(self, x):
        if self.opt.encode_mask:
            x = self.layer1(x)
            x = self.layer2(nn.leaky_relu(x,0.2))
            x = self.layer3(nn.leaky_relu(x,0.2))
            x = self.layer4(nn.leaky_relu(x,0.2))
            x = self.layer5(nn.leaky_relu(x,0.2))
            x = self.layer6(nn.leaky_relu(x,0.2))
            x = nn.leaky_relu(x,0.2)

            return x
        else:
            if x.size(2) != 256 or x.size(3) != 256:
                x = nn.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

            x = self.layer1(x)
            x = self.layer2(nn.leaky_relu(x,0.2))
            x = self.layer3(nn.leaky_relu(x,0.2))
            x = self.layer4(nn.leaky_relu(x,0.2))
            x = self.layer5(nn.leaky_relu(x,0.2))
            if self.opt.crop_size >= 256:
                x = self.layer6(nn.leaky_relu(x,0.2))
            x = nn.leaky_relu(x,0.2)

            x = x.view(x.size(0), -1)
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)

            return mu, logvar
