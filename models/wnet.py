import torch.nn as nn

from base import BaseModel
from models.unet import UNet


class WNet(BaseModel):
    def __init__(self, hidden=64, in_channels=3, freeze_bn=False, **_):
        super(WNet, self).__init__()

        self.unet_encoder = UNet(out_dim=hidden, in_dim=in_channels, freeze_bn=freeze_bn)
        self.unet_decoder = UNet(out_dim=in_channels, in_dim=hidden, freeze_bn=freeze_bn)

    def forward(self, x):
        hidden = self.unet_encoder(x)
        x_rec = self.unet_decoder(hidden)
        return hidden, x_rec

    def get_backbone_params(self):
        # There is no backbone for wnet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
