import torch
import torch.nn as nn

model_cfg = {
    "densnet121":[6,12,24,16],
    "densnet169":[6,12,32,32],
    "densnet201":[6,12,48,32],
    "densnet264":[6,12,64,48],
}

def one_layer(in_ch, out_ch):

class Densnet(nn.Module):
    def __init__(
        self,
        layers, 
        num_classes=1000, 
        channels=[64,128,256],
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
    ) -> None:
        super(Densnet, self).__init__()
        self.conv0 = nn.Conv2d(3, channels[0], 7, 2, 3)
        self.norm0 = norm_layer(channels[0])
        self.relu0 = act_layer(inplace=True)
        self.pool0 = nn.MaxPool2d(3, 2, 3)

        self.feature = self.make_blocks(layers)
    
    def make_blocks(self, layers: list):
        # for c in channels:

    def _forward_impl(self, x:torch.Tensor) -> torch.Tensor:
        x = self.one_l