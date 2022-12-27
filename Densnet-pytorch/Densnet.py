import torch
import torch.nn as nn

model_cfg = {
    "densnet121":[6,12,24,16],
    "densnet169":[6,12,32,32],
    "densnet201":[6,12,48,32],
    "densnet264":[6,12,64,48],
}


class DenseLayer(nn.Module):
    def __init__(
        self, in_ch, growth_rate, bottleneck_size,
        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, drop_rate=0.,
    ) -> None:
        super(DenseLayer, self).__init__()
        self.add_module("norm1", norm_layer(in_ch))
        self.add_module("act1", act_layer(inplace=True))
        self.add_module("conv1", nn.Conv2d(in_ch, growth_rate*bottleneck_size, 1, 1))
        self.add_module("norm2", norm_layer(growth_rate*bottleneck_size))
        self.add_module("act2", act_layer(inplace=True))
        self.add_module("conv2", nn.Conv2d(growth_rate*bottleneck_size, growth_rate, 3, 1, 1))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = [x]
        x = torch.cat(x, 1)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x



class DenseBlock(nn.ModuleDict):
    def __init__(
        self, num_layer, in_ch, growth_rate, bottleneck_size,
        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, drop_rate=0.
    ) -> None:
        super(DenseBlock, self).__init__()
        for i in range(num_layer):
            layer = DenseLayer(
                in_ch=in_ch+(i*growth_rate),
                growth_rate=growth_rate,
                bottleneck_size=bottleneck_size,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            self.add_module(f"denselayer{i+1}", layer)

    def forward(self, init_features:torch.Tensor) -> torch.Tensor:
        features = [init_features]
        for _, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
        


class Densnet(nn.Module):
    """
    hyperparameter for Densnet:
        growth_rate - how many filters to add each layer ('k' in paper)
        bottleneck_size - multiplicative factor at the bottle neck layer
    """
    def __init__(
        self, block_cfg, growth_rate=32, bottleneck_size=4, num_classes=1000, channels=[64,128,256],
        global_pool='avg', norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU,
    ) -> None:
        super(Densnet, self).__init__()
        self.conv0 = nn.Conv2d(3, channels[0], 7, 2, 3)
        self.norm0 = norm_layer(channels[0])
        self.relu0 = act_layer(inplace=True)
        self.pool0 = nn.MaxPool2d(3, 2, 3)

        self.feature = self.make_blocks(block_cfg)
    
    # def make_blocks(self, block_cfg: list):
        # for c in channels:

    # def _forward_impl(self, x:torch.Tensor) -> torch.Tensor:
    #     x = self.one_l