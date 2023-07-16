import torch
import torch.nn as nn
from .darknet import CSPDarknet
from .network_blocks import DilatedParllelResidualBlockB, GhostModule, CBAMBlock

class YOLOPAFPN(nn.Module):

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=True,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = GhostModule(int(in_channels[2] * width), int(in_channels[1] * width), kernel_size=1, stride=1)
        self.C3_p4 = CBAMBlock(channel=int(in_channels[2] * width), reduction=2, kernel_size=3)
        self.reduce_conv1 = GhostModule(int(in_channels[2] * width), int(in_channels[0] * width), kernel_size=1, stride=1)
        self.reduce_conv1_1 = GhostModule(int(in_channels[1] * width), int(in_channels[0] * width), kernel_size=1, stride=1)
        self.C3_p3 = CBAMBlock(channel=int(in_channels[0] * width), reduction=4, kernel_size=3)
        self.bu_conv2 = GhostModule(int(in_channels[0] * width), int(in_channels[0] * width), kernel_size=1, stride=2)
        self.C3_n3 = CBAMBlock(channel=int(in_channels[1] * width), reduction=4, kernel_size=3)
        self.bu_conv1 = GhostModule(int(in_channels[1] * width), int(in_channels[1] * width), kernel_size=1, stride=2)
        self.C3_n4 = CBAMBlock(channel=int(in_channels[2] * width), reduction=4, kernel_size=3)
        self.x0 = DilatedParllelResidualBlockB(int(in_channels[2] * width), int(in_channels[2] * width), add=True)
        self.x1 = DilatedParllelResidualBlockB(int(in_channels[1] * width), int(in_channels[1] * width), add=True)
        self.x2 = DilatedParllelResidualBlockB(int(in_channels[0] * width), int(in_channels[0] * width), add=True)
    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features
        x0 = self.x0(x0)
        x1 = self.x1(x1)
        x2 = self.x2(x2)
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        f_out1 = self.reduce_conv1_1(f_out1)
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
