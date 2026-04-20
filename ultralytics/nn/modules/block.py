# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
from .efficientvim import EfficientViMBlock, EfficientViMBlock_Enhance
from .lsnet import SKA, LSConv, Block as LightBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision","ScalSeq","Add","Zoom_cat","C3k2_MSEARM","GSConv","EfficientViMBlock", "EfficientViMBlock_Enhance","EfficientVIM_CG_CSP","Slim_Efficient_CSP","A2C2f_CG","C3k2_EDMS",
    "HMSF","CSP_MSAS","MSFF",
    "LayerNorm","Mix","FCA_Attention","GroupGLKA_AttnMap_Simple","TDP_Attention","TDP_Attention_Wrapper",
    "CBR","Semantic_Information_Decoupling","Pred_Layer","ASPP",
    # "CrossAttentionBlock",
    "DSAM_CrossAttention","DSAM_CrossAttention_Wrapper",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """
        Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """
        Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (Tuple[int, int, int]): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        # x = self.cv2(torch.cat(y, 1))
        # print(x.shape)
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """
        Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """
        Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1, c2, k=3, s=1):
        """
        Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (Tuple[int, int]): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """
        Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """
        Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """
        Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """
        Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """
        Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """
        Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """
        Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """
        Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """
        Forward pass of ImagePoolingAttn.

        Args:
            x (List[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """
        Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """
        Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x, w):
        """
        Passes input out unchanged.

        TODO: Update or remove?
        """
        return x

    def forward(self, x, w):
        """
        Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (Tuple[int, int]): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """
        Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """
        Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """
        Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """
        Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """
        Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """
        Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (List[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """
        Initialize CBFuse module.

        Args:
            idx (List[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """
        Forward pass through CBFuse layer.

        Args:
            xs (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """
        Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """
        Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """
        Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """
        Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """
        Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """
        Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """
        Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | List[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, area=1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided, default is 1.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x):
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """
        Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        """
        Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        """
        Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc, ec, e=4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor."""
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x):
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m) -> None:
        """Initialize residual module with the wrapped module."""
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x):
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch, c3, embed):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension."""
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x, vp):
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min

        score = F.softmax(score, dim=-1, dtype=torch.float).to(score.dtype)

        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)

######################################## Attentional ########################################

class Zoom_cat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms

class ScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(ScalSeq, self).__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel,1)
        self.conv1 =  Conv(inc[1], channel,1)
        self.conv2 =  Conv(inc[2], channel,1)
        self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        if hasattr(self, 'conv0'):
            p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = F.interpolate(p4_2, p3.size()[2:], mode='nearest')
        p5_2 = self.conv2(p5)
        p5_2 = F.interpolate(p5_2, p3.size()[2:], mode='nearest')
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x

class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(torch.stack(x, dim=0), dim=0)
###############################################################################################################################################


##################################################################### start ###################################################################
class EdgeAwareRefinementModule(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class MSEARM(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()
        
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeAwareRefinementModule(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.final_conv = Conv(inc * 2, inc)
    
    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(torch.cat(out, 1))

class C3k_MSEARM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MSEARM(c_, [3, 6, 9, 12]) for _ in range(n)))

class C3k2_MSEARM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MSEARM(self.c, self.c, 2, shortcut, g) if c3k else MSEARM(self.c, [3, 6, 9, 12]) for _ in range(n))

############################################################ end ####################################################################


################################################# SlimNeck begin ##############################################

class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

        
######################################## SlimNeck end ########################################


class EfficientVIM_CG_C3SP(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(EfficientViMBlock_Enhance(c_) for _ in range(n)))

class EfficientVIM_CG_CSP(C2f):
    def __init__(self, c1, c2, n=1, e=0.25, g=1, shortcut=True):
        super().__init__(c1, c2, n, e, g, shortcut)
        self.m = nn.ModuleList(EfficientViMBlock_Enhance(self.c) for _ in range(n))

############################################ lightBlock ##########################


class C3k_LSBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(LightBlock(c_) for _ in range(n)))

class C3k2_LSBlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.25, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_LSBlock(self.c, self.c, n, shortcut, g) if c3k else LightBlock(self.c) for _ in range(n))

# class GatedLSBlock_BCHW(nn.Module):
#     r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
#     Args: 
#         conv_ratio: control the number of channels to conduct depthwise convolution.
#             Conduct convolution on partial channels can improve practical efficiency.
#             The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
#             also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
#     """
#     def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
#                  norm_layer=partial(LayerNormGeneral,eps=1e-6,normalized_dim=(1, 2, 3)), 
#                  act_layer=nn.GELU,
#                  drop_path=0.,
#                  **kwargs):
#         super().__init__()
#         self.norm = norm_layer((dim, 1, 1))
#         hidden = int(expansion_ratio * dim)
#         self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
#         self.act = act_layer()
#         conv_channels = int(conv_ratio * dim)
#         self.split_indices = (hidden, hidden - conv_channels, conv_channels)
#         # self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
#         self.conv = LSConv(conv_channels)
#         self.fc2 = nn.Conv2d(hidden, dim, 1)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         shortcut = x # [B, H, W, C]
#         x = self.norm(x)
#         g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
#         # c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
#         c = self.conv(c)
#         # c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
#         x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))
#         x = self.drop_path(x)
#         return x + shortcut

# class C3k_MambaOut_LSConv(C3k):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
#         super().__init__(c1, c2, n, shortcut, g, e, k)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(GatedLSBlock_BCHW(c_) for _ in range(n)))

# class C3k2_MambaOut_LSConv(C3k2):
#     def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
#         super().__init__(c1, c2, n, c3k, e, g, shortcut)
#         self.m = nn.ModuleList(C3k_MambaOut_LSConv(self.c, self.c, n, shortcut, g) if c3k else GatedLSBlock_BCHW(self.c) for _ in range(n))

################################################################################


#######################################end ########################################
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class CG(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
    
    # def forward(self, x):
    #     x, v = self.fc1(x).chunk(2, dim=1)
    #     x = self.dwconv(x) * v
    #     x = self.drop(x)
    #     x = self.fc2(x)
    #     x = self.drop(x)
    #     return x

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x
    

class S_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

class Slim_Efficient_Block(S_Block):
    def __init__(self, dim, mlp_ratio=2, drop_path=0):
        super().__init__(dim, mlp_ratio, drop_path)
        
        self.mlp = CG(dim)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(self.mlp(x))
        return x




class Slim_Efficient_CSP(C3k2):
    # def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.4):
    #     super().__init__(c1, c2, n, shortcut, g, e)
    #     self.m = nn.ModuleList(Slim_Efficient_Block(self.c) for _ in range(n))
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(Slim_Efficient_Block(self.c) for _ in range(n))
######################################## end ########################################

######################################## A2C2f-CG ########################################

class ABlock_CGLU(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CG(dim, mlp_hidden_dim)
    
class A2C2f_CG(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_CGLU(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

########################################  end ########################################


######################################## DynamicConvMixerBlock start ########################################
from einops import rearrange, reduce

class DynamicMixDWConv2d(nn.Module):
    """ Dynamic Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        self.dwconv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, square_kernel_size, padding=square_kernel_size//2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=in_channels)
        ])
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()
        
        # Dynamic Kernel Weights
        self.dkw = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels * 3, 1)
        )
        
    def forward(self, x):
        x_dkw = rearrange(self.dkw(x), 'bs (g ch) h w -> g bs ch h w', g=3)
        x_dkw = F.softmax(x_dkw, dim=0)
        x = torch.stack([self.dwconv[i](x) * x_dkw[i] for i in range(len(self.dwconv))]).sum(0)
        return self.act(self.bn(x))

class DynamicMixer(nn.Module):
    def __init__(self, channel=256, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 2
        
        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(DynamicMixDWConv2d(min_ch, ks, ks * 3 + 2))
        self.conv_1x1 = Conv(channel, channel, k=1)
        
    def forward(self, x):
        _, c, _, _ = x.size()
        x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = torch.cat([self.convs[i](x_group[i]) for i in range(len(self.convs))], dim=1)
        x = self.conv_1x1(x_group)
        return x

class DynamicMixerStructure(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mixer = DynamicMixer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = CG(dim)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class C3k_EDMS(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DynamicMixerStructure(c_) for _ in range(n)))

class C3k2_EDMS(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_EDMS(self.c, self.c, 2, shortcut, g) if c3k else DynamicMixerStructure(self.c) for _ in range(n))
        # self.m = nn.ModuleList(DynamicMixerStructure(self.c) for _ in range(n))






######################################## HMSF Block start ########################################

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True)) 
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class HMSF(nn.Module):
    # Hierarchical Attention Fusion Block
    def __init__(self, inc, ouc, group=False):
        super(HMSF, self).__init__()
        ch_1, ch_2 = inc
        hidc = ouc // 2

        self.lgb1_local = LocalGlobalAttention(hidc, 2)
        self.lgb1_global = LocalGlobalAttention(hidc, 4)
        self.lgb2_local = LocalGlobalAttention(hidc, 2)
        self.lgb2_global = LocalGlobalAttention(hidc, 4)

        self.W_x1 = Conv(ch_1, hidc, 1, act=False)
        self.W_x2 = Conv(ch_2, hidc, 1, act=False)
        self.W = Conv(hidc, ouc, 3, g=4)

        self.conv_squeeze = Conv(ouc * 3, ouc, 1)
        self.rep_conv = RepConv(ouc, ouc, 3, g=(16 if group else 1))
        self.conv_final = Conv(ouc, ouc, 1)

    def forward(self, inputs):
        x1, x2 = inputs
        W_x1 = self.W_x1(x1)
        W_x2 = self.W_x2(x2)
        bp = self.W(W_x1 + W_x2)

        x1 = torch.cat([self.lgb1_local(W_x1), self.lgb1_global(W_x1)], dim=1)
        x2 = torch.cat([self.lgb2_local(W_x2), self.lgb2_global(W_x2)], dim=1)

        return self.conv_final(self.rep_conv(self.conv_squeeze(torch.cat([x1, x2, bp], 1))))

######################################## HMSF ck end ########################################



######################################## Partial Multi-Scale Feature Aggregation Block end ########################################

class MSA(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        
        self.conv1 = Conv(inc, inc, k=3)
        self.conv2 = Conv(inc // 2, inc // 2, k=5, g=inc // 2)
        self.conv3 = Conv(inc // 4, inc // 4, k=7, g=inc // 4)
        self.conv4 = Conv(inc, inc, 1)
    
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)
        conv2_out = self.conv2(conv1_out_1)
        conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)
        conv3_out = self.conv3(conv2_out_1)
        
        out = torch.cat([conv3_out, conv2_out_2, conv1_out_2], dim=1)
        out = self.conv4(out) + x
        return out


class C3k_MSAS(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MSA(c_) for _ in range(n)))

class CSP_MSAS(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MSAS(self.c, self.c, 2, shortcut, g) if c3k else MSA(self.c) for _ in range(n))
        # self.m = nn.ModuleList(DynamicMixerStructure(self.c) for _ in range(n))

########################################  end ########################################



####################################### Focus Diffusion Pyramid Network end ########################################

class MSFF(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
        super().__init__()
        hidc = int(inc[1] * e)
        
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()
        self.conv3 = ADown(inc[2], hidc)
        
        
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
        self.pw_conv = Conv(hidc * 3, hidc * 3)
    
    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        
        x = torch.cat([x1, x2, x3], dim=1)
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        feature = self.pw_conv(feature)
        
        x = x + feature
        return x
        
######################################## Focus Diffusion Pyramid Network end ########################################






########################################################  first idea  ###########################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        return fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))


class FCA_Attention(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(FCA_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)

        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)

        out = self.mix(out1, out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return self.sigmoid(out)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            return self.weight[:, None, None] * (x - u) / torch.sqrt(s + self.eps) + self.bias[:, None, None]


class GroupGLKA_AttnMap_Simple(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        assert n_feats % 2 == 0, "n_feats 必须能被 2 整除"
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        c = n_feats // 2

        self.LKA3 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, groups=c),
            nn.Conv2d(c, c, 5, 1, (5 // 2) * 2, groups=c, dilation=2),
            nn.Conv2d(c, c, 1, 1, 0)
        )
        self.X3 = nn.Conv2d(c, c, 3, 1, 1, groups=c)

        self.LKA7 = nn.Sequential(
            nn.Conv2d(c, c, 7, 1, 7 // 2, groups=c),
            nn.Conv2d(c, c, 9, 1, (9 // 2) * 4, groups=c, dilation=4),
            nn.Conv2d(c, c, 1, 1, 0)
        )
        self.X7 = nn.Conv2d(c, c, 7, 1, 7 // 2, groups=c)

        self.to_spatial_map = nn.Sequential(
            nn.Conv2d(n_feats, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_norm = self.norm(x)
        a_min, a_max = torch.chunk(x_norm, 2, dim=1)
        a = torch.cat([
            self.LKA3(a_min) * self.X3(a_min),
            self.LKA7(a_max) * self.X7(a_max)
        ], dim=1)
        return self.to_spatial_map(a)  # (B, 1, H, W)


class TDP_Attention(nn.Module):
    def __init__(self, c1, c2):  # 输入两个通道数
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.attn_channel = FCA_Attention(c1)
        self.attn_spatial = GroupGLKA_AttnMap_Simple(c1)

        self.fcs = nn.ModuleList([
            nn.Conv2d(c1, c1, 1, padding=0, bias=True)
            for _ in range(2)
        ])
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 1, 1, padding=0, bias=True)
            for _ in range(2)
        ])

        self.soft = nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(c1, c1, 1)
        self.conv2 = nn.Conv2d(c1, c1, 1)

        self.bn1 = nn.BatchNorm2d(c1)
        self.bn2 = nn.BatchNorm2d(c1)

        # 根据 x2 输入通道自动适配到 x1 的通道
        if c1 != c2:
            self.channel_align = nn.Conv2d(c2, c1, 1)
        else:
            self.channel_align = nn.Identity()

        self.output_channels = c1  # 或 c2 或其他你实际输出的通道数

    def forward(self, x1, x2):
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')
        x2 = self.channel_align(x2)
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x = x1 + x2

        temp = self.pool(x)
        t1 = self.attn_channel(temp)
        t2 = self.attn_spatial(x)

        arr1 = [fc(t1) for fc in self.fcs]
        arr2 = [conv(t2) for conv in self.convs]

        #师兄原版
        # y1 = self.conv1(arr1[0] + arr2[0])
        # y2 = self.conv2(arr1[1] + arr2[1])
        y1 = arr1[0] + arr2[0]
        y2 = arr1[1] + arr2[1]

        weights = self.soft(torch.stack([y1, y2], dim=1))  # (B, 2, C, H, W)
        return x1 * weights[:, 0] + x2 * weights[:, 1]



##################################################################################################################################################################



class TDP_Attention_Wrapper(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.attn = TDP_Attention(c1, c2)

    def forward(self, x):
        # x 是一个 list 或 tuple，来自于多个输入层
        x1, x2 = x
        return self.attn(x1, x2)


################################################################### second idea #####################################################################################


####################################################################  修改SID  #########################################################################################


import torch.nn as nn
import torch

# # 定义卷积、批归一化和激活函数的模块
# class CBR(nn.Module):
#     def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
#         super().__init__()
#         self.act = act  # 是否使用激活函数
#         # 定义卷积和批归一化的顺序
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
#             nn.BatchNorm2d(out_c)
#         )
#         self.relu = nn.ReLU(inplace=True)  # 定义ReLU激活函数
#
#     def forward(self, x):
#         x = self.conv(x)  # 执行卷积和批归一化
#         if self.act == True:  # 如果需要激活函数
#             x = self.relu(x)  # 执行ReLU激活
#         return x  # 返回结果
#
#
# # 定义解耦层
# class Semantic_Information_Decoupling(nn.Module):
#     def __init__(self, in_c=1024, out_c=256):
#         super(Semantic_Information_Decoupling, self).__init__()
#         # 定义前景特征提取模块
#         self.cbr_fg = nn.Sequential(
#             CBR(in_c, 512, kernel_size=3, padding=1),
#             CBR(512, out_c, kernel_size=3, padding=1),
#             CBR(out_c, out_c, kernel_size=1, padding=0)
#         )
#         # 定义背景特征提取模块
#         self.cbr_bg = nn.Sequential(
#             CBR(in_c, 512, kernel_size=3, padding=1),
#             CBR(512, out_c, kernel_size=3, padding=1),
#             CBR(out_c, out_c, kernel_size=1, padding=0)
#         )
#         # 定义不确定性特征提取模块
#         self.cbr_uc = nn.Sequential(
#             CBR(in_c, 512, kernel_size=3, padding=1),
#             CBR(512, out_c, kernel_size=3, padding=1),
#             CBR(out_c, out_c, kernel_size=1, padding=0)
#         )
#         # 定义前景分支
#         self.branch_fg = nn.Sequential(
#             CBR(in_c, 256, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/8
#             CBR(256, 256, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/4
#             CBR(256, 128, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/2
#             CBR(128, 64, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1
#             CBR(64, 64, kernel_size=3, padding=1),
#             nn.Conv2d(64, 1, kernel_size=1, padding=0),
#             nn.Sigmoid()  # 使用Sigmoid激活
#         )
#         # 定义背景分支
#         self.branch_bg = nn.Sequential(
#             CBR(in_c, 256, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/8
#             CBR(256, 256, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/4
#             CBR(256, 128, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/2
#             CBR(128, 64, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1
#             CBR(64, 64, kernel_size=3, padding=1),
#             nn.Conv2d(64, 1, kernel_size=1, padding=0),
#             nn.Sigmoid()  # 使用Sigmoid激活
#         )
#         # 定义不确定性分支
#         self.branch_uc = nn.Sequential(
#             CBR(in_c, 256, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/8
#             CBR(256, 256, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/4
#             CBR(256, 128, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1/2
#             CBR(128, 64, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样到1
#             CBR(64, 64, kernel_size=3, padding=1),
#             nn.Conv2d(64, 1, kernel_size=1, padding=0),
#             nn.Sigmoid()  # 使用Sigmoid激活
#         )
#
#     def forward(self, x):
#         # print("before:", x.shape)
#         f_fg = self.cbr_fg(x)  # 前景特征提取
#         f_bg = self.cbr_bg(x)  # 背景特征提取
#         f_uc = self.cbr_uc(x)  # 不确定性特征提取
#         # print("after:", f_fg.shape)
#         # 图中 Auxiliary Head
#         mask_fg = self.branch_fg(f_fg)  # 前景掩码生成
#         mask_bg = self.branch_bg(f_bg)  # 背景掩码生成
#         mask_uc = self.branch_uc(f_uc)  # 不确定性掩码生成
#         return [f_fg, f_bg]
#         # return mask_fg, mask_bg, mask_uc  # 返回三个掩码












# # 定义深度可分离卷积模块
# class CBR(nn.Module):
#     def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1):
#         super(CBR, self).__init__()
#         # 深度卷积
#         self.depthwise = nn.Conv2d(in_c, in_c, kernel_size, stride=stride, padding=padding, dilation=dilation,
#                                    groups=in_c, bias=False)
#         # 逐点卷积
#         self.pointwise = nn.Conv2d(in_c, out_c, 1, stride=1, padding=0, bias=False)
#         # 批归一化
#         self.bn = nn.BatchNorm2d(out_c)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.depthwise(x)  # 深度卷积
#         x = self.pointwise(x)  # 逐点卷积
#         x = self.bn(x)  # 批归一化
#         x = self.relu(x)  # 激活函数
#         return x
#
#
# # 定义修改后的 Semantic_Information_Decoupling 类
# class Semantic_Information_Decoupling(nn.Module):
#     def __init__(self, in_c=256, out_c=64):
#         super(Semantic_Information_Decoupling, self).__init__()
#
#         # 减少通道数，使用深度可分离卷积
#         self.cbr_fg = nn.Sequential(
#             CBR(in_c, 128, kernel_size=3, padding=1),
#             CBR(128, out_c, kernel_size=3, padding=1),
#             CBR(out_c, out_c, kernel_size=1, padding=0)
#         )
#
#         self.cbr_bg = nn.Sequential(
#             CBR(in_c, 128, kernel_size=3, padding=1),
#             CBR(128, out_c, kernel_size=3, padding=1),
#             CBR(out_c, out_c, kernel_size=1, padding=0)
#         )
#
#         self.cbr_uc = nn.Sequential(
#             CBR(in_c, 128, kernel_size=3, padding=1),
#             CBR(128, out_c, kernel_size=3, padding=1),
#             CBR(out_c, out_c, kernel_size=1, padding=0)
#         )
#
#         # 使用更小的通道数和深度可分离卷积来减小参数量
#         self.branch_fg = nn.Sequential(
#             CBR(in_c//4, 32, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#             CBR(32, 32, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#             CBR(32, 16, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#             CBR(16, 16, kernel_size=3, padding=1),
#             nn.Conv2d(16, 1, kernel_size=1, padding=0),
#             nn.Sigmoid()
#         )
#
#         self.branch_bg = nn.Sequential(
#             CBR(in_c//4, 32, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#             CBR(32, 32, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#             CBR(32, 16, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#             CBR(16, 16, kernel_size=3, padding=1),
#             nn.Conv2d(16, 1, kernel_size=1, padding=0),
#             nn.Sigmoid()
#         )
#
#         self.branch_uc = nn.Sequential(
#             CBR(in_c//4, 32, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#             CBR(32, 32, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#             CBR(32, 16, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
#             CBR(16, 16, kernel_size=3, padding=1),
#             nn.Conv2d(16, 1, kernel_size=1, padding=0),
#             nn.Sigmoid()
#         )
#
#
#
#     def forward(self, x):
#         f_fg = self.cbr_fg(x)  # 前景特征提取
#         f_bg = self.cbr_bg(x)  # 背景特征提取
#         f_uc = self.cbr_uc(x)  # 不确定性特征提取
#
#         # 生成掩码
#         mask_fg = self.branch_fg(f_fg)  # 前景掩码
#         mask_bg = self.branch_bg(f_bg)  # 背景掩码
#         mask_uc = self.branch_uc(f_uc)  # 不确定性掩码
#
#
#         return [f_fg, f_bg]  # 返回前景和背景特征图






################################################################### 再次修改SID,改进不确定性区域掩码的生成方式 ##########################################################




class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1):
        super(CBR, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                   groups=in_c, bias=False)
        # 逐点卷积
        self.pointwise = nn.Conv2d(in_c, out_c, 1, stride=1, padding=0, bias=False)
        # 批归一化
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)  # 深度卷积
        x = self.pointwise(x)  # 逐点卷积
        x = self.bn(x)  # 批归一化
        x = self.relu(x)  # 激活函数
        return x





class Semantic_Information_Decoupling(nn.Module):
    def __init__(self, in_c=256, out_c=64, uncertainty_threshold=0.1):
        super(Semantic_Information_Decoupling, self).__init__()

        # 前景、背景特征提取
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 128, kernel_size=3, padding=1),
            CBR(128, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

        self.cbr_bg = nn.Sequential(
            CBR(in_c, 128, kernel_size=3, padding=1),
            CBR(128, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

        # 不确定性区域掩码生成
        self.branch_fg = nn.Sequential(
            CBR(out_c, 32, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(32, 32, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(32, 16, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(16, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.branch_bg = nn.Sequential(
            CBR(out_c, 32, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(32, 32, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(32, 16, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            CBR(16, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.uncertainty_threshold = uncertainty_threshold  # 不确定性阈值
        self.masks = None  # 临时存储掩码，用于损失计算

    def __getstate__(self):
        """在序列化/深拷贝时排除临时 tensor 状态"""
        state = self.__dict__.copy()
        # 排除 masks，因为它是临时 tensor，无法被 deepcopy
        if 'masks' in state:
            state['masks'] = None
        return state

    def __setstate__(self, state):
        """恢复状态时，masks 会被初始化为 None"""
        self.__dict__.update(state)
        if not hasattr(self, 'masks'):
            self.masks = None

    def forward(self, x):
        f_fg = self.cbr_fg(x)  # 前景特征提取
        f_bg = self.cbr_bg(x)  # 背景特征提取

        # 🔥 只在训练阶段生成掩码
        if self.training:
            # 生成掩码
            mask_fg = self.branch_fg(f_fg)  # 前景掩码
            mask_bg = self.branch_bg(f_bg)  # 背景掩码

            # 计算不确定性区域的掩码：前景和背景的预测概率差异小于阈值
            mask_uc = (torch.abs(mask_fg - mask_bg) < self.uncertainty_threshold).float()

            # 保存掩码用于损失计算
            self.masks = (mask_fg, mask_bg, mask_uc)
        else:
            # 推理阶段不生成掩码，节省计算
            self.masks = None

        return [f_fg, f_bg]









# # kiro修改但暂未采纳

# class CBR(nn.Module):
#     def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1):
#         super(CBR, self).__init__()
#         # 深度卷积
#         self.depthwise = nn.Conv2d(in_c, in_c, kernel_size, stride=stride, padding=padding, dilation=dilation,
#                                    groups=in_c, bias=False)
#         # 逐点卷积
#         self.pointwise = nn.Conv2d(in_c, out_c, 1, stride=1, padding=0, bias=False)
#         # 批归一化
#         self.bn = nn.BatchNorm2d(out_c)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.depthwise(x)  # 深度卷积
#         x = self.pointwise(x)  # 逐点卷积
#         x = self.bn(x)  # 批归一化
#         x = self.relu(x)  # 激活函数
#         return x





# class Semantic_Information_Decoupling(nn.Module):
#     def __init__(self, in_c=256, out_c=64, uncertainty_threshold=0.1, target_size=640):
#         super(Semantic_Information_Decoupling, self).__init__()

#         # 前景、背景特征提取
#         self.cbr_fg = nn.Sequential(
#             CBR(in_c, 128, kernel_size=3, padding=1),
#             CBR(128, out_c, kernel_size=3, padding=1),
#             CBR(out_c, out_c, kernel_size=1, padding=0)
#         )

#         self.cbr_bg = nn.Sequential(
#             CBR(in_c, 128, kernel_size=3, padding=1),
#             CBR(128, out_c, kernel_size=3, padding=1),
#             CBR(out_c, out_c, kernel_size=1, padding=0)
#         )

#         # 🔥 修复 1：使用 out_c 而不是 in_c//4
#         # 🔥 修复 2：使用自适应上采样到目标尺寸
#         self.branch_fg = nn.Sequential(
#             CBR(out_c, 32, kernel_size=3, padding=1),  # 🔥 改为 out_c
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             CBR(32, 32, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             CBR(32, 16, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             CBR(16, 16, kernel_size=3, padding=1),
#             nn.Conv2d(16, 1, kernel_size=1, padding=0),
#             nn.Sigmoid()
#         )

#         self.branch_bg = nn.Sequential(
#             CBR(out_c, 32, kernel_size=3, padding=1),  # 🔥 改为 out_c
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             CBR(32, 32, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             CBR(32, 16, kernel_size=3, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
#             CBR(16, 16, kernel_size=3, padding=1),
#             nn.Conv2d(16, 1, kernel_size=1, padding=0),
#             nn.Sigmoid()
#         )

#         self.uncertainty_threshold = uncertainty_threshold  # 不确定性阈值
#         self.target_size = target_size  # 🔥 新增：目标图像尺寸
#         self.masks = None  # 临时存储掩码，用于损失计算

#     def __getstate__(self):
#         """在序列化/深拷贝时排除临时 tensor 状态"""
#         state = self.__dict__.copy()
#         # 排除 masks，因为它是临时 tensor，无法被 deepcopy
#         if 'masks' in state:
#             state['masks'] = None
#         return state

#     def __setstate__(self, state):
#         """恢复状态时，masks 会被初始化为 None"""
#         self.__dict__.update(state)
#         if not hasattr(self, 'masks'):
#             self.masks = None

#     def forward(self, x):
#         f_fg = self.cbr_fg(x)  # 前景特征提取
#         f_bg = self.cbr_bg(x)  # 背景特征提取

#         # 🔥 只在训练阶段生成掩码
#         if self.training:
#             # 生成掩码
#             mask_fg = self.branch_fg(f_fg)  # 前景掩码
#             mask_bg = self.branch_bg(f_bg)  # 背景掩码

#             # 🔥 新增：确保掩码尺寸与目标图像一致
#             if mask_fg.shape[2] != self.target_size or mask_fg.shape[3] != self.target_size:
#                 mask_fg = F.interpolate(
#                     mask_fg, 
#                     size=(self.target_size, self.target_size), 
#                     mode='bilinear', 
#                     align_corners=False
#                 )
#                 mask_bg = F.interpolate(
#                     mask_bg, 
#                     size=(self.target_size, self.target_size), 
#                     mode='bilinear', 
#                     align_corners=False
#                 )

#             # 计算不确定性区域的掩码：前景和背景的预测概率差异小于阈值
#             mask_uc = (torch.abs(mask_fg - mask_bg) < self.uncertainty_threshold).float()

#             # 保存掩码用于损失计算
#             self.masks = (mask_fg, mask_bg, mask_uc)
#         else:
#             # 推理阶段不生成掩码，节省计算
#             self.masks = None

#         return [f_fg, f_bg]








######################################################################### 修改DSAM ##################################################################################


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Pred_Layer(nn.Module):
#     def __init__(self, in_c=256):
#         super(Pred_Layer, self).__init__()
#         self.enlayer = nn.Sequential(
#             nn.Conv2d(in_c, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#         # self.outlayer = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         x = self.enlayer(x)
#         # x1 = self.outlayer(x)
#         # return x, x1
#         return x
#
# class ASPP(nn.Module):
#     def __init__(self, in_c):
#         super(ASPP, self).__init__()
#         self.aspp1 = nn.Sequential(
#             nn.Conv2d(in_c , 256, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#         self.aspp2 = nn.Sequential(
#             nn.Conv2d(in_c , 256, 3, 1, padding=3, dilation=3),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#         self.aspp3 = nn.Sequential(
#             nn.Conv2d(in_c , 256, 3, 1, padding=5, dilation=5),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#         self.aspp4 = nn.Sequential(
#             nn.Conv2d(in_c , 256, 3, 1, padding=7, dilation=7),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         x1 = self.aspp1(x)
#         x2 = self.aspp2(x)
#         x3 = self.aspp3(x)
#         x4 = self.aspp4(x)
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#         return x
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, in_c):
#         super(CrossAttentionBlock, self).__init__()
#         self.query_conv = nn.Conv2d(in_c, in_c // 8, 1)
#         self.key_conv = nn.Conv2d(in_c, in_c // 8, 1)
#         self.value_conv = nn.Conv2d(in_c, in_c, 1)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, feat1, feat2):
#         B, C, H, W = feat1.size()
#         Q = self.query_conv(feat1).view(B, -1, H * W)  # (B, C//8, N)
#         K = self.key_conv(feat2).view(B, -1, H * W)    # (B, C//8, N)
#         V = self.value_conv(feat2).view(B, -1, H * W)  # (B, C, N)
#
#         attention = torch.bmm(Q.permute(0, 2, 1), K)    # (B, N, N)
#         attention = self.softmax(attention / (K.size(1) ** 0.5))
#
#         out = torch.bmm(V, attention.permute(0, 2, 1))  # (B, C, N)
#         out = out.view(B, C, H, W)
#         return out
#
# class DSAM_CrossAttention(nn.Module):
#     def __init__(self, in_c):
#         super(DSAM_CrossAttention, self).__init__()
#
#         self.ff_conv = ASPP(in_c)
#         self.bf_conv = ASPP(in_c)
#         self.cross_attention = CrossAttentionBlock(256 * 4)
#         self.rgbd_pred_layer = Pred_Layer(256 * 8)
#
#     def forward(self, feat, list):   # list接收SID返回的特征图列表
#
#         f_fg, f_bg = list
#         # f_fg = list[0]
#         # f_bg = list[1]
#         # print(f_fg.shape)
#
#         B, C, H, W = feat.size()
#
#         # 对齐空间维度，将f_fg和f_bg对齐到feat的空间维度
#         f_fg = torch.sigmoid(F.interpolate(f_fg, size=(H, W), mode='bilinear', align_corners=True))
#         f_bg = torch.sigmoid(F.interpolate(f_bg, size=(H, W), mode='bilinear', align_corners=True))
#
#         # 使用1x1卷积将 f_fg 和 f_bg 的通道数调整到 feat 相同的通道数
#         # 根据 f_fg 和 f_bg 的通道数动态创建卷积层
#         fg_channels = f_fg.size(1)  # 获取 f_fg 的通道数
#         bg_channels = f_bg.size(1)  # 获取 f_bg 的通道数
#
#         # 动态创建卷积层
#         fg_conv = nn.Conv2d(fg_channels, C, kernel_size=1, stride=1, padding=0)  # 前景通道对齐
#         bg_conv = nn.Conv2d(bg_channels, C, kernel_size=1, stride=1, padding=0)  # 背景通道对齐
#
#         # 对 f_fg 和 f_bg 使用卷积进行通道对齐
#         f_fg = fg_conv(f_fg)
#         f_bg = bg_conv(f_bg)
#
#         # 对特征图feat分别进行前景和背景的增强
#         ff_feat = self.ff_conv(feat * f_fg)
#         bf_feat = self.bf_conv(feat * f_bg)
#
#         # 前后流互注意力增强
#         ff_enhanced = self.cross_attention(ff_feat, bf_feat)
#         bf_enhanced = self.cross_attention(bf_feat, ff_feat)
#
#         fusion = torch.cat((ff_enhanced, bf_enhanced), dim=1)
#         enhanced_feat = self.rgbd_pred_layer(fusion)
#
#         # 在通道维度上拼接enhanced_feat和feat
#         fused_feat = torch.cat((enhanced_feat, feat), dim=1)
#         print(fused_feat.shape)
#
#         return fused_feat





# class Pred_Layer(nn.Module):
#     def __init__(self, in_c=256):
#         super(Pred_Layer, self).__init__()
#         self.enlayer = nn.Sequential(
#             nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#         )
#         # self.outlayer = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         x = self.enlayer(x)
#         # x1 = self.outlayer(x)
#         # return x, x1
#         return x
#
# class ASPP(nn.Module):
#     def __init__(self, in_c):
#         super(ASPP, self).__init__()
#         self.aspp1 = nn.Sequential(
#             nn.Conv2d(in_c , 32, 1, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#         )
#         self.aspp2 = nn.Sequential(
#             nn.Conv2d(in_c , 32, 3, 1, padding=3, dilation=3),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#         )
#         self.aspp3 = nn.Sequential(
#             nn.Conv2d(in_c , 32, 3, 1, padding=5, dilation=5),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#         )
#         self.aspp4 = nn.Sequential(
#             nn.Conv2d(in_c , 32, 3, 1, padding=7, dilation=7),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         x1 = self.aspp1(x)
#         x2 = self.aspp2(x)
#         x3 = self.aspp3(x)
#         x4 = self.aspp4(x)
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#         return x
#
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, in_c):
#         super(CrossAttentionBlock, self).__init__()
#         self.query_conv = nn.Conv2d(in_c, in_c // 8, 1)
#         self.key_conv = nn.Conv2d(in_c, in_c // 8, 1)
#         self.value_conv = nn.Conv2d(in_c, in_c, 1)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, feat1, feat2):
#         B, C, H, W = feat1.size()
#         Q = self.query_conv(feat1).view(B, -1, H * W)  # (B, C//8, N)
#         K = self.key_conv(feat2).view(B, -1, H * W)    # (B, C//8, N)
#         V = self.value_conv(feat2).view(B, -1, H * W)  # (B, C, N)
#
#         attention = torch.bmm(Q.permute(0, 2, 1), K)    # (B, N, N)
#         attention = self.softmax(attention / (K.size(1) ** 0.5))
#
#         out = torch.bmm(V, attention.permute(0, 2, 1))  # (B, C, N)
#         out = out.view(B, C, H, W)
#         return out
#
# class DSAM_CrossAttention(nn.Module):
#     def __init__(self, in_c):
#         super(DSAM_CrossAttention, self).__init__()
#
#         self.ff_conv = ASPP(in_c)
#         self.bf_conv = ASPP(in_c)
#         self.cross_attention = CrossAttentionBlock(32 * 4)
#         self.rgbd_pred_layer = Pred_Layer(32 * 8)
#
#     def forward(self, feat, list):   # list接收SID返回的特征图列表
#
#         f_fg, f_bg = list
#         # f_fg = list[0]
#         # f_bg = list[1]
#         # print(f_fg.shape)
#
#         B, C, H, W = feat.size()
#
#         # 对齐空间维度，将f_fg和f_bg对齐到feat的空间维度
#         f_fg = torch.sigmoid(F.interpolate(f_fg, size=(H, W), mode='bilinear', align_corners=True))
#         f_bg = torch.sigmoid(F.interpolate(f_bg, size=(H, W), mode='bilinear', align_corners=True))
#
#         # 使用1x1卷积将 f_fg 和 f_bg 的通道数调整到 feat 相同的通道数
#         # 根据 f_fg 和 f_bg 的通道数动态创建卷积层
#         fg_channels = f_fg.size(1)  # 获取 f_fg 的通道数
#         bg_channels = f_bg.size(1)  # 获取 f_bg 的通道数
#
#         # 动态创建卷积层
#         fg_conv = nn.Conv2d(fg_channels, C, kernel_size=1, stride=1, padding=0)  # 前景通道对齐
#         bg_conv = nn.Conv2d(bg_channels, C, kernel_size=1, stride=1, padding=0)  # 背景通道对齐
#
#         # 对 f_fg 和 f_bg 使用卷积进行通道对齐
#         f_fg = fg_conv(f_fg)
#         f_bg = bg_conv(f_bg)
#         # print(f_fg.shape)
#
#         # 对特征图feat分别进行前景和背景的增强
#         ff_feat = self.ff_conv(feat * f_fg)
#         bf_feat = self.bf_conv(feat * f_bg)
#         # print(ff_feat.shape)
#
#         # 前后流互注意力增强
#         ff_enhanced = self.cross_attention(ff_feat, bf_feat)
#         bf_enhanced = self.cross_attention(bf_feat, ff_feat)
#         # print(ff_enhanced.shape)
#
#         fusion = torch.cat((ff_enhanced, bf_enhanced), dim=1)
#         # print(fusion.shape)
#
#         enhanced_feat = self.rgbd_pred_layer(fusion)
#         # print(enhanced_feat.shape)
#
#         # 在通道维度上拼接enhanced_feat和feat
#         fused_feat = torch.cat((enhanced_feat, feat), dim=1)
#         # print(fused_feat.shape)
#
#         return fused_feat







##################################################################### 再次修改DSAM 无qkv ###############################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

class Pred_Layer(nn.Module):
    def __init__(self, in_c=256):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x




class ASPP(nn.Module):
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        # 1x1 分支不重，保留
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c, 32, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 下面三个分支：Depthwise(3x3 dilated) + Pointwise(1x1) 代替原 full conv
        def dw_pw(dilation, padding):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, 1, padding=padding, dilation=dilation,
                          groups=in_c, bias=False),          # depthwise
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, 32, 1, 1, bias=False),     # pointwise
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )

        self.aspp2 = dw_pw(dilation=3, padding=3)
        self.aspp3 = dw_pw(dilation=5, padding=5)
        self.aspp4 = dw_pw(dilation=7, padding=7)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x



class DSAM_CrossAttention(nn.Module):
    def __init__(self, in_c, guide_c = 64):    # in_c是上一层输入的通道数，guide_c是前景、背景特征图的通道数
        super(DSAM_CrossAttention, self).__init__()

        self.ff_conv = ASPP(in_c)
        self.bf_conv = ASPP(in_c)
        self.rgbd_pred_layer = Pred_Layer(32 * 8)

        # ✅ 关键：注册到模型里（不在 forward 里创建）
        self.fg_conv = nn.Conv2d(guide_c, in_c, kernel_size=1, stride=1, padding=0)
        self.bg_conv = nn.Conv2d(guide_c, in_c, kernel_size=1, stride=1, padding=0)

    def forward(self, feat, list):  # 不改你的函数签名与变量名
        f_fg, f_bg = list

        B, C, H, W = feat.size()

        # 对齐空间维度（建议 align_corners=False 更稳）
        f_fg = F.interpolate(f_fg, size=(H, W), mode='bilinear', align_corners=False)
        f_bg = F.interpolate(f_bg, size=(H, W), mode='bilinear', align_corners=False)

        # ✅ 先通道对齐，再 sigmoid 作为 gate（你原来是先 sigmoid 再 conv）
        f_fg = torch.sigmoid(self.fg_conv(f_fg))
        f_bg = torch.sigmoid(self.bg_conv(f_bg))

        # 前景增强 / 背景分支（你原有逻辑保留）
        ff_feat = self.ff_conv(feat * f_fg)
        bf_feat = self.bf_conv(feat * f_bg)

        fusion = torch.cat((ff_feat, bf_feat), dim=1)
        enhanced_feat = self.rgbd_pred_layer(fusion)

        fused_feat = torch.cat((enhanced_feat, feat), dim=1)
        return fused_feat








##################################################################################################################################################################


class DSAM_CrossAttention_Wrapper(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.attn = DSAM_CrossAttention(in_c)

    def forward(self, x):
        # x 是一个 list 或 tuple，来自于多个输入层
        x1, x2 = x
        return self.attn(x1, x2)