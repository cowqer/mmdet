import torch
import torch.nn as nn
import torch.nn.functional as F

    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations, bias=False),###, bias=False
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    
class SElayer(nn.Module):
    def __init__(self, channel, rediction=16):
        super(SElayer, self).__init__()
        self.globalavgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // rediction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // rediction  , channel *4),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, _, _ = x.size()
        y = self.globalavgpool(x).view(batch, channel)
        y = self.fc(y).view(batch, channel*4, 1, 1)
        return  y


class SSGpconv(nn.Module):
    ''' Pinwheel-shaped Convolution with Gating Mechanism '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.c2 = c2
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = nn.ModuleList([nn.ZeroPad2d(p[i]) for i in range(4)])
        
        # Branch convolutions
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        
        self.se = SElayer(c2 // 4)
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c2 // 4 * 4, kernel_size=1),  # 输出 [B, c2 // 4 * 4, 1, 1]
            nn.Sigmoid()
        )

        # Final fusion layer
        self.fusion = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        # Compute feature maps for each direction
        yw0 = self.cw(self.pad[0](x))  # Horizontal-1
        yw1 = self.cw(self.pad[1](x))  # Horizontal-2
        yh0 = self.ch(self.pad[2](x))  # Vertical-1
        yh1 = self.ch(self.pad[3](x))  # Vertical-2
        
        y = yw0 + yw1 + yh0 + yh1
        
        gate = self.se(y)
        
        assert gate.shape[1] % 4 == 0, "Number of channels should be divisible by 4"
        
        gate = torch.split(gate, self.c2 // 4, dim=1)

        gate = [torch.sigmoid(g) for g in gate]
        
        # Apply learned gating weights to each branch
        yw0 = gate[0] * yw0
        yw1 = gate[1] * yw1
        yh0 = gate[2] * yh0
        yh1 = gate[3] * yh1

        output = self.fusion(torch.cat([yw0, yw1, yh0, yh1], dim=1))

        return output

if __name__ == "__main__":
    # Create a random input tensor of shape (batch_size, channels, height, width)
    x = torch.randn(1, 3, 64, 64)  # 1 image, 3 channels, 64x64 size
    
    # Create an instance of PConv
    apconv = GatedPConv(c1=3, c2=128, k=3, s=1 )# output channels = 64
    
    # Forward pass
    output = apconv(x)
    
    # Print output shape
    print("Output shape:", output.shape)
