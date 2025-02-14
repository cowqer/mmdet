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
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class SGPConv(nn.Module):
    ''' Pinwheel-shaped Convolution with Gating Mechanism '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.c2 = c2
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = nn.ModuleList([nn.ZeroPad2d(p[i]) for i in range(4)])
        
        # Branch convolutions
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.SE1 = oneConv(c2 // 4,c2 // 4 ,1,0,1)
        self.SE2 = oneConv(c2 // 4,c2 // 4 ,1,0,1)
        self.SE3 = oneConv(c2 // 4,c2 // 4 ,1,0,1)
        self.SE4 = oneConv(c2 // 4,c2 // 4 ,1,0,1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.softmax = nn.Softmax(dim = 2)
        self.softmax_1 = nn.Sigmoid()
        # Final fusion layer
        self.conv1x1= nn.Conv2d(c1, c2//4, kernel_size=1)
        self.fusion0 = Conv(c2 // 4, c2 , 2, s=1, p=0)
        self.fusion = nn.Sequential(
            nn.Conv2d(c2 // 4, c2, 2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(),)
        
    def forward(self, x):
        # Compute feature maps for each direction 
        yw0 = self.cw(self.pad[0](x))  # Horizontal-1
        yw1 = self.cw(self.pad[1](x))  # Horizontal-2
        yh0 = self.ch(self.pad[2](x))  # Vertical-1
        yh1 = self.ch(self.pad[3](x))  # Vertical-2
        ### shape: 4 * [1, 32, 65, 65]
        
        # Compute gate weights
        y0_weight = self.SE1(self.gap(yw0))
        y1_weight = self.SE2(self.gap(yw1))
        y2_weight = self.SE3(self.gap(yh0))
        y3_weight = self.SE4(self.gap(yh1))
        #### shape: 4 * [1, 32, 1, 1]
        
        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight],2)
        # print(weight.shape) 
        ### shape: [1, 32, 4, 1]
        
        weight = self.softmax(self.softmax_1(weight))
        # print(weight.shape) ### shape: [1, 32, 4, 1]
        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)
        
        ### shape: 4 * [1, 32, 1, 1]
        x_att = y0_weight*yw0+y1_weight*yw1+y2_weight*yh0+y3_weight*yh1 ###
        # print(x_att.shape) #torch.Size([1, 32, 65, 65])
        x = self.conv1x1(F.interpolate(x, size=(x_att.size(2), x_att.size(3)), mode='bilinear', align_corners=False))
        # Weighted sum instead of simple concatenation
        output = self.fusion(x_att+x) 
        # print(output.shape)
        return output

if __name__ == "__main__":
    # Create a random input tensor of shape (batch_size, channels, height, width)
    x = torch.randn(1, 3, 64, 64)  # 1 image, 3 channels, 64x64 size
    
    # Create an instance of PConv
    apconv = GatedPConv(c1=3, c2=128, k=3, s=1)#
    
    # Forward pass
    output = apconv(x)
    
    # Print output shape
    print("Output shape:", output.shape)
