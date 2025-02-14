import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PConv(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()

        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        
    def forward(self, x):
        return [pad(x) for pad in self.pad]  # 返回所有 pad 结果，方便可视化

# 生成一个简单的测试输入 (单通道)
x = torch.zeros((1, 1, 5, 5))  
x[:, :, 2, 2] = 1  # 在中心放一个值，方便观察 padding 变化

# 初始化 PConv 并进行 padding
model = PConv(c1=1, c2=1, k=2, s=1)
padded_results = model(x)

# 可视化 padding 结果
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
axs[0].imshow(x[0, 0].numpy(), cmap='gray')
axs[0].set_title("Original Input")

titles = ["Left Pad", "Right Pad", "Top Pad", "Bottom Pad"]
for i in range(4):
    axs[i+1].imshow(padded_results[i][0, 0].numpy(), cmap='gray')
    axs[i+1].set_title(titles[i])

plt.show()
