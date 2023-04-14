import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [1,1,1,1,1/2,1/4,1/8,1/16,1/32]

class WSConv2d(nn.Module):

    def __init__(
        self, in_channels, out_channels, k=3, s=1, padding=1, gain=2
    ):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, padding)
        self.scale = (gain / (in_channels * (k ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.e = 1e-8
    def forward(self,x):
        return x / torch.sqrt(torch.mean(x**2,dim=1,keepdim=True) + self.e) # 在通道维度进行标准化

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,use_pn = True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pn
        self.conv1 = WSConv2d(in_channels,out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leak = nn.LeakyReLU(0.2)
        self.pn=PixelNorm()

    def forward(self,x):
        x = self.leak(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leak(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs):
        return inputs * nn.Sigmoid(inputs)

class Generator(nn.Module):
    def __init__(self,z_dim,in_channels,img_channels):
        super(Generator, self).__init__()
        self.init = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim,in_channels,4,1,0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels,in_channels,k=3,s=1,padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        self.init_rgb = WSConv2d(in_channels,img_channels,k=1,s=1,padding=0)
        self.prog_blocks,self.rgb_blocks = nn.ModuleList(),nn.ModuleList([self.init_rgb]) # 这个地方的ggb是因为上面的初始化会有一个对应的rgb

        for i in range(len(factors) -1): #因为长度减去了一个 就不会超range
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_c,conv_out_c))
            self.rgb_blocks.append(WSConv2d(conv_out_c,img_channels,k=1,s=1,padding=0))

    def fade_in(self,alpha,upscaled,generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self,x,alpha,steps):
        out = self.init(x)
        if steps == 0:
            return self.init_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out ,scale_factor=2,mode='nearest')
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_blocks[steps-1](upscaled)
        final_out = self.rgb_blocks[steps](out)
        return self.fade_in(alpha=alpha,upscaled=final_upscaled,generated=final_out) # 图像融合

class Discriminator(nn.Module):
    def __init__(self,z_dim,in_channels,img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks,self.rgb_layers = nn.ModuleList(),nn.ModuleList()
        self.leak = nn.LeakyReLU(0.2)

        for i in range(len(factors)-1,0,-1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i-1])
            self.prog_blocks.append(ConvBlock(conv_in_c,conv_out_c))
            self.rgb_layers.append(WSConv2d(img_channels,conv_in_c,k=1,s=1,padding=0))

        self.initial_rgb = WSConv2d(img_channels,in_channels,k=1,s=1,padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)

        self.final_block = nn.Sequential(
            WSConv2d(in_channels+1,in_channels,k=3,s=1,padding=1),  # 为什么要加一是因为后面我们要加一个batch
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, k=4, s=1,padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, k=1, s=1, padding=0),
        )

    def fade_in(self, alpha, downscale, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * downscale)

    def minibatch_std(self,x):
        batch_statistics = torch.std(x,dim=0).mean().repeat(x.shape[0],1,x.shape[2],x.shape[3])
        return torch.cat([x,batch_statistics],dim=1)

    def forward(self,x,alpha,steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.leak(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0],-1) # 将后面进行展开，确保获得[?,1]的数据

        downscale = self.leak(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha,downscale,out)

        for step in range(cur_step + 1,len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0],-1)

if __name__ =="__main__":
    Z_DIM = 50
    IN_CHANNELS = 256
    gen = Generator(Z_DIM,IN_CHANNELS,img_channels=3)
    critic = Discriminator(Z_DIM,IN_CHANNELS,img_channels=3)
    x = torch.randn((4,Z_DIM,1,1))
    z = gen(x,0.5,4)

    for img_size in [4,8,16,32,64,128,258,512,1024]:
        num_steps = int(log2(img_size / 4))
        print("num:{}".format(num_steps))
        x = torch.randn((2,Z_DIM,1,1))
        z = gen(x,0.5,num_steps)
        print(z.shape)
        out = critic(z,alpha=0.5,steps=num_steps)
        print("out=={}".format(out.shape))








































































