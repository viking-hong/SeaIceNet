import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import math
import os
import torch.fft as fft
import cv2
import numpy as np
from geoseg.utils.dca_utils import *

os.environ['CURL_CA_BUNDLE'] = ''

'''
GaborCNN
'''


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y_out = self.fc1(y)
        return x * y_out


class GaborConv(nn.Module):
    def __init__(self, kernel_size, in_channels, channel1, eps=1e-8):
        super(GaborConv, self).__init__()
        # 生成滤波器所需的参数
        self.sigma1, self.theta1, self.Lambda1, self.psi1, self.gamma1, self.bias1 = self.generate_parameters(
            channel1 // 2, in_channels)
        self.sigma2, self.theta2, self.Lambda2, self.psi2, self.gamma2, self.bias2 = self.generate_parameters(
            channel1 // 2, in_channels)
        # 生成实部和虚部的gabor滤波器 as a tensor of shape in_channels * channel1//2 * kernel_size * kernel_size
        self.filter_cos = self.whole_filter(in_channels, channel1 // 2, kernel_size, self.sigma1, self.theta1,
                                            self.Lambda1, self.psi1, self.gamma1, True).cuda()
        self.filter_sin = self.whole_filter(in_channels, channel1 // 2, kernel_size, self.sigma1, self.theta1,
                                            self.Lambda1, self.psi1, self.gamma1, False).cuda()
        self.se = SELayer(channel1)
        # the second layer of the network is a conventional CNN layer
        self.conv = nn.Conv2d(64, 64, 3, 2)
        # last two layers of the network are fully connected
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(in_channels, in_channels, kernel_size=3)

    def forward(self, x):
        # 分别用实部和虚部的滤波器对x进行卷积，然后将结果进行拼接
        x_cos = F.conv2d(x, self.filter_cos, bias=self.bias1)  # 4 32 23 23
        x_sin = F.conv2d(x, self.filter_sin, bias=self.bias2)  # 4 32 23 23
        x_comb = torch.cat((x_cos, x_sin), 1)  # 4 64 23 23

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x_comb = fuse_weights[0] * fft_h(x_comb) + fuse_weights[1] * x_comb
        x_comb = self.post_conv(x_comb)

        # x_comb = self.se(x_comb)
        x_size = x.size()[2]

        x_comb = F.max_pool2d(x_comb, 2, 1)
        x_comb = F.relu(self.conv(x_comb))
        x_comb = F.max_pool2d(x_comb, 2, 2)

        x_comb = F.interpolate(x_comb, size=(x_size, x_size), mode='bilinear', align_corners=False)
        ####conv 3 3
        x_comb = self.conv1(x_comb)
        x_comb = F.relu(x_comb)
        ####conv 11
        x_comb = self.conv2(x_comb)

        return F.log_softmax(x_comb, dim=1)

    def generate_parameters(self, dim_out, dim_in):
        sigma = nn.Parameter(torch.randn(dim_out, dim_in))
        theta = nn.Parameter(torch.randn(dim_out, dim_in))
        Lambda = nn.Parameter(torch.randn(dim_out, dim_in))
        psi = nn.Parameter(torch.randn(dim_out, dim_in))
        gamma = nn.Parameter(torch.randn(dim_out, dim_in))
        bias = nn.Parameter(torch.randn(dim_out))
        return sigma, theta, Lambda, psi, gamma, bias

    def one_filter(self, in_channels, kernel_size, sigma, theta, Lambda, psi, gamma, cos):
        # generate Gabor filters as a tensor of shape in_channels * kernel_size * kernel_size
        result = torch.zeros(in_channels, kernel_size, kernel_size)
        # 对每个输入通道生成一个 Gabor 滤波器
        # 然后将这些滤波器拼接成一个形状为 (in_channels, kernel_size, kernel_size) 的张量
        # 并将其作为 nn.Parameter 返回
        for i in range(in_channels):
            result[i] = self.gabor_fn(sigma[i], theta[i], Lambda[i], psi[i], gamma[i], kernel_size, cos)
        return nn.Parameter(result)

    def whole_filter(self, in_channels, out_channels, kernel_size, sigma_column, theta_column, Lambda_column,
                     psi_column, gamma_column, cos):
        # generate Gabor filters as a tensor of shape out_channels * in_channels * kernel_size * kernel_size
        result = torch.zeros(out_channels, in_channels, kernel_size,
                             kernel_size)  # \text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW
        # 对每个输出通道生成一组 Gabor 滤波器
        # 然后将这些滤波器拼接成一个形状为 (out_channels, in_channels, kernel_size, kernel_size) 的张量
        # 并将其作为 nn.Parameter 返回
        for i in range(out_channels):
            result[i] = self.one_filter(in_channels, kernel_size, sigma_column[i], theta_column[i], Lambda_column[i],
                                        psi_column[i], gamma_column[i], cos)
        return nn.Parameter(result)

    def gabor_fn(self, sigma, theta, Lambda, psi, gamma, kernel_size, cos):
        # generate a single Gabor filter, modified https://en.wikipedia.org/wiki/Gabor_filter#Example_implementations
        # 设置 Gabor 滤波器的 x 方向标准差为参数 sigma
        sigma_x = sigma
        # sigma_y = float(sigma) / gamma
        # 根据参数 gamma 计算 Gabor 滤波器的 y 方向标准差
        sigma_y = sigma / gamma

        # Bounding box 设置网格的边界
        half_size = (kernel_size - 1) // 2
        ymin, xmin = -half_size, -half_size
        ymax, xmax = half_size, half_size
        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
        # 将 y 和 x 转换为 PyTorch 的浮点张量
        y = torch.FloatTensor(y)
        x = torch.FloatTensor(x)
        ##############
        # Rotation 根据参数 theta 对网格进行旋转
        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

        if cos:
            gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.cos(
                2 * np.pi / Lambda * x_theta + psi)
        else:
            gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.sin(
                2 * np.pi / Lambda * x_theta + psi)

        ###########
        return gb


def fft_h(x):
    img_fft = torch.fft.fft2(x)
    # 中心化频谱
    img_fft_shifted = torch.fft.fftshift(img_fft)
    # 提取幅度谱
    magnitude_spectrum = torch.abs(img_fft_shifted)
    # 构造高通滤波器
    _, _, rows, cols = x.shape
    center_row, center_col = rows // 2, cols // 2
    mask = torch.ones((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if distance <= 40:
                mask[i, j] = 0
    # 高通滤波
    filtered_spectrum = magnitude_spectrum * mask.cuda()
    # 逆变换
    filtered_img_fft_shifted = img_fft_shifted * filtered_spectrum
    filtered_img_fft = torch.fft.ifftshift(filtered_img_fft_shifted)
    filtered_img = torch.fft.ifft2(filtered_img_fft)
    filtered_image = torch.abs(filtered_img)

    return filtered_image


'''
GaborCNN
'''


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, groups=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, groups=groups,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    ###############################################################
    '''
    新模块开始
    新模块开始
    新模块开始
    '''


from pytorch_wavelets import DWT2D, IDWT2D

'''
这部分是信号重构的部分
'''


class ConvGuidedFilter(nn.Module):
    def __init__(self, dim=256, radius=1, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()
        self.dim = dim
        self.box_filter = nn.Conv2d(dim, dim, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=8)
        self.conv_a = nn.Sequential(nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False),
                                    norm(dim),
                                    nn.ReLU(inplace=True),
                                    # nn.LeakyReLU(inplace=True)
                                    # nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                                    # norm(32),
                                    # nn.ReLU(inplace=True),
                                    # nn.Conv2d(dim, dim, kernel_size=1, bias=False)
                                    )
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):  ##########y_lr 自注意力输入，x_lr引导图->残差的小波细节, x_hr残差
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, self.dim, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr) / N  # G        G_ave = self.ave(G)   # 2
        ## mean_y
        mean_y = self.box_filter(y_lr) / N  # F        F_ave = self.ave(F)   # 1
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y  # GF_var = GF_ave - F_ave*G_ave           # 6
        ## var_x
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x  # G_var = GG_ave - G_ave*G_ave + self.eps # 5

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))  # GF_var/G_var.expand_as(GF_var)   # 7
        ## b
        b = mean_y - A * mean_x  # F_ave - a*G_ave

        '''
        ##############################
        '''
        ## mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        #       a = self.ave(a)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        #       b = self.ave(b)
        '''
        ################################
        '''

        return mean_A * x_hr + mean_b  # out = a * G.expand_as(a) + b   # 10


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
        )

        self.gb = GaborConv(kernel_size=3, in_channels=dim, channel1=dim)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='constant')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='constant')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='constant')  # reflect
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        '''
        局部分支
        '''
        local = self.local2(x) + self.local1(x)  # 4 64 25 25
        local = self.gb(local)

        '''
        全局分支
        '''
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2,
                                -1)) * self.scale  #######torch.Size([64, 8]) torch.Size([8, 64]) -> torch.Size([64, 64])

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            # print(relative_position_bias.unsqueeze(0).shape)
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v  # torch.Size([64, 64]) torch.Size([64, 8]) -> torch.Size([64, 8])
        # attn = tpv.view(channel, bs, -1, n)

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]
        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        '''
        全局分支
        '''
        out = out + local

        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x_clone = x.clone()
        x_comb = self.norm1(x)

        # for cutoff_freq in range(0, 101, 10):
        #     x = x_clone + self.drop_path(self.attn(x_comb, cutoff_freq))
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        #
        #     save_first_layer(x, f'D:/seaice_1024/view/viewer/gabor_cutoff_{cutoff_freq}.jpg')

        x = x_clone + self.drop_path(self.attn(x_comb))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


'''
网络之间的连接部分在这
'''


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.pre_conv_dwt = Conv(decode_channels * 4, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.DGF = ConvGuidedFilter(dim=decode_channels)
        self.dwt = DWT2D(wave='haar')

    def forward(self, x, res):
        '''
        核心修改部分
        '''
        xup = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        res = self.pre_conv(res)
        dwt = self.dwt(res)  # return: (cA, (cH, cV, cD))要注意返回的值，分别为低频分量，水平高频、垂直高频、对角线高频。高频的值包含在一个tuple中。
        guid = self.pre_conv_dwt(
            torch.cat((dwt[0], dwt[1][0][:, :, 0, :, :], dwt[1][0][:, :, 1, :, :], dwt[1][0][:, :, 2, :, :]), dim=1))
        x = self.DGF(guid, x, xup)  # 16, 64, 16, 16
        '''
        核心修改部分
        '''
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)

        return x


class WF1(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF1, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        res = self.pre_conv(res)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


'''
网络之间的连接部分在这
'''


class ChannelAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = depthwise_projection(in_features=out_features,
                                          out_features=out_features,
                                          groups=out_features)
        self.k_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)
        self.v_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)

        self.projection = depthwise_projection(in_features=out_features,
                                               out_features=out_features,
                                               groups=out_features)
        self.sdp = ScaleDotProduct()
        self.Conv1 = ConvBNReLU(in_features, in_features, stride=1, kernel_size=1)
        self.Conv2 = ConvBNReLU(in_features, in_features, stride=1, kernel_size=1)
        self.Conv3 = ConvBNReLU(in_features, in_features, stride=1, kernel_size=1)  # , groups=in_features

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c_q = q.shape
        c = k.shape[2]
        scale = c ** -0.5
        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        v = v.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        att = self.sdp(q, k, v, scale).permute(0, 3, 1, 2).flatten(2)
        att = self.projection(att)
        return att


class SpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = depthwise_projection(in_features=out_features,
                                          out_features=out_features,
                                          groups=out_features)
        self.k_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)
        self.v_map = depthwise_projection(in_features=in_features,
                                          out_features=in_features,
                                          groups=in_features)

        self.projection = depthwise_projection(in_features=out_features,
                                               out_features=out_features,
                                               groups=out_features)
        self.sdp = ScaleDotProduct()
        self.Conv1 = ConvBNReLU(in_features, in_features, stride=1, kernel_size=1)
        self.Conv2 = ConvBNReLU(in_features, in_features, stride=1, kernel_size=1)
        self.Conv3 = ConvBNReLU(in_features, in_features, stride=1, kernel_size=1)

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c = q.shape
        c_v = v.shape[2]
        scale = (c // self.n_heads) ** -0.5
        q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k, v, scale).transpose(1, 2).flatten(2)
        x = self.projection(att)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels, decode_channels):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels // 16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels // 16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

        self.dwt = DWT2D(wave='haar')
        self.pre_conv_dwt = Conv(decode_channels * 4, decode_channels, kernel_size=1)
        self.DGF = ConvGuidedFilter(dim=decode_channels)

        '''
        改进部分
        '''

        self.CAttention = ChannelAttention(in_channels * 4, decode_channels * 4)
        self.SAttention = SpatialAttention(in_channels * 4, decode_channels * 4)
        '''
        改进部分
        '''

    def forward(self, x, res):
        '''
        这部分还是WF的内容
        '''
        '''
        #########
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        #############
        '''
        # save_first_layer(x, 'D:/seaice_1024/view/viewer/GLFF.jpg')

        xup = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        res = self.pre_conv(res)
        dwt = self.dwt(res)  # return: (cA, (cH, cV, cD))要注意返回的值，分别为低频分量，水平高频、垂直高频、对角线高频。高频的值包含在一个tuple中。
        guid = self.pre_conv_dwt(
            torch.cat((dwt[0], dwt[1][0][:, :, 0, :, :], dwt[1][0][:, :, 1, :, :], dwt[1][0][:, :, 2, :, :]), dim=1))

        x = self.DGF(guid, x, xup)  # 16, 64, 16, 16

        # save_first_layer(x, 'D:/seaice_1024/view/viewer/DGD.jpg')

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x

        x = self.post_conv(x)  # 4,64,256,256
        shortcut = self.shortcut(x)
        fusion_result = x
        '''
        shortcut = self.shortcut(x)
        '''

        '''
        channel_path = self.ca(x) * x
        spatial_path = self.pa(x) * x
        '''
        judge = 0
        if x.shape[2] != 256:
            judge = 1
            x = torch.nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        channel_path = self.CAttention(x)
        spatial_path = self.SAttention(x)

        if judge == 1:
            channel_path = channel_path.unsqueeze(0)
            spatial_path = spatial_path.unsqueeze(0)
            channel_path = F.interpolate(channel_path, size=(128, 128), mode='bilinear',
                                         align_corners=False)
            spatial_path = F.interpolate(spatial_path, size=(128, 128), mode='bilinear',
                                         align_corners=False)

        channel_path = channel_path.squeeze(0)
        spatial_path = spatial_path.squeeze(0)

        channel_path = channel_path * fusion_result
        spatial_path = spatial_path * fusion_result

        sup = spatial_path + channel_path
        sup = self.proj(sup) + shortcut
        sup = self.act(sup)

        # save_first_layer(sup, 'D:/seaice_1024/view/viewer/DCAH.jpg')

        return sup


'''
新模块结束
新模块结束
新模块结束
###################################################################
'''


def save_first_layer(image_tensor, output_image_path):
    first_layer = image_tensor[0, 0, :, :]

    first_layer_np = first_layer.detach().cpu().numpy()
    first_layer_np = (
            (first_layer_np - first_layer_np.min()) / (first_layer_np.max() - first_layer_np.min()) * 255).astype(
        np.uint8)

    cv2.imwrite(output_image_path, first_layer_np)


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        if self.training:
            x = self.b4(self.pre_conv(res4))
            h4 = self.up4(x)

            x = self.p3(x, res3)
            x = self.b3(x)
            h3 = self.up3(x)

            x = self.p2(x, res2)
            x = self.b2(x)
            h2 = x
            x = self.p1(x, res1)
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)

            return x, ah
        else:
            x = self.b4(self.pre_conv(res4))
            x = self.p3(x, res3)
            x = self.b3(x)

            x = self.p2(x, res2)
            x = self.b2(x)

            x = self.p1(x, res1)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SeaIceNet(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='resnet18.fb_swsl_ig1b_ft_in1k',
                 pretrained=True,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        if self.training:
            x, ah = self.decoder(res1, res2, res3, res4, h, w)
            return x, ah
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x