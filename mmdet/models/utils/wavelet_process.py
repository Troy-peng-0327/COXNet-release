import torch
from torch import nn
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from torch.nn.functional import kl_div
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import math
import pywt
import matplotlib.pyplot as plt


class BasicConv2d(nn.Module):
    # 标准卷积块
    def __init__(
            self,
            in_channels,
            out_channls,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channls,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channls)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class TransBasicConv2d(nn.Module):
    # 标准反卷积块 上采样
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=1,
            output_padding=0,
            bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.inch = in_channels
    
    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class DWTC(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            mode='upsample',
            inner_channel=64):
        super(DWTC, self).__init__()
        self.mode = mode
        if mode == 'up_new':
            self.DWT = DWT_2D(wavename='haar')
            self.IWT = IDWT_2D(wavename='haar')
            self.deconv = TransBasicConv2d(out_channel, out_channel)
            self.hconv = BasicConv2d(out_channel * 2, out_channel)
            self.fusion_conv = BasicConv2d(out_channel * 2, out_channel)
            self.softmax = nn.Softmax(dim=1)

        elif mode == 'up_new_deconv_wo':
            self.DWT = DWT_2D(wavename='haar')
            self.IWT = IDWT_2D(wavename='haar')
            self.hconv = BasicConv2d(out_channel * 2, out_channel)
            self.fusion_conv = BasicConv2d(out_channel * 2, out_channel)
            self.softmax = nn.Softmax(dim=1)

        elif mode == 'up_new_dwt_wo':
            self.deconv = TransBasicConv2d(out_channel, out_channel)

        elif mode == 'up_new_dwt_wo_deconv_wo':
            pass

        else:
            self.DWT = DTCWTForward(
                J=3,
                biort='near_sym_b',
                qshift='qshift_b')
            self.IWT = DTCWTInverse(
                biort='near_sym_b',
                qshift='qshift_b')

        if mode == 'upsample':
            self.proj_conv = BasicConv2d(in_channel, out_channel)
            self.deconv = TransBasicConv2d(out_channel, out_channel)

            self.conv = BasicConv2d(out_channel, out_channel)
            self.out_conv = BasicConv2d(out_channel, out_channel)

        if mode == 'upsample_v4':
            self.proj_conv = BasicConv2d(in_channel, out_channel)
            self.deconv = TransBasicConv2d(out_channel, out_channel)
            self.fusion_conv = BasicConv2d(out_channel * 2, out_channel)
            self.conv = BasicConv2d(out_channel, out_channel)
            self.out_conv = BasicConv2d(out_channel, out_channel)

        if mode == 'upsample_v5' or mode == 'upsample_v7':
            # self.proj_conv = BasicConv2d(in_channel, out_channel)
            self.deconv = TransBasicConv2d(out_channel, out_channel)
            self.fusion_conv = BasicConv2d(out_channel * 2, out_channel, dilation=3, padding=3)
            # self.conv = BasicConv2d(out_channel, out_channel)
            # self.out_conv = BasicConv2d(out_channel, out_channel)

        if mode == 'upsample_v6' or mode == 'upsample_v8':
            # self.proj_conv = BasicConv2d(in_channel, out_channel)
            # self.deconv = TransBasicConv2d(out_channel, out_channel)
            self.fusion_conv = BasicConv2d(out_channel * 2, out_channel, dilation=3, padding=3)
            # self.conv = BasicConv2d(out_channel, out_channel)
            # self.out_conv = BasicConv2d(out_channel, out_channel)
        
        if mode == 'fusion_v2':
            self.proj_conv = BasicConv2d(in_channel, out_channel)
            self.conv = BasicConv2d(out_channel * 2, out_channel)
            self.out_conv = BasicConv2d(out_channel * 2, out_channel)

        if mode == 'fusion_v3':
            self.proj_conv = BasicConv2d(in_channel, out_channel)
            self.conv = BasicConv2d(out_channel * 2, out_channel)
            self.out_conv = BasicConv2d(out_channel * 2, out_channel)
        
        self.iter = -1

    def forward(self, x, y):
        if self.mode == 'upsample':
            y = self.deconv(self.proj_conv(y))

            x_l, x_h = self.DWT(x)
            y_l, y_h = self.DWT(y)
            xy_l = self.conv(x_l) + self.conv(y_l)

            x_m = self.IWT((xy_l, x_h))
            y_m = self.IWT((xy_l, y_h))
            # out = self.out_conv(x_m, y_m)
            return x_m, y_m
        
        if self.mode == 'upsample_v4':
            y = self.deconv(self.proj_conv(y))

            x_l, x_h = self.DWT(x)
            y_l, y_h = self.DWT(y)
            xy_l = self.fusion_conv(torch.cat((x_l, y_l), dim=1))

            # x_m = self.IWT((xy_l, x_h))
            y_m = self.IWT((xy_l, y_h))
            # out = self.out_conv(x_m, y_m)
            return y_m
        
        if self.mode == 'upsample_v5':
            y = self.deconv(self.proj_conv(y))

            x_l, x_h = self.DWT(x)
            y_l, y_h = self.DWT(y)
            # x_h = self.conv(x_h)
            xy_l = self.fusion_conv(torch.cat((x_l, y_l), dim=1))

            # x_m = self.IWT((xy_l, x_h))
            y_m = self.IWT((xy_l, y_h))
            # out = self.out_conv(x_m, y_m)
            return y_m
        
        if self.mode == 'upsample_v6':
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)

            x_l, x_h = self.DWT(x)
            y_l, y_h = self.DWT(y)
            # x_h = self.conv(x_h)
            xy_l = self.fusion_conv(torch.cat((x_l, y_l), dim=1))

            # x_m = self.IWT((xy_l, x_h))
            y_m = self.IWT((xy_l, y_h))
            # out = self.out_conv(x_m, y_m)
            return y_m
        
        if self.mode == 'upsample_v7':
            y = self.deconv(self.proj_conv(y))

            x_l, x_h = self.DWT(x)
            y_l, y_h = self.DWT(y)
            # x_h = self.conv(x_h)
            xy_l = self.fusion_conv(torch.cat((x_l, y_l), dim=1))

            # x_m = self.IWT((xy_l, x_h))
            y_m = self.IWT((xy_l, y_h)) + y
            # out = self.out_conv(x_m, y_m)
            return y_m
        
        if self.mode == 'upsample_v8':
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)

            x_l, x_h = self.DWT(x)
            y_l, y_h = self.DWT(y)
            # x_h = self.conv(x_h)
            xy_l = self.fusion_conv(torch.cat((x_l, y_l), dim=1))

            # x_m = self.IWT((xy_l, x_h))
            y_m = self.IWT((xy_l, y_h)) + y
            # out = self.out_conv(x_m, y_m)
            return y_m
        
        if self.mode == 'up_new':
            # self.iter += 1
            # H = y.shape[2]
            # if H == 64:
            #     _y = y
            #     # draw_feature_map(y, name='y', iter=self.iter)
            #     y = self.deconv(y)

            #     x_ll, _, _, _ = self.DWT(x, mode='full')
            #     # draw_feature_map(x_ll, name='x_ll', iter=self.iter)
            #     y_ll, y_lh, y_hl, y_hh = self.DWT(y, mode='full')
            #     # draw_feature_map(y_ll, name='y_ll', iter=self.iter)
            #     # draw_feature_map(y_lh, name='y_lh', iter=self.iter)
            #     # draw_feature_map(y_hl, name='y_hl', iter=self.iter)
            #     # draw_feature_map(y_hh, name='y_hh', iter=self.iter)

            #     xy_ll = self.fusion_conv(torch.cat((x_ll, y_ll), dim=1))
            #     # draw_feature_map(xy_ll, name='xy_ll', iter=self.iter)
            #     y_h = self.softmax(self.hconv(torch.cat((y_lh, y_hl), dim=1)))
            #     # draw_feature_map(y_h, name='y_h', iter=self.iter)
            #     attn = torch.mul(xy_ll, y_h)
            #     # draw_feature_map(attn, name='attn', iter=self.iter)
            #     _xy_ll = torch.add(xy_ll, attn)
            #     # draw_feature_map(_xy_ll, name='_xy_ll', iter=self.iter)

            #     y_m = self.IWT(_xy_ll, LH=y_lh, HL=y_hl, HH=y_hh)
            #     # draw_feature_map(y_m, name='y_m', iter=self.iter)
            #     return y_m
            # else:
            y = self.deconv(y)

            x_ll, _, _, _ = self.DWT(x, mode='full')
            y_ll, y_lh, y_hl, y_hh = self.DWT(y, mode='full')

            xy_ll = self.fusion_conv(torch.cat((x_ll, y_ll), dim=1))
            y_h = self.softmax(self.hconv(torch.cat((y_lh, y_hl), dim=1)))
            attn = torch.mul(xy_ll, y_h)
            _xy_ll = torch.add(xy_ll, attn)

            y_m = self.IWT(_xy_ll, LH=y_lh, HL=y_hl, HH=y_hh)
            return y_m


        if self.mode == 'up_new_deconv_wo':
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)
        
            x_ll, _, _, _ = self.DWT(x, mode='full')
            y_ll, y_lh, y_hl, y_hh = self.DWT(y, mode='full')

            xy_ll = self.fusion_conv(torch.cat((x_ll, y_ll), dim=1))
            y_h = self.softmax(self.hconv(torch.cat((y_lh, y_hl), dim=1)))
            attn = torch.mul(xy_ll, y_h)
            _xy_ll = torch.add(xy_ll, attn)

            y_m = self.IWT(_xy_ll, LH=y_lh, HL=y_hl, HH=y_hh)
            
            return y_m
        
        if self.mode == 'up_new_dwt_wo':
            y_m = self.deconv(y)

        if self.mode == 'up_new_dwt_wo_deconv_wo':
            y_m = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)
            
            return y_m

        
        elif self.mode == 'fusion_v2':
            y = self.proj_conv(y)
            x_l, x_h = self.DWT(x)
            y_l, y_h = self.DWT(y)
            xy_l = self.conv(torch.cat((x_l, y_l), dim=1))
            x_m = self.IWT((xy_l, x_h))
            y_m = self.IWT((xy_l, y_h))
            out = self.out_conv(torch.cat((x_m, y_m), dim=1))
            return out
        
        elif self.mode == 'fusion_v3':
            x_l, x_h = self.DWT(x)
            y_l, y_h = self.DWT(y)
            xy_l = self.conv(torch.cat((x_l, y_l), dim=1))
            x_m = self.IWT((xy_l, x_h))
            y_m = self.IWT((xy_l, y_h))
            out = self.out_conv(torch.cat((x_m, y_m), dim=1))
            return out
    

def visualize_wavelet_components(components, title, save_path):
    if isinstance(components, torch.Tensor):
        components = components.cpu().detach().numpy()

    # 计算通道最大值
    max_map = components.max(axis=0)
    plt.imshow(max_map, cmap='viridis')
    plt.axis('off')
    plt.title(f"{title} - Max")
    plt.savefig(f"{save_path}_max.png")
    plt.close()


class DWT_2D(nn.Module):
    def __init__(
            self,
            wavename='haar'):
        super(DWT_2D, self).__init__()
        wavelet = _as_wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self, input_height, input_width):
        # 生成低频 (\mathcal{L}) 和高频 (\mathcal{H}) 的变换矩阵，用于二维数据的分解。
        L1 = np.max((input_height, input_width))
        L = math.floor(L1 / 2)

        # 初始化滤波矩阵 (低频和高频)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        # 填充低频矩阵 (matrix_h)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2  # 滤波器步长为 2，确保下采样

        # 提取矩阵的适当部分
        matrix_h_0 = matrix_h[0:(math.floor(input_height / 2)),
                              0:(input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(input_width / 2)),
                              0:(input_width + self.band_length - 2)]
        
        # 填充高频矩阵 (matrix_g)
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2  # 滤波器步长为 2，确保下采样

        # 提取高频矩阵的适当部分
        matrix_g_0 = matrix_g[0:(input_height - math.floor(input_height / 2)),
                              0:(input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(input_width - math.floor(input_width / 2)),
                              0:(input_width + self.band_length - 2)]

        # 对矩阵进行裁剪，去除不必要的边缘元素
        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)  # 转置矩阵，使其适应后续的矩阵乘法
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        # 将矩阵转换为 PyTorch 张量，并根据 GPU 的可用性决定是否移动到 CUDA
        if torch.cuda.is_available():
            matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            matrix_low_0 = torch.Tensor(matrix_h_0)
            matrix_low_1 = torch.Tensor(matrix_h_1)
            matrix_high_0 = torch.Tensor(matrix_g_0)
            matrix_high_1 = torch.Tensor(matrix_g_1)
        return matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1
    
    def forward(self, input, mode='full'):
        # 前向传播，计算输入数据的低频和高频分量
        assert len(input.size()) == 4  # 确保输入数据是四维的 (N, C, H, W)
        input = input.cuda()  # 将输入数据移动到 GPU
        assert input.is_cuda  # 确保输入数据在 GPU 上
        input_height = input.size()[-2]  # 获取输入数据的高度
        input_width = input.size()[-1]  # 获取输入数据的宽度
        matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1 = self.get_matrix(input_height, input_width)  # 生成低频和高频滤波矩阵
        if mode == 'full':
            return DWTFunction_2D.apply(input, matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1)
        elif mode == 'low':
            return DWTFunction_2D_low.apply(input, matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1)
        elif mode == 'high':
            return DWTFunction_2D_high.apply(input, matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1)


class IDWT_2D(nn.Module):
    def __init__(self, wavename):
        super(IDWT_2D, self).__init__()
        wavelet = _as_wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self, input_height, input_width):
        L1 = np.max((input_height, input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L,      L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            -self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            input_height / 2)), 0:(input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            input_width / 2)), 0:(input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(input_height - math.floor(
            input_height / 2)), 0:(input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(input_width - math.floor(
            input_width / 2)), 0:(input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            matrix_low_0 = torch.Tensor(matrix_h_0)
            matrix_low_1 = torch.Tensor(matrix_h_1)
            matrix_high_0 = torch.Tensor(matrix_g_0)
            matrix_high_1 = torch.Tensor(matrix_g_1)
        return matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1
    
    def forward(self, LL, LH, HL, HH):
        assert len(LL.size()) == len(LH.size()) == len(
            HL.size()) == len(HH.size()) == 4
        input_height = LL.size()[-2] + HH.size()[-2]
        input_width = LL.size()[-1] + HH.size()[-1]
        matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1 = self.get_matrix(input_height, input_width)
        return IDWTFunction_2D.apply(LL, LH, HL, HH, matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1)


class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1,
                              matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH

    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()),
                           torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()),
                           torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(
            matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None
    

class DWTFunction_2D_low(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1,
                              matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        return LL

    @staticmethod
    def backward(ctx, grad_LL):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.matmul(grad_LL, matrix_Low_1.t())
        grad_input = torch.matmul(matrix_Low_0.t(), grad_L)
        return grad_input, None, None, None, None
    

class DWTFunction_2D_high(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1,
                              matrix_High_0, matrix_High_1)
        H = torch.matmul(matrix_High_0, input)
        HH = torch.matmul(H, matrix_High_1)
        return HH

    @staticmethod
    def backward(ctx, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_H = torch.matmul(grad_HH, matrix_High_1.t())
        grad_input = torch.matmul(matrix_High_0.t(), grad_H)
        return grad_input, None, None, None, None
    

class IDWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input_LL, input_LH, input_HL, input_HH,
                matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1,
                              matrix_High_0, matrix_High_1)
        L = torch.add(torch.matmul(input_LL, matrix_Low_1.t()),
                      torch.matmul(input_LH, matrix_High_1.t()))
        H = torch.add(torch.matmul(input_HL, matrix_Low_1.t()),
                      torch.matmul(input_HH, matrix_High_1.t()))
        output = torch.add(torch.matmul(matrix_Low_0.t(), L),
                           torch.matmul(matrix_High_0.t(), H))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1)
        grad_LH = torch.matmul(grad_L, matrix_High_1)
        grad_HL = torch.matmul(grad_H, matrix_Low_1)
        grad_HH = torch.matmul(grad_H, matrix_High_1)
        return grad_LL, grad_LH, grad_HL, grad_HH, None, None, None, None
    
def _as_wavelet(wavelet):
    """Ensure the input argument to be a pywt wavelet compatible object.

    Args:
        wavelet (Wavelet or str): The input argument, which is either a
            pywt wavelet compatible object or a valid pywt wavelet name string.

    Returns:
        Wavelet: the input wavelet object or the pywt wavelet object described by the
            input str.
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet
    

import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps


def draw_feature_map(features,save_dir = 'work_dir/rgbtdroneperson/vis_results/wavelet',name = None, iter=0):
    i=0
    file_name_list = [
        {"id": 0, "file_name": "05120.jpg", "height": 512, "width": 640}, 
        {"id": 1, "file_name": "05563.jpg", "height": 512, "width": 640}, 
        {"id": 2, "file_name": "05569.jpg", "height": 512, "width": 640}, 
        {"id": 3, "file_name": "05650.jpg", "height": 512, "width": 640}, 
        {"id": 4, "file_name": "05661.jpg", "height": 512, "width": 640}, 
        {"id": 5, "file_name": "06006.jpg", "height": 512, "width": 640}
    ]
    gt_name = file_name_list[iter]['file_name']
    img = cv2.imread(f'/data3/pengpeiran/datasets/RGBTDronePerson/val/visible/{gt_name[:-4]}.jpg', cv2.IMREAD_COLOR)
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) 
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+f'_{gt_name[:-4]}.png'), superimposed_img)

                superimposed_img = heatmap * 0.5 + img*0.3
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
                cv2.imwrite(os.path.join(save_dir,name +str(i)+f'overlay_{gt_name[:-4]}.png'), superimposed_img)
                i=i+1

    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray')
                # plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_im