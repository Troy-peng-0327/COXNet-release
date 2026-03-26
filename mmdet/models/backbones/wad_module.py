import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import pywt
import imageio
import math
from torch.nn import Module
from torch.autograd import Function

def show_tensor(a: torch.Tensor, fig_num=None, title=None):
    """Display a 2D tensor.
    args:
        fig_num: Figure number.
        title: Title of figure.
    """
    a_np = a.squeeze().cpu().clone().detach().numpy()
    # a_np = a
    if a_np.ndim == 3:
        a_np = np.transpose(a_np, (1, 2, 0))
    plt.figure(fig_num)
    plt.tight_layout()
    # plt.tight_layout(pad=0)  # 也可以设为默认的，但这样图大点
    plt.cla()
    plt.imshow(a_np)
    plt.axis('off')
    plt.axis('equal')
    # plt.colorbar()  # 创建颜色条
    if title is not None:
        plt.title(title)
    plt.draw()
    # plt.show()  # imshow是对图像的处理，show是展示图片
    plt.pause(0.1)

class wad_module(nn.Module):
    '''
    This module is used in directly connected networks.
    X --> output
    Args:
        wavename: Wavelet family
    '''
    def __init__(self, wavename='haar'):  # wavelist() or [‘haar’, ‘db’, ‘sym’, ‘coif’, ‘bior’, ‘rbio’, ‘dmey’]
        super(wad_module, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.softmax = nn.Softmax2d()

    @staticmethod
    def get_module_name():
        return "wad"

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)

        # lam = 1  # soft_threshold
        # LH=softthresholding(LH,lam)
        # HL=softthresholding(HL,lam)
        # _=softthresholding(_,lam)


        # for ll, lh, hl, hh in zip(LL, LH, HL, _):  # 可视化 x,  xyl 20221004
        #     ll = ll.sum(dim=0)
        #     plt.subplot(2, 2, 1)
        #     show_tensor(ll, 1, 'LL')
        #     lh = lh.sum(dim=0)
        #     plt.subplot(2, 2, 2)
        #     show_tensor(lh, 1, 'LH')
        #     hl = hl.sum(dim=0)
        #     plt.subplot(2, 2, 3)
        #     show_tensor(hl, 1, 'HL')
        #     hh = hh.sum(dim=0)
        #     plt.subplot(2, 2, 4)
        #     show_tensor(hh, 1, 'HH')

        output = LL
        # output = LL.sub(_)  # xyl 20221013 想减去噪声信息 HH

        x_high = self.softmax(torch.add(LH, HL))
        # x_high = self.softmax(torch.add(LH, HL).sub(_))  # xyl 20221013 想减去噪声信息 HH
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        # for k in x_high:  # 可视化 xyl 20221011
        #     show_tensor(k.sum(dim=0),3, 'x_high')
        # for k in AttMap:
        #     show_tensor(k.sum(dim=0),2, 'AttMap')
        # for k in output:
        #     show_tensor(k.sum(dim=0),1, 'haar')
        return output

class wad_module2(nn.Module):
    '''
    This module is used in directly connected networks.
    X --> output
    Args:
        wavename: Wavelet family
    '''
    def __init__(self, wavename='haar'):  # wavelist() or [‘haar’, ‘db’, ‘sym’, ‘coif’, ‘bior’, ‘rbio’, ‘dmey’]
        super(wad_module2, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.softmax = nn.Softmax2d()

    @staticmethod
    def get_module_name():
        return "wad"

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)

        # for ll, lh, hl, hh in zip(LL, LH, HL, _):  # 可视化 x,  xyl 20221004
        #     ll = ll.sum(dim=0)
        #     plt.subplot(2, 2, 1)
        #     show_tensor(ll, 1, 'LL')
        #     lh = lh.sum(dim=0)
        #     plt.subplot(2, 2, 2)
        #     show_tensor(lh, 1, 'LH')
        #     hl = hl.sum(dim=0)
        #     plt.subplot(2, 2, 3)
        #     show_tensor(hl, 1, 'HL')
        #     hh = hh.sum(dim=0)
        #     plt.subplot(2, 2, 4)
        #     show_tensor(hh, 1, 'HH')


        output = LL

        x_high = self.softmax(torch.add(LH, HL))
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        # for k in x_high:  # 可视化 xyl 20221011
        #     show_tensor(k.sum(dim=0))
        # for k in AttMap:
        #     show_tensor(k.sum(dim=0))
        for k in output:
            show_tensor(k.sum(dim=0), 2, 'haar')
        return output

class wad_module3(nn.Module):
    '''
    This module is used in directly connected networks.
    X --> output
    Args:
        wavename: Wavelet family
    '''
    def __init__(self, wavename='bior2.2'):  # wavelist() or [‘haar’, ‘db’, ‘sym’, ‘coif’, ‘bior’, ‘rbio’, ‘dmey’]
        super(wad_module3, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.softmax = nn.Softmax2d()

    @staticmethod
    def get_module_name():
        return "wad"

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)

        # for ll, lh, hl, hh in zip(LL, LH, HL, _):  # 可视化 x,  xyl 20221004
        #     ll = ll.sum(dim=0)
        #     plt.subplot(2, 2, 1)
        #     show_tensor(ll, 1, 'LL')
        #     lh = lh.sum(dim=0)
        #     plt.subplot(2, 2, 2)
        #     show_tensor(lh, 1, 'LH')
        #     hl = hl.sum(dim=0)
        #     plt.subplot(2, 2, 3)
        #     show_tensor(hl, 1, 'HL')
        #     hh = hh.sum(dim=0)
        #     plt.subplot(2, 2, 4)
        #     show_tensor(hh, 1, 'HH')

        output = LL

        x_high = self.softmax(torch.add(LH, HL))
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        # for k in x_high:  # 可视化 xyl 20221011
        #     show_tensor(k.sum(dim=0))
        # for k in AttMap:
        #     show_tensor(k.sum(dim=0))
        for k in output:
            show_tensor(k.sum(dim=0),3, 'bior2.2')
        return output

# Soft Thresholding 太慢了
def softthresholding(b, lam):
    m = nn.Softshrink(lambd=lam)  # 还有 Hardshrink
    soft_thresh = m(b)
    return soft_thresh.cuda()


"""
实现离散小波变换操作，可用于一维、二维数据。
REF: Wavelet Integrated CNNs for Noise-Robust Image Classification
"""


class DWT_1D(Module):
    """
    input: the 1D data to be decomposed -- (N, C, Length)
    output: lfc -- (N, C, Length/2)
            hfc -- (N, C, Length/2)
    """

    def __init__(self, wavename):
        """
        1D discrete wavelet transform (DWT) for sequence decomposition
        用于序列分解的一维离散小波变换 DWT
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L,      L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            -self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:, (self.band_length_half-1):end]
        matrix_g = matrix_g[:, (self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.Tensor(matrix_h).cuda()
            self.matrix_high = torch.Tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.Tensor(matrix_h)
            self.matrix_high = torch.Tensor(matrix_g)

    def forward(self, input):
        """
        input_low_frequency_component = \mathcal{L} * input
        input_high_frequency_component = \mathcal{H} * input
        :param input: the data to be decomposed
        :return: the low-frequency and high-frequency components of the input data
        """
        assert len(input.size()) == 3
        self.input_height = input.size()[-1]
        self.get_matrix()
        return DWTFunction_1D.apply(input, self.matrix_low, self.matrix_high)


class IDWT_1D(Module):
    """
    input:  lfc -- (N, C, Length/2)
            hfc -- (N, C, Length/2)
    output: the original data -- (N, C, Length)
    """

    def __init__(self, wavename):
        """
        1D inverse DWT (IDWT) for sequence reconstruction
        用于序列重构的一维离散小波逆变换 IDWT
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        generating the matrices: \mathcal{L}, \mathcal{H}
        生成变换矩阵
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L,      L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            -self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:, (self.band_length_half-1):end]
        matrix_g = matrix_g[:, (self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.Tensor(matrix_h).cuda()
            self.matrix_high = torch.Tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.Tensor(matrix_h)
            self.matrix_high = torch.Tensor(matrix_g)

    def forward(self, L, H):
        """
        :param L: the low-frequency component of the original data
        :param H: the high-frequency component of the original data
        :return: the original data
        """
        assert len(L.size()) == len(H.size()) == 3
        self.input_height = L.size()[-1] + H.size()[-1]
        self.get_matrix()
        return IDWTFunction_1D.apply(L, H, self.matrix_low, self.matrix_high)


class DWT_2D_tiny(Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              #hfc_lh: (N, C, H/2, W/2)
              #hfc_hl: (N, C, H/2, W/2)
              #hfc_hh: (N, C, H/2, W/2)
    DWT_2D_tiny only outputs the low-frequency component, which is used in WaveCNet;
    the all four components could be get using DWT_2D, which is used in WaveUNet.
    """

    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D_tiny, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L,      L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            -self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        #input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        #input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        #input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency component of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D_tiny.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class DWT_2D(Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              hfc_lh: (N, C, H/2, W/2)
              hfc_hl: (N, C, H/2, W/2)
              hfc_hh: (N, C, H/2, W/2)
    """

    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        # matrix_h = np.zeros((L + 1,      L1 + self.band_length - 2))                                   # xyl 20221011 代替最大池化
        matrix_h = np.zeros((L,      L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            -self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        # matrix_h_0 = matrix_h[0:(math.floor(
        #     self.input_height / 2 + 1)), 0:(self.input_height + self.band_length - 2)]                 # xyl 20221011 代替最大池化
        # matrix_h_1 = matrix_h[0:(math.floor(
        #     self.input_width / 2 + 1)), 0:(self.input_width + self.band_length - 2)]                   # xyl 20221011 代替最大池化
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
        # for i in range(L1 - L - 1):                                           # xyl 20221008 针对输入是奇数的情况
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        # matrix_g_0 = matrix_g[0:(math.floor(                                   # xyl 20221008 针对输入是奇数的情况
        #     self.input_height / 2 + 1)), 0:(self.input_height + self.band_length - 2)]                   # xyl 20221011 代替最大池化
        # matrix_g_1 = matrix_g[0:(math.floor(
        #     self.input_width / 2 + 1)), 0:(self.input_width + self.band_length - 2)]                   # xyl 20221011 代替最大池化

        matrix_h_0 = matrix_h_0[:, (self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency and high-frequency components of the input 2D data
        """
        assert len(input.size()) == 4
        self.input = input.cuda()
        assert self.input.is_cuda
        self.input_height = self.input.size()[-2]
        self.input_width = self.input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(self.input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class IDWT_2D(Module):
    """
    input:  lfc -- (N, C, H/2, W/2)
            hfc_lh -- (N, C, H/2, W/2)
            hfc_hl -- (N, C, H/2, W/2)
            hfc_hh -- (N, C, H/2, W/2)
    output: the original 2D data -- (N, C, H, W)
    """

    def __init__(self, wavename):
        """
        2D inverse DWT (IDWT) for 2D image reconstruction
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L,      L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            -self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, LL, LH, HL, HH):
        """
        recontructing the original 2D data
        the original 2D data = \mathcal{L}^T * lfc * \mathcal{L}
                             + \mathcal{H}^T * hfc_lh * \mathcal{L}
                             + \mathcal{L}^T * hfc_hl * \mathcal{H}
                             + \mathcal{H}^T * hfc_hh * \mathcal{H}
        :param LL: the low-frequency component
        :param LH: the high-frequency component, hfc_lh
        :param HL: the high-frequency component, hfc_hl
        :param HH: the high-frequency component, hfc_hh
        :return: the original 2D data
        """
        assert len(LL.size()) == len(LH.size()) == len(
            HL.size()) == len(HH.size()) == 4
        self.input_height = LL.size()[-2] + HH.size()[-2]
        self.input_width = LL.size()[-1] + HH.size()[-1]
        self.get_matrix()
        return IDWTFunction_2D.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


"""
实现离散小波变换函数，可用于一维、二维数据。
REF: Wavelet Integrated CNNs for Noise-Robust Image Classification
"""


class DWTFunction_1D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low, matrix_High):
        ctx.save_for_backward(matrix_Low, matrix_High)
        L = torch.matmul(input, matrix_Low.t())
        H = torch.matmul(input, matrix_High.t())
        return L, H

    @staticmethod
    def backward(ctx, grad_L, grad_H):
        matrix_L, matrix_H = ctx.saved_variables
        grad_input = torch.add(torch.matmul(
            grad_L, matrix_L), torch.matmul(grad_H, matrix_H))
        return grad_input, None, None


class IDWTFunction_1D(Function):
    @staticmethod
    def forward(ctx, input_L, input_H, matrix_L, matrix_H):
        ctx.save_for_backward(matrix_L, matrix_H)
        output = torch.add(torch.matmul(input_L, matrix_L),
                           torch.matmul(input_H, matrix_H))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        matrix_L, matrix_H = ctx.saved_variables
        grad_L = torch.matmul(grad_output, matrix_L.t())
        grad_H = torch.matmul(grad_output, matrix_H.t())
        return grad_L, grad_H, None, None


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


class DWTFunction_2D_tiny(Function):
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


if __name__ == '__main__':
    net = wad_module()
    # net.eval()

    # 会报错： RuntimeError: "addmm_cuda" not implemented for 'Byte'
    img = imageio.imread(r'mmdet/models/backbones/dog.jpg')  # array (255,255,3)
    img1 = np.transpose(img, (2, 0, 1))  # array (3, 255, 255)
    img2 = torch.FloatTensor(img1)  # tensor (3, 255, 255)
    img3 = img2.unsqueeze(dim=0)  # tensor (1, 3, 255, 255)
    from torch.nn import functional as F
    img4 = F.interpolate(img3, size=(256, 320), mode='bilinear', align_corners=False)
    out = net(img4)

    # var = torch.randn(1, 256, 31, 31)  # pysot 下采样之前的维度
    # # var = pywt.data.camera().astype(np.float32)
    # out = net(var)

    print(out[0].shape)