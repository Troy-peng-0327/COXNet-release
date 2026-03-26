import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from mmcv.runner import BaseModule
import pywt
import math
import numpy as np
from torch.autograd import Function
from typing import Sequence, Tuple, Union, List
from einops import rearrange, repeat
from .wavelet_process import DWTC


class COXFusionLayer(nn.Module):
    def __init__(
            self, 
            in_channels,
            reduction=16,
            num_layers=5,
            fs_type='cat',
            use_clfm=[],
            clfm_initialize=False,
            clfm_use_ca=False,
            clfm_use_sa=False,
            use_aam=False,
            use_grid=False,
            aam_kernels=[9, 7, 5, 3, 1],
            aam_dc=[False, False, False, False, False],
            aam_range_factor=[1, 1, 1, 1, 1],
            use_wf=[False, False, False, False, False],
            wf_loss=[],
            wf_initialize=[False, False, False, False, False],
            wf_use_sa=[False, False, False, False, False],
            wf_use_ca=[False, False, False, False, False],
            use_msf=[False, False, False, False, False],
            use_msf_dc=[False, False, False, False, False],
            msf_kernels=[3, 3, 3],
            ):
        super(COXFusionLayer, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.num_layers = num_layers
        self.fs_type = fs_type
        self.use_clfm = use_clfm
        self.wf_loss = wf_loss

        if 'wdown_t' in self.use_clfm or 'wdown_v' in self.use_clfm:
            self.dwt = DWTModule(mode='dwt')
        if 'wup_t' in self.use_clfm or 'wup_v' in self.use_clfm or 'wup_tv' in self.use_clfm or 'wup_vt' in self.use_clfm:
            self.idwt = DWTModule(mode='idwt')
        if 'wup_tv_v2' in self.use_clfm or 'wup_vt_v2' in self.use_clfm:
            self.idwt_layers = nn.ModuleList()
            for i in range(num_layers):
                self.idwt_layers.append(
                    DWTModuleV2(
                        in_channels=in_channels, 
                        mode='upsample', 
                        initialize=clfm_initialize, 
                        use_ca=clfm_use_ca, 
                        use_sa=clfm_use_sa)
                )
        if 'wup_tv_v3' in self.use_clfm or 'wup_vt_v3' in self.use_clfm:
            self.idwt_layers = nn.ModuleList()
            for i in range(num_layers):
                self.idwt_layers.append(
                    DWTC(
                        in_channel=in_channels, 
                        out_channel=in_channels,
                        mode='upsample',))
        if 'wup_tv_v4' in self.use_clfm or 'wup_vt_v4' in self.use_clfm:
            self.idwt_layers = nn.ModuleList()
            for i in range(num_layers):
                self.idwt_layers.append(
                    DWTC(
                        in_channel=in_channels, 
                        out_channel=in_channels,
                        mode='upsample_v4',))
        if 'wup_tv_v5' in self.use_clfm or 'wup_vt_v5' in self.use_clfm:
            self.idwt_layers = nn.ModuleList()
            for i in range(num_layers):
                self.idwt_layers.append(
                    DWTC(
                        in_channel=in_channels, 
                        out_channel=in_channels,
                        mode='upsample_v5',))
        if 'wup_tv_v6' in self.use_clfm or 'wup_vt_v6' in self.use_clfm:
            self.idwt_layers = nn.ModuleList()
            for i in range(num_layers):
                self.idwt_layers.append(
                    DWTC(
                        in_channel=in_channels, 
                        out_channel=in_channels,
                        mode='upsample_v6',))
        if 'wup_tv_v7' in self.use_clfm or 'wup_vt_v7' in self.use_clfm:
            self.idwt_layers = nn.ModuleList()
            for i in range(num_layers):
                self.idwt_layers.append(
                    DWTC(
                        in_channel=in_channels, 
                        out_channel=in_channels,
                        mode='upsample_v7',))
        if 'wup_tv_v8' in self.use_clfm or 'wup_vt_v8' in self.use_clfm:
            self.idwt_layers = nn.ModuleList()
            for i in range(num_layers):
                self.idwt_layers.append(
                    DWTC(
                        in_channel=in_channels, 
                        out_channel=in_channels,
                        mode='upsample_v8',))

        if fs_type == 'cat':
            self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        elif fs_type == 'add':
            assert in_channels % 2 == 0, "in_channels should be divisible by 2 for add fusion type"
        elif fs_type == 'dasr':
            self.dasr_layers = nn.ModuleList()
            for i in range(num_layers):
                self.dasr_layers.append(
                    DASR(
                        in_channels=in_channels,
                        reduction=reduction,
                        use_aam=use_aam,
                        use_grid=use_grid,
                        aam_kernels=aam_kernels[i],
                        aam_dc=aam_dc[i],
                        aam_range_factor=aam_range_factor[i],
                        use_wf=use_wf[i],
                        wf_initialize=wf_initialize[i],
                        wf_use_sa=wf_use_sa[i],
                        wf_use_ca=wf_use_ca[i],
                        use_msf=use_msf[i],
                        use_msf_dc=use_msf_dc[i],
                        msf_kernels=msf_kernels,
                    )
                )
        else:
            raise ValueError(f"Unsupported fusion type: {fs_type}")

    def forward(self, v_feats, t_feats, gt_bboxes=None, img_metas=None):
        fused_feats = []
        for i in range(self.num_layers):
            v_feat = v_feats[i]
            t_feat = t_feats[i]

            if len(self.use_clfm) != 0:
                if 'down_t' in self.use_clfm or 'up_t' in self.use_clfm:
                    t_feat = F.interpolate(t_feat, size=v_feat.shape[2:], mode='bilinear', align_corners=True)
                if 'down_v' in self.use_clfm or 'up_v' in self.use_clfm:
                    v_feat = F.interpolate(v_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=True)
                if 'wdown_t' in self.use_clfm:
                    t_feat = self.dwt(t_feat)
                if 'wdown_v' in self.use_clfm:
                    v_feat = self.dwt(v_feat)
                if 'wup_t' in self.use_clfm:
                    t_feat = self.idwt(t_feat)
                if 'wup_tv' in self.use_clfm:
                    t_feat = self.idwt(t_feat, HH=v_feat)
                if 'wup_v' in self.use_clfm:
                    v_feat = self.idwt(v_feat)
                if 'wup_vt' in self.use_clfm:
                    v_feat = self.idwt(v_feat, HH=t_feat)
                if 'wup_tv_v2' in self.use_clfm:
                    t_feat = self.idwt_layers[i](t_feat, v_feat)
                if 'wup_vt_v2' in self.use_clfm:
                    v_feat = self.idwt_layers[i](v_feat, t_feat)    
                if 'wup_vt_v3' in self.use_clfm:
                    t_feat, v_feat = self.idwt_layers[i](t_feat, v_feat)          
                if 'wup_vt_v4' in self.use_clfm or 'wup_vt_v5' in self.use_clfm or 'wup_vt_v6' in self.use_clfm or 'wup_vt_v7' in self.use_clfm or 'wup_vt_v8' in self.use_clfm:
                    v_feat = self.idwt_layers[i](t_feat, v_feat)          
                if 'wup_tv_v4' in self.use_clfm:
                    t_feat = self.idwt_layers[i](v_feat, t_feat)         

            if self.fs_type == 'cat':
                fused_feat = torch.cat([v_feat, t_feat], dim=1)
                fused_feats.append(self.conv(fused_feat))

            elif self.fs_type == 'add':
                fused_feat = v_feat + t_feat
                fused_feats.append(fused_feat)
            
            elif self.fs_type == 'dasr':
                fused_feat = self.dasr_layers[i](v_feat, t_feat)
                fused_feats.append(fused_feat)

        if self.training:
            if 'v' in self.wf_loss:
                wf_loss = 0
                for i in range(self.num_layers):
                    wf_loss += self.compute_kl_loss_near_objects(fused_feats[i], v_feats[i], gt_bboxes, img_metas)
                return fused_feats, wf_loss
            elif 't' in self.wf_loss:
                wf_loss = 0
                for i in range(self.num_layers):
                    wf_loss += self.compute_kl_loss_near_objects(fused_feats[i], t_feats[i], gt_bboxes, img_metas)
                return fused_feats, wf_loss

        return fused_feats
    
    def get_wavelet_loss(self, mode='wf'):
        wavelet_loss = 0
        if mode == 'clfm':
            for i in range(self.num_layers):
                wavelet_loss += self.idwt_layers[i].get_wavelet_loss()
        if mode == 'wf':
            for i in range(self.num_layers):
                wavelet_loss += self.dasr_layers[i].get_wavelet_loss()
        if mode == 'wf_v2':
            for i in range(self.num_layers):
                wavelet_loss += self.idwt_layers[i].get_wavelet_loss()
        return wavelet_loss
    
    def compute_kl_loss_near_objects(self, fused_x, x, bboxes, img_metas):
        kl_loss = 0
        b, _, h, w = fused_x.shape

        for i in range(b):
            H, W = img_metas[i]['pad_shape'][:2]
            scale_h = h / H
            scale_w = w / W

            _bboxes = bboxes[i].clone()
            for bbox in _bboxes:
                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                
                new_width = (bbox[2] - bbox[0]) * 1.5
                new_height = (bbox[3] - bbox[1]) * 1.5
                
                bbox[0] = max(0, x_center - new_width / 2)
                bbox[2] = min(W, x_center + new_width / 2)
                bbox[1] = max(0, y_center - new_height / 2)
                bbox[3] = min(H, y_center + new_height / 2)

                bbox[[0, 2]] *= scale_w  # x 坐标
                bbox[[1, 3]] *= scale_h  # y 坐标

                # 取整并确保在特征图范围内
                x_min, y_min, x_max, y_max = bbox.int()
                x_min, x_max = x_min.clamp(0, w-1), x_max.clamp(0, w-1)
                y_min, y_max = y_min.clamp(0, h-1), y_max.clamp(0, h-1)

                # 提取目标区域的特征
                fused_x_region = fused_x[i, :, y_min:y_max, x_min:x_max]
                x_region = x[i, :, y_min:y_max, x_min:x_max]

                # 将目标区域特征转换为概率分布
                fused_x_region_log = F.log_softmax(fused_x_region, dim=0)  # 使用 log_softmax 作为预测分布
                x_region_soft = F.softmax(x_region, dim=0)  # 使用 softmax 作为目标分布

                # 计算 KL 散度
                kl_loss += F.kl_div(fused_x_region_log, x_region_soft, reduction='batchmean')
        
        kl_loss /= b
        return kl_loss
    

    

class DASR(BaseModule):
    def __init__(
            self, 
            in_channels,
            reduction=16,
            use_aam=False,
            use_grid=False,
            aam_kernels=3,
            aam_dc=False,
            aam_range_factor=1.0,
            use_wf=False,
            wf_initialize=False,
            wf_use_sa=False,
            wf_use_ca=False,
            use_msf=False,
            use_msf_dc=False,
            msf_kernels=[3, 3, 3],
            ):
        super(DASR, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_aam = use_aam
        self.use_grid = use_grid
        self.aam_kernels = aam_kernels
        self.aam_dc = aam_dc
        self.aam_range_factor = aam_range_factor
        self.use_wf = use_wf
        self.use_msf = use_msf
        self.use_msf_dc = use_msf_dc
        self.msf_kernels = msf_kernels

        if self.use_aam:
            self.proj_offset = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
            if self.use_grid:
                if self.aam_dc and aam_kernels > 3:
                    self.conv_offset = nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels, 
                            out_channels=in_channels, 
                            kernel_size=3, 
                            stride=1, 
                            dilation=aam_kernels // 2,
                            padding=aam_kernels // 2, 
                            groups=in_channels),
                        LayerNormProxy(in_channels),
                        nn.GELU(),
                        nn.Conv2d(
                            in_channels=in_channels, 
                            out_channels=2, 
                            kernel_size=1, 
                            stride=1, 
                            padding=0, 
                            bias=False))
                else:
                    self.conv_offset = nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels, 
                            out_channels=in_channels, 
                            kernel_size=aam_kernels, 
                            stride=1, 
                            padding=aam_kernels // 2, 
                            groups=in_channels),
                        LayerNormProxy(in_channels),
                        nn.GELU(),
                        nn.Conv2d(
                            in_channels=in_channels, 
                            out_channels=2, 
                            kernel_size=1, 
                            stride=1, 
                            padding=0, 
                            bias=False))
            else:
                self.conv_offset = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=in_channels, 
                        kernel_size=aam_kernels, 
                        stride=1, 
                        padding=aam_kernels // 2, 
                        groups=in_channels),
                    LayerNormProxy(in_channels),
                    nn.GELU(),
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=in_channels, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0, 
                        bias=False))
        
        if self.use_wf == True:
            self.wave_fuse = DWTModuleV2(in_channels=in_channels, mode='fusion', initialize=wf_initialize, use_sa=wf_use_sa, use_ca=wf_use_ca)
        elif self.use_wf == 'fusion_v2':
            self.wave_fuse = DWTC(in_channel=in_channels, out_channel=in_channels, mode='fusion_v2')
        elif self.use_wf == 'fusion_v3':
            self.wave_fuse = DWTC(in_channel=in_channels, out_channel=in_channels, mode='fusion_v3')
        else:
            self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

        if self.use_msf:
            self.msf_conv_layers = nn.ModuleList()
            for kernel in msf_kernels:
                if self.use_msf_dc and kernel > 3:
                    msf_conv = nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=in_channels // reduction, 
                        kernel_size=3, 
                        padding=kernel // 2,
                        dilation=kernel // 2)
                else:
                    msf_conv = nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=in_channels // reduction, 
                        kernel_size=kernel, 
                        padding=kernel // 2)
                self.msf_conv_layers.append(msf_conv)
            self.msf_gap = nn.AdaptiveAvgPool2d(1)
            self.relu = nn.ReLU(inplace=True)
            self.msf_fc = nn.Conv2d(
                in_channels=in_channels // reduction, 
                out_channels=in_channels // reduction * 3, 
                kernel_size=1)
            self.proj_msf = nn.Conv2d(
                in_channels=in_channels // reduction * 3, 
                out_channels=in_channels, 
                kernel_size=1)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, feat_v, feat_t):
        if self.use_aam:
            proj_offset = self.proj_offset(torch.cat([feat_v, feat_t], dim=1))
            offset = self.conv_offset(proj_offset)
            Hk, Wk = offset.size(2), offset.size(3)

            if self.use_grid:
                if self.aam_range_factor > 0:
                    offset_range = torch.tensor(
                        [1.0 / Hk, 1.0 / Wk], 
                        device=offset.device).reshape(1, 2, 1, 1)
                    offset = offset.tanh().mul(offset_range).mul(self.aam_range_factor)
            
                offset = einops.rearrange(offset, 'b c h w -> b h w c')

                vis_reference = self._get_ref_points(
                    Hk, Wk, feat_v.shape[0], feat_v.dtype, feat_v.device)
                lwir_reference = self._get_ref_points(
                    Hk, Wk, feat_v.shape[0], feat_v.dtype, feat_v.device)
                
                if self.aam_range_factor >= 0:
                    vis_pos = vis_reference + offset
                    lwir_pos = lwir_reference
                else:
                    vis_pos = (vis_reference + offset).tanh()
                    lwir_pos = lwir_reference.tanh()
                
                feat_v = F.grid_sample(
                    input=feat_v,
                    grid=vis_pos[..., (1, 0)],  
                    mode='bilinear', align_corners=True)
                
                feat_t = F.grid_sample(
                    input=feat_t,
                    grid=lwir_pos[..., (1, 0)],  
                    mode='bilinear', align_corners=True)
            
            else:
                feat_t = feat_t + offset

        if self.use_wf == True:
            feat_c = self.wave_fuse(feat_v, feat_t)
        elif self.use_wf == 'fusion_v2':
            feat_c = self.wave_fuse(feat_v, feat_t)
        elif self.use_wf == 'fusion_v3':
            feat_c = self.wave_fuse(feat_v, feat_t)
        else:
            feat_c = self.conv(torch.cat([feat_v, feat_t], dim=1))

        if self.use_msf:
            u_feats = []
            for msf_conv in self.msf_conv_layers:
                u_feats.append(msf_conv(feat_c))
            u_feat = sum(u_feats)
            wc = self.relu(self.msf_gap(u_feat))
            ws = self.msf_fc(wc)
            z_feats = torch.chunk(ws, len(self.msf_conv_layers), dim=1)
            a_feats = []
            for i in range(len(z_feats)):
                a_feats.append(u_feats[i] * F.softmax(z_feats[i], dim=1))
            a = self.proj_msf(torch.cat(a_feats, dim=1))
            feat_c = feat_c + a

        return feat_c
    
    def get_wavelet_loss(self):
        return self.wave_fuse.get_wavelet_loss()


class PoolingUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.maxpooling = nn.MaxPool2d(2, 2, dilation=1)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, 1)
    
    def forward(self, x):
        x_ = self.maxpooling(x)
        # x_ = self.upsample(x_)
        x_ = F.interpolate(x_, mode='bilinear', size=x.shape[-2:], align_corners=True)
        # import pdb; pdb.set_trace()
        x = self.conv1x1(torch.cat((x, x_), 1))
        return x


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DWTModule(nn.Module):
    def __init__(
            self,
            wavename='haar',
            mode='dwt'):
        super(DWTModule, self).__init__()
        self.mode = mode

        wavelet = pywt.Wavelet(wavename)

        if mode == 'dwt':
            self.dwt = DWT_2D(wavename=wavelet)
        elif mode == 'idwt':
            self.dwt = DWT_2D(wavename=wavelet)
            self.idwt = IDWT_2D(wavename=wavelet)
        self.softmax = nn.Softmax2d()

    def forward(self, input, LH=None, HL=None, HH=None):
        if self.mode == 'dwt':
            LL, LH, HL, HH = self.dwt(input, mode='full')

            # 融合高频特征
            x_high = self.softmax(torch.add(LH, HL))
            attn = torch.mul(LL, x_high)
            out = torch.add(LL, attn)
        if self.mode == 'idwt':
            if LH == HL == HH == None:
                _, LH, HL, HH = self.dwt(input, mode='full')
                LH = F.interpolate(LH, mode='bilinear', size=input.shape[-2:], align_corners=True)
                HL = F.interpolate(HL, mode='bilinear', size=input.shape[-2:], align_corners=True)
                HH = F.interpolate(HH, mode='bilinear', size=input.shape[-2:], align_corners=True)
            
            elif HH != None and HL == LH == None:
                # 输入另一模态的高层特征
                _, LH, HL, HH = self.dwt(HH, mode='full')

            out = self.idwt(input, LH, HL, HH)
        return out
    

class DWTModuleV2(nn.Module):
    def __init__(
            self,
            in_channels,
            wavename='haar',
            mode='fusion',
            initialize=False,
            use_sa=False,
            use_ca=False):
        super(DWTModuleV2, self).__init__()
        self.in_channels = in_channels
        self.mode = mode
        self.wavelet = _as_wavelet(wavename)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wavename, flip=True
        )

        if initialize:
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
            self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
            self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)
            self.dec_lo_t = nn.Parameter(dec_lo, requires_grad=True)
            self.dec_hi_t = nn.Parameter(dec_hi, requires_grad=True)
            if mode == 'fusion_v2':
                self.rec_lo_t = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
                self.rec_hi_t = nn.Parameter(rec_hi.flip(-1), requires_grad=True)
        else:
            self.dec_lo = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
            self.rec_lo = nn.Parameter(torch.rand_like(rec_lo) * 2 - 1, requires_grad=True)
            self.rec_hi = nn.Parameter(torch.rand_like(rec_hi) * 2 - 1, requires_grad=True)
            self.dec_lo_t = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi_t = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
            if mode == 'fusion_v2':
                self.rec_lo_t = nn.Parameter(torch.rand_like(rec_lo) * 2 - 1, requires_grad=True)
                self.rec_hi_t = nn.Parameter(torch.rand_like(rec_hi) * 2 - 1, requires_grad=True)

        if mode == 'fusion':
            self.wavedec = DWT(self.dec_lo, self.dec_hi, wavename=wavename, level=1)
            self.wavedec_t = DWT(self.dec_lo_t, self.dec_hi_t, wavename=wavename, level=1)
            self.waverec = IDWT(self.rec_lo, self.rec_hi, wavename=wavename, level=1)

            self.fusion = nn.Conv2d(in_channels=in_channels*8, out_channels=in_channels*4, kernel_size=3, padding=1)
            # self.fusion = nn.Sequential(
            #     nn.Conv2d(
            #         in_channels=in_channels*8,
            #         out_channels=in_channels*12,
            #         kernel_size=1,
            #     ),
            #     nn.Conv2d(
            #         in_channels=in_channels*12,
            #         out_channels=in_channels*12,
            #         kernel_size=7,
            #         padding=3,
            #         groups=in_channels*12,
            #     ),
            #     nn.GELU(),
            #     nn.Conv2d(
            #         in_channels=in_channels*12,
            #         out_channels=in_channels*4,
            #         kernel_size=1,
            #     )
            # )

        elif mode == 'fusion_v2':
            self.wavedec = DWT(self.dec_lo, self.dec_hi, wavename=wavename, level=1)
            self.waverec = IDWT(self.rec_lo, self.rec_hi, wavename=wavename, level=1)
            self.wavedec_t = DWT(self.dec_lo_t, self.dec_hi_t, wavename=wavename, level=1)
            self.waverec_t = IDWT(self.rec_lo_t, self.rec_hi_t, wavename=wavename, level=1)

            self.fusion = nn.Conv2d(in_channels=in_channels*8, out_channels=in_channels*4, kernel_size=3, padding=1)

        elif mode == 'upsample' or mode == 'upsample_v2':
            self.wavedec = DWT(self.dec_lo, self.dec_hi, wavename=wavename, level=1)
            self.waverec = IDWT(self.rec_lo, self.rec_hi, wavename=wavename, level=1)

        self.use_sa = use_sa
        self.use_ca = use_ca
        if self.use_sa:
            self.sa_h = nn.Sequential(
                nn.PixelShuffle(2),  # 上采样
                nn.Conv2d(in_channels // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)  # c -> 1
            )
            self.sa_v = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)
            )
            # self.sa_norm = LayerNorm2d(dim)
        if self.use_ca:
            self.ca_h = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局池化
                nn.Conv2d(in_channels, in_channels, 1, padding=0, stride=1, groups=1, bias=True),  # conv2d
            )
            self.ca_v = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels, 1, padding=0, stride=1, groups=1, bias=True)
            )
            self.shuffle = ShuffleBlock(2)

    def forward(self, x_v, x_t):
        if self.mode == 'fusion':
            ya_v, (yh_v, yv_v, yd_v) = self.wavedec(x_v)
            ya_t, (yh_t, yv_t, yd_t) = self.wavedec_t(x_t)
            dec_x = torch.cat([ya_v, ya_t, yh_v, yh_t, yv_v, yv_t, yd_v, yd_t], dim=1)
            x = self.fusion(dec_x)
            ya, yh, yv, yd = torch.chunk(x, 4, dim=1)
            y = self.waverec([ya, (yh, yv, yd)], None)
        elif self.mode == 'fusion_v2':
            ya_v, (yh_v, yv_v, yd_v) = self.wavedec(x_v)
            ya_t, (yh_t, yv_t, yd_t) = self.wavedec_t(x_t)
        elif self.mode == 'upsample':
            ya, (yh, yv, yd) = self.wavedec(x_t)
            y = self.waverec([x_v, (yh, yv, yd)], None)
        elif self.mode == 'upsample_v2':
            ya, (yh, yv, yd) = self.wavedec(x_t)
            y = self.waverec([x_v, (yh, yv, yd)], None)
        if self.use_sa:
            sa_yh = self.sa_h(yh)
            sa_yv = self.sa_v(yv)
            y = y * (sa_yv + sa_yh)
        if self.use_ca:
            yh = F.interpolate(yh, scale_factor=2, mode='area')
            yv = F.interpolate(yv, scale_factor=2, mode='area')
            ca_yh = self.ca_h(yh)
            ca_yv = self.ca_v(yv)
            ca = self.shuffle(torch.cat([ca_yv, ca_yh], 1))  # channel shuffle
            ca_1, ca_2 = ca.chunk(2, dim=1)
            ca = ca_1 * ca_2   # gated channel attention
            y = y * ca
        return y
    
    def get_wavelet_loss(self):
        return self.perfect_reconstruction_loss()[0] + self.alias_cancellation_loss()[0]

    def perfect_reconstruction_loss(self):
        """ Strang 107: Assuming alias cancellation holds:
        P(z) = F(z)H(z)
        Product filter P(z) + P(-z) = 2.
        However since alias cancellation is implemented as soft constraint:
        P_0 + P_1 = 2
        Somehow numpy and torch implement convolution differently.
        For some reason the machine learning people call cross-correlation
        convolution.
        https://discuss.pytorch.org/t/numpy-convolve-and-conv1d-in-pytorch/12172
        Therefore for true convolution one element needs to be flipped.
        """
        pad = self.dec_lo.shape[-1] - 1
        p_lo = F.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0),
            self.rec_hi.flip(-1).unsqueeze(0),
            padding=pad)
        pad = self.dec_hi.shape[-1] - 1
        p_hi = F.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0),
            self.rec_hi.flip(-1).unsqueeze(0),
            padding=pad)
        
        p_test = p_lo + p_hi

        two_at_power_zero = torch.zeros(
            p_test.shape, device=p_test.device, dtype=p_test.dtype)
        two_at_power_zero[..., p_test.shape[-1] // 2] = 2

        errs = (p_test - two_at_power_zero) * (p_test - two_at_power_zero)
        return torch.sum(errs), p_test, two_at_power_zero

    def alias_cancellation_loss(self):
        """ Implementation of the ac-loss as described on page 104 of Strang+Nguyen.
            F0(z)H0(-z) + F1(z)H1(-z) = 0 """
        m1 = torch.tensor([-1], device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        length = self.dec_lo.shape[-1]
        mask = torch.tensor([torch.pow(m1, n) for n in range(length)][::-1],
                            device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        # polynomial multiplication is convolution, compute p(z):
        pad = self.dec_lo.shape[-1] - 1
        p_lo = torch.nn.functional.conv1d(
            self.dec_lo.flip(-1).unsqueeze(0) * mask,
            self.rec_lo.flip(-1).unsqueeze(0),
            padding=pad)

        pad = self.dec_hi.shape[-1] - 1
        p_hi = torch.nn.functional.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0) * mask,
            self.rec_hi.flip(-1).unsqueeze(0),
            padding=pad)

        p_test = p_lo + p_hi
        zeros = torch.zeros(p_test.shape, device=p_test.device,
                            dtype=p_test.dtype)
        errs = (p_test - zeros) * (p_test - zeros)
        return torch.sum(errs), p_test, zeros


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
    

class DWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, wavename='haar', level=1, mode="replicate"):
        super(DWT, self).__init__()
        self.wavelet = _as_wavelet(wavename)
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi

        self.level = level
        self.mode = mode

    def forward(self, x):
        b, c, h, w = x.shape
        if self.level is None:
            self.level = pywt.dwtn_max_level([h, w], self.wavelet)
        wavelet_component: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []

        l_component = x
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)
        for _ in range(self.level):
            l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
            res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
            l_component, lh_component, hl_component, hh_component = res.split(1, 2)
            wavelet_component.append((lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)))
        wavelet_component.append(l_component.squeeze(2))
        return wavelet_component[::-1]
    

class IDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi, wavename='haar', level=1, mode="constant"):
        super(IDWT, self).__init__()
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.wavelet = _as_wavelet(wavename)
        self.level = level
        self.mode = mode

    def forward(self, x, weight=None):
        l_component = x[0]
        _, c, _, _ = l_component.shape
        if weight is None:  # soft orthogonal
            idwt_kernel = construct_2d_filt(lo=self.rec_lo, hi=self.rec_hi)
            idwt_kernel = idwt_kernel.repeat(c, 1, 1)
            idwt_kernel = idwt_kernel.unsqueeze(dim=1)
        else:  # hard orthogonal
            idwt_kernel= torch.flip(weight, dims=[-1, -2])

        self.filt_len = idwt_kernel.shape[-1]
        for c_pos, component_lh_hl_hh in enumerate(x[1:]):
            l_component = torch.cat(
                # ll, lh, hl, hl, hh
                [l_component.unsqueeze(2), component_lh_hl_hh[0].unsqueeze(2),
                 component_lh_hl_hh[1].unsqueeze(2), component_lh_hl_hh[2].unsqueeze(2)], 2
            )
            # cat is not work for the strange transpose
            l_component = rearrange(l_component, 'b c f h w -> b (c f) h w')
            l_component = F.conv_transpose2d(l_component, idwt_kernel, stride=2, groups=c)

            # remove the padding
            padl = (2 * self.filt_len - 3) // 2
            padr = (2 * self.filt_len - 3) // 2
            padt = (2 * self.filt_len - 3) // 2
            padb = (2 * self.filt_len - 3) // 2
            if c_pos < len(x) - 2:
                pred_len = l_component.shape[-1] - (padl + padr)
                next_len = x[c_pos + 2][0].shape[-1]
                pred_len2 = l_component.shape[-2] - (padt + padb)
                next_len2 = x[c_pos + 2][0].shape[-2]
                if next_len != pred_len:
                    padr += 1
                    pred_len = l_component.shape[-1] - (padl + padr)
                    assert (
                            next_len == pred_len
                    ), "padding error, please open an issue on github "
                if next_len2 != pred_len2:
                    padb += 1
                    pred_len2 = l_component.shape[-2] - (padt + padb)
                    assert (
                            next_len2 == pred_len2
                    ), "padding error, please open an issue on github "
            if padt > 0:
                l_component = l_component[..., padt:, :]
            if padb > 0:
                l_component = l_component[..., :-padb, :]
            if padl > 0:
                l_component = l_component[..., padl:]
            if padr > 0:
                l_component = l_component[..., :-padr]
        return l_component


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        x = rearrange(x, 'b (g f) h w -> b g f h w', g=self.groups)
        x = rearrange(x, 'b g f h w -> b f g h w')
        x = rearrange(x, 'b f g h w -> b (f g) h w')
        return x

def fwt_pad2(data, wavelet, mode='replicate'):
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))
    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad

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

def get_filter_tensors(
        wavelet,
        flip: bool,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert input wavelet to filter tensors.

    Args:
        wavelet (Wavelet or str): A pywt wavelet compatible object or
                the name of a pywt wavelet.
        flip (bool): If true filters are flipped.
        device (torch.device) : PyTorch target device.
        dtype (torch.dtype): The data type sets the precision of the
               computation. Default: torch.float32.

    Returns:
        tuple: Tuple containing the four filter tensors
        dec_lo, dec_hi, rec_lo, rec_hi

    """
    wavelet = _as_wavelet(wavelet)

    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor

def construct_2d_filt(lo, hi):
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filter = torch.stack([ll, lh, hl, hh], 0)
    return filter

def _outer(a, b):
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul

def _get_pad(data_len, filt_len):
    # we pad half of the total requried padding on each side.
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padr += 1

    return padr, padl
