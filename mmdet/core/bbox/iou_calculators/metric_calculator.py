import math
import torch

from .builder import IOU_CALCULATORS


@IOU_CALCULATORS.register_module()
class BboxDistanceMetric(object):
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='wasserstein', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


def bbox_overlaps(bboxes1, bboxes2, mode='siwd', is_aligned=False, eps=1e-6, constant=12.8 , weight=2, img_shape=(512, 640), scales=[1, 2, 4], scale_weights=[1, 0.5, 0.25]):
    assert mode in ['iou', 'iof', 'giou', 'normalized_giou', 'ciou', 'diou', 'nwd', 'siwd', 'simd', 'simd_giou', 'harmonic_giou_simd',
                    'dotd','box1_box2','focaliou2', 'focaliou3', 'amswd', 'amswdv2', 'wdgiou', 'wgiou', 'simdw',
                    'swdv2', 'center_distance', 'center_distance2', 'kl', 'gwd', 'bcd', 'gjsd'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)

    if rows * cols == 0:
        return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if mode in ['box1_box2']:
        box1_box2 = area1[...,None] / area2[None,...]
        return box1_box2

    lt = torch.max(bboxes1[..., :, None, :2],
                    bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
    rb = torch.min(bboxes1[..., :, None, 2:],
                    bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

    wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
    overlap = wh[..., 0] * wh[..., 1]

    union = area1[..., None] + area2[..., None, :] - overlap + eps

    if mode in ['giou', 'normalized_giou', 'ciou', 'diou', 'simd', 'simd_giou', 'harmonic_giou_simd']:
        enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                bboxes2[..., None, :, 2:])
        

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    
    if mode in ['iou', 'iof']:
        return ious

    if mode in ['focaliou2']:
        area1=area1.float()
        focalious = ious.pow(torch.sqrt(torch.sqrt(area1[..., None]) / 12.8))
        return focalious

    if mode in ['focaliou3']:
        vt_ind= (area1 < 64)
        t_ind = (area1 >= 64) & (area1 < 16*16)
        s_ind = (area1 >= 16*16) & (area1 < 32*32)
        m_ind = (area1 >= 32*32)
        focalious = ious *(vt_ind[...,None]*1.3 + t_ind[...,None]*1.3 + s_ind[...,None]*0.9+ m_ind[...,None]*0.8)
        return focalious
    # calculate gious
    if mode in ['giou', 'normalized_giou', 'ciou', 'diou', 'simd_giou', 'harmonic_giou_simd']:
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps)
        gious = ious - (enclose_area - union) / enclose_area

    if mode in ['simd', 'simd_giou', 'simd_giou', 'harmonic_giou_simd']:
        # Calculate sim_location
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        sim_location = torch.exp(-torch.norm(center1 - center2, dim=-1))

        # Calculate sim_shape
        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]  + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]  + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]  + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]  + eps
        # import pdb
        # pdb.set_trace()
        sim_shape = torch.exp(-torch.abs(w1 - w2) / (w1 + w2 + eps)) * \
                    torch.exp(-torch.abs(h1 - h2) / (h1 + h2 + eps))

        # Combine to calculate SimD
        simd = torch.exp(-sim_location - sim_shape)
        
        if mode == 'simd':
            return simd
        
    if mode == 'simd_giou':
        combined_metric = gious + simd
        return combined_metric
    
    if mode == 'harmonic_giou_simd':
        combined_metric = 2 * (gious * simd) / (gious + simd + eps)
        return combined_metric

    if mode == 'giou':
        return gious

    if mode == 'normalized_giou':
        gious = (1 + gious) / 2
        
        return gious

    if mode == 'diou':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #distances of center points between gt and pre

        enclosed_diagonal_distances = enclose_wh[..., 0] * enclose_wh[..., 0] + enclose_wh[..., 1] * enclose_wh[..., 1] # distances of diagonal of enclosed bbox
        
        dious = ious - center_distance / torch.max(enclosed_diagonal_distances, eps)
        
        dious = torch.clamp(dious,min=-1.0,max = 1.0)
        
        return dious

    if mode == 'ciou':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps # distances of center points between gt and pre

        enclosed_diagonal_distances = enclose_wh[..., 0] * enclose_wh[..., 0] + enclose_wh[..., 1] * enclose_wh[..., 1] # distances of diagonal of enclosed bbox

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]  + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]  + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]  + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]  + eps

        factor = 4 / math.pi ** 2
        v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        cious = ious - (center_distance / torch.max(enclosed_diagonal_distances, eps) + v ** 2 / torch.max(1 - ious + v, eps))

        cious = torch.clamp(cious, min=-1.0, max=1.0)
        
        return cious
    
    if mode == 'siwd':
        H = img_shape[0]
        W = img_shape[1]

        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]
        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]

        center_distance = 4*whs[..., 0] * whs[..., 0] / (w1 + w2)**2 + 4*whs[..., 1] * whs[..., 1] / (h1 + h2)**2
        wh_distance = (w1 - w2) ** 2 / (w1 + w2)**2 + (h1 - h2) ** 2 / (h1 + h2)**2
        wassersteins = torch.sqrt(center_distance + wh_distance)
        
        normalized_wasserstein = torch.exp(-wassersteins*2)
        return normalized_wasserstein

    if mode == 'simdw':
        # Calculate sim_location
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        # Calculate sim_shape
        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        # Add eps for numerical stability
        center_distance = (whs[..., 0] ** 2 / (w1 + w2 + eps) ** 2) + (whs[..., 1] ** 2 / (h1 + h2 + eps) ** 2)
        sim_location = torch.exp(-center_distance)

        sim_shape = torch.exp(-((w1 - w2) ** 2 / (w1 + w2 + eps) ** 2 + (h1 - h2) ** 2 / (h1 + h2 + eps) ** 2))

        # Combine to calculate SimDW
        simdw = sim_location * sim_shape

        return simdw


    if mode == 'nwd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]  + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1]  + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0]  + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1]  + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / (weight**2)

        wassersteins = torch.sqrt(center_distance + wh_distance)
        # constant = 11.7
        normalized_wasserstein = torch.exp(-wassersteins/constant)

        return normalized_wasserstein



    if mode == 'dotd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #

        distance = torch.sqrt(center_distance)

        dotd = torch.exp(-distance/constant)

        return dotd

    if mode == 'center_distance':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance2 = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #

        distance = torch.sqrt(center_distance2)
    
        return distance

    if mode == 'center_distance2':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance2 = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps #

        return center_distance2


    '''
    if mode == 'swd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

        wassersteins = torch.sqrt(center_distance + wh_distance)

        factor = torch.sqrt(area1[...,None]*area2[...,None,:])**(1/3)

        swd = torch.exp(-wassersteins /factor)

        return swd
    '''
    if mode == 'swdv2':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

        wassersteins = torch.sqrt(center_distance + wh_distance)

        factor = ((w1+w2)+(h1+h2))/4
        #swd = np.exp(-wassersteins / factor)
        swd = torch.exp(- wassersteins**0.5/ factor**0.5)

        return swd


    if mode == 'kl':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        kl=(w2**2/w1**2+h2**2/h1**2+4*whs[..., 0]**2/w1**2+4*whs[..., 1]**2/h1**2+torch.log(w1**2/w2**2)+torch.log(h1**2/h2**2)-2)/2

        kld = 1/(1+kl)

        return kld

    if mode == 'gwd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps  #

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
        wasserstein = center_distance + wh_distance

        gwd = 1/(1+wasserstein)

        return gwd

    if mode == 'bcd':
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        x1 = center1[..., 0]
        x2 = center2[..., 0]
        y1 = center1[..., 1]
        y2 = center2[..., 1]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        a1 = w1**2/4
        a2 = w2**2/4
        b1 = h1**2/4
        b2 = h2**2/4
 
        bcd = ((x1-x2)**2/(a1+a2)+(y1-y2)**2/(a1+a2))/4 + torch.log((a1+a2)*(b1+b2))/2 - torch.log(a1*a2*b1*b2)/4

        bc = torch.exp(-bcd)

        hd = torch.sqrt(1-bc)

        bcd1 = 1-hd

        return bcd1

    if mode == 'gjsd':
        alpha = 1/2
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        x1 = center1[..., 0]
        x2 = center2[..., 0]
        y1 = center1[..., 1]
        y2 = center2[..., 1]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        gjsd = 0.5 * (4*(1-alpha)*x1**2/(w1**2)+4*(1-alpha)*y1**2/(h1**2)+4*alpha*(x2**2)/(w2**2)+4*alpha*y2**2/(h2**2)-4*(((1-alpha)*x1/(w1**2)+alpha*x2/(w2**2))**2/((1-alpha)/(w1**2)+alpha/(w2**2)) + ((1-alpha)*y1/(h1**2)+alpha*y2/(h2**2))**2/((1-alpha)/(h1**2)+alpha/(h2**2))) + torch.log(16*((w1*h1)**2/16)**(1-alpha)*((w2*h2)**2/16)**(alpha)*((1-alpha)/w1**2+alpha/w2**2)*((1-alpha)/h1**2+alpha/h2**2)))

        gjsd = 1/(1+gjsd)

        return gjsd
    
    if mode == 'amswd':
        H, W = img_shape
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        S = (w1 + w2) / 2
        center_distance = sum([((sc * whs[..., 0]) ** 2 / (sc * w1 + sc * w2) ** 2 + 
                                (sc * whs[..., 1]) ** 2 / (sc * h1 + sc * h2) ** 2) for sc in scales])

        wh_distance = sum([((sc * w1 - sc * w2) ** 2 / (sc * w1 + sc * w2) ** 2 + 
                            (sc * h1 - sc * h2) ** 2 / (sc * h1 + sc * h2) ** 2) for sc in scales])

        wassersteins = torch.sqrt(center_distance + wh_distance)
        normalized_wasserstein = torch.exp(-wassersteins * 2)

        return normalized_wasserstein

    if mode == 'amswdv2':
        H, W = img_shape
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        center_distance = sum([weight * ((sc * whs[..., 0]) ** 2 / (sc * w1 + sc * w2) ** 2 + 
                                         (sc * whs[..., 1]) ** 2 / (sc * h1 + sc * h2) ** 2) 
                               for sc, weight in zip(scales, scale_weights)])

        wh_distance = sum([weight * ((sc * w1 - sc * w2) ** 2 / (sc * w1 + sc * w2) ** 2 + 
                                     (sc * h1 - sc * h2) ** 2 / (sc * h1 + sc * h2) ** 2) 
                           for sc, weight in zip(scales, scale_weights)])

        wassersteins = torch.sqrt(center_distance + wh_distance)
        normalized_wasserstein = torch.exp(-wassersteins * 2)

        return normalized_wasserstein

    if mode == 'wdgiou':
        alpha = 0.5
        H, W = img_shape
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        center_distance = sum([((sc * whs[..., 0]) ** 2 / (sc * w1 + sc * w2) ** 2 + 
                                (sc * whs[..., 1]) ** 2 / (sc * h1 + sc * h2) ** 2) for sc in scales])

        wh_distance = sum([((sc * w1 - sc * w2) ** 2 / (sc * w1 + sc * w2) ** 2 + 
                            (sc * h1 - sc * h2) ** 2 / (sc * h1 + sc * h2) ** 2) for sc in scales])

        wassersteins = torch.sqrt(center_distance + wh_distance)

        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]
        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        union = area1[..., :, None] + area2[..., None, :] - overlap

        enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

        giou_component = (enclose_area - union) / enclose_area.clamp(min=eps)

        # 使用加权方式结合 Wasserstein 距离和 GIoU
        combined_wasserstein = alpha * wassersteins - (1 - alpha) * giou_component
        normalized_combined_wasserstein = torch.exp(-combined_wasserstein * 2)

        return normalized_combined_wasserstein
    
    if mode == 'wgiou':
        H, W = img_shape
        center1 = (bboxes1[..., :, None, :2] + bboxes1[..., :, None, 2:]) / 2
        center2 = (bboxes2[..., None, :, :2] + bboxes2[..., None, :, 2:]) / 2
        whs = center1[..., :2] - center2[..., :2]

        w1 = bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0] + eps
        h1 = bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1] + eps
        w2 = bboxes2[..., None, :, 2] - bboxes2[..., None, :, 0] + eps
        h2 = bboxes2[..., None, :, 3] - bboxes2[..., None, :, 1] + eps

        S = (w1 + w2) / 2
        center_distance = sum([((sc * whs[..., 0]) ** 2 / (sc * w1 + sc * w2) ** 2 + 
                                (sc * whs[..., 1]) ** 2 / (sc * h1 + sc * h2) ** 2) for sc in scales])

        wh_distance = sum([((sc * w1 - sc * w2) ** 2 / (sc * w1 + sc * w2) ** 2 + 
                            (sc * h1 - sc * h2) ** 2 / (sc * h1 + sc * h2) ** 2) for sc in scales])

        wassersteins = torch.sqrt(center_distance + wh_distance)

        # 调整最小外接矩形的尺寸
        adjusted_w1 = w1 * torch.exp(-wassersteins)
        adjusted_h1 = h1 * torch.exp(-wassersteins)
        adjusted_w2 = w2 * torch.exp(-wassersteins)
        adjusted_h2 = h2 * torch.exp(-wassersteins)

        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]
        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        area1 = adjusted_w1 * adjusted_h1
        area2 = adjusted_w2 * adjusted_h2
        union = area1 + area2 - overlap

        enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

        giou_component = (enclose_area - union) / enclose_area.clamp(min=eps)
        combined_giou_wasserstein = giou_component

        return combined_giou_wasserstein