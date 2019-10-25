import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RPN(nn.Module):
    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1.2, 2.7],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):

    super(PRN,self).__init__()
    self.anchor = anchors()
    self.feat_stride = feat_stride

    self.localization = sp.conv
    self.classification = sp.conv

    def forward(self, x, imag_size, scale=1):
        """
        Args:
          x(sparseTensor):  The feature map
          img_size = (height width, length)
          scale: the amount of scale down of input image (2*2*2=8)
        Returns:
          (SparseTensor,)
        """
        n,_,h,w,l = x.shape
        
        loc_cls = self.compression(x)
        p, n = self.balancer(loc_cls[:loc]) #AnchorTargetCreator
        rois = self.roi_sampler(loc_cls)
        return rois, 1self.loss(p,n)
