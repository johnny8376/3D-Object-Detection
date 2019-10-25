# -*- coding: future_fstrings -*-
import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from model.common import get_norm

from model.residual_block import get_block


class ResUNet2(ME.MinkowskiNetwork):
  NORM_TYPE = None
  BLOCK_NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
# CHANNELS = [None, 16, 32, 64, 64]
  TR_CHANNELS = [None, 32, 64, 64, 128]

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=None,
               conv1_kernel_size=None,
               D=3):
    ME.MinkowskiNetwork.__init__(self, D)
    NORM_TYPE = self.NORM_TYPE
    BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    self.normalize_feature = normalize_feature
    self.conv1 = ME.MinkowskiConvolution(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.block1 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.conv2 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1],
        out_channels=CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv3 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[2],
        out_channels=CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv4 = ME.MinkowskiConvolution(
        in_channels=CHANNELS[3],
        out_channels=CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4 = get_block(
        BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv4_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.block4_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.block3_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = ME.MinkowskiConvolutionTranspose(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=D)
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.block2_tr = get_block(
        BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = ME.MinkowskiConvolution(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=D)

    # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = ME.MinkowskiConvolution(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        has_bias=True,
        dimension=D)

  def forward(self,x):
    out, out_s1 = self.Diversion1(x)
    out, out_s2 = self.Diversion2(out)
    out, out_s3 = self.Diversion3(out)
    out_s4, out = self.Diversion4(out)
    ##
    out = self.junction4( _ , out)
    out = self.junction3(out, out_s3)
    out = self.junction2(out, out_s2)
    out = self.junction1(out, out_s1)
    ## RPN
    rois= self.RPN(out_s4)
#    rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        #[[idx,x1,x2,...,x8],...]
        # do roi pooling based on predicted rois
        pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1,8))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)

        return rois, cls_prob, bbox_pred, rois_label


class ResUNetBN2C(ResUNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 128]

