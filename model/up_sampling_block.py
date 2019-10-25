import torch.nn as nn

from model.common import get_norm

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF


class BasicBlockBase(nn.Module):
  expansion = 1
  NORM_TYPE = 'BN'

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               bn_momentum=0.1,
               D=3):
    super(BasicBlockBase, self).__init__()

    self.conv1 = ME.MinkowskiConvolution(
        inplanes, planes, kernel_size=3, stride=stride, dimension=D)
    self.norm1 = get_norm(self.NOR  M_TYPE, planes, bn_momentum=bn_momentum, D=D)
    self.conv2 = ME.MinkowskiConvolution(
        planes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=dilation,
        has_bias=False,
        dimension=D)
    self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
    self.downsample = downsample

  def forward(self, x, y):
    residual1 = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = MEF.relu(out)
    out += residual1

    out = concate(out,y)
    
    residual2 = out
    
    out = self.conv2(out)
    out = self.norm2(out)
    out = MEF.relu(out)
    
    residual2 = self.conv3(residual2)
    residual2 = self.norm3(residual2)
    residual2 = MEF.relu(residual2)

    out += residual2
    out = self.conv_tr1(out)

    return out


class BasicBlockBN(BasicBlockBase):
  NORM_TYPE = 'BN'


class BasicBlockIN(BasicBlockBase):
  NORM_TYPE = 'IN'


def get_block(norm_type,
              inplanes,
              planes,
              stride=1,
              dilation=1,
              downsample=None,
              bn_momentum=0.1,
              D=3):
  if norm_type == 'BN':
    return BasicBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  elif norm_type == 'IN':
    return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  else:
    raise ValueError(f'Type {norm_type}, not defined')
