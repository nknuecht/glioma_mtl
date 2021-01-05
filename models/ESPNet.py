#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
# File Description: This file contains the CNN models and is adapted from ESPNet and Y-Net
# ESPNET: https://arxiv.org/pdf/1803.06815.pdf
# Y-Net: https://arxiv.org/abs/1806.01313
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.cnn_utils import *

# import torch.nn.functional as F
# import Model as Net

from torch.autograd import Variable

from typing import List, NamedTuple, Optional
from torch import Tensor

SegOut = NamedTuple(
    "SegOut",
    [
        ("mask_out", Tensor),  # N x C x H/4 x W/4
        ("enc_out", Tensor),  # N x C x H x W
    ],
)


class ESPNet(nn.Module):
    def __init__(self, classes=4, channels=1):
        super().__init__()
        self.input1 = InputProjectionA(1)
        self.input2 = InputProjectionA(1)

        initial = 16 # feature maps at level 1
        config = [32, 128, 256, 256] # feature maps at level 2 and onwards
        reps = [2, 2, 3]
        self.config = config

        ### ENCODER

        # all dimensions are listed with respect to an input  of size 4 x 128 x 128 x 128
        self.level0 = CBR(channels, initial, 7, 2) # initial x 64 x 64 x64
        self.level1 = nn.ModuleList()
        for i in range(reps[0]):
            if i==0:
                self.level1.append(DilatedParllelResidualBlockB1(initial, config[0]))  # config[0] x 64 x 64 x64
            else:
                self.level1.append(DilatedParllelResidualBlockB1(config[0], config[0]))  # config[0] x 64 x 64 x64

        # downsample the feature maps
        self.level2 = DilatedParllelResidualBlockB1(config[0], config[1], stride=2) # config[1] x 32 x 32 x 32
        self.level_2 = nn.ModuleList()
        for i in range(0, reps[1]):
            self.level_2.append(DilatedParllelResidualBlockB1(config[1], config[1])) # config[1] x 32 x 32 x 32

        # downsample the feature maps
        self.level3_0 = DilatedParllelResidualBlockB1(config[1], config[2], stride=2) # config[2] x 16 x 16 x 16
        self.level_3 = nn.ModuleList()
        for i in range(0, reps[2]):
            self.level_3.append(DilatedParllelResidualBlockB1(config[2], config[2])) # config[2] x 16 x 16 x 16


        ### DECODER

        # upsample the feature maps
        self.up_l3_l2 = UpSampler(config[2], config[1])  # config[1] x 32 x 32 x 32
        # Note the 2 in below line. You need this because you are concatenating feature maps from encoder
        # with upsampled feature maps
        self.merge_l2 = DilatedParllelResidualBlockB1(2 * config[1], config[1]) # config[1] x 32 x 32 x 32
        self.dec_l2 = nn.ModuleList()
        for i in range(0, reps[0]):
            self.dec_l2.append(DilatedParllelResidualBlockB1(config[1], config[1])) # config[1] x 32 x 32 x 32

        self.up_l2_l1 = UpSampler(config[1], config[0])  # config[0] x 64 x 64 x 64
        # Note the 2 in below line. You need this because you are concatenating feature maps from encoder
        # with upsampled feature maps
        self.merge_l1 = DilatedParllelResidualBlockB1(2*config[0], config[0]) # config[0] x 64 x 64 x 64
        self.dec_l1 = nn.ModuleList()
        for i in range(0, reps[0]):
            self.dec_l1.append(DilatedParllelResidualBlockB1(config[0], config[0])) # config[0] x 64 x 64 x 64

        self.dec_l1.append(CBR(config[0], classes, 3, 1)) # classes x 64 x 64 x 64
        # We use ESP block without reduction step because the number  of input feature maps are very small (i.e. 4 in
        # our case)
        self.dec_l1.append(ASPBlock(classes, classes))

        # Using PSP module to learn the representations at different scales
        self.pspModules = nn.ModuleList()
        scales = [0.2, 0.4, 0.6, 0.8]
        for sc in scales:
             self.pspModules.append(PSPDec(classes, classes, sc))

        # Classifier
        self.classifier = self.classifier = nn.Sequential(
             CBR((len(scales) + 1) * classes, classes, 3, 1),
             ASPBlock(classes, classes), # classes x 64 x 64 x 64
             # nn.Upsample(scale_factor=2), # classes x 128 x 128 x 128
             nn.functional.interpolate(scale_factor=2, mode='trilinear', align_corners=True),
             CBR(classes, classes, 7, 1), # classes x 128 x 128 x 128
             C(classes, classes, 1, 1) # classes x 128 x 128 x 128
        )
        #

        for m in self.modules():
             if isinstance(m, nn.Conv3d):
                 n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                 m.weight.data.normal_(0, math.sqrt(2. / n))
             if isinstance(m, nn.ConvTranspose3d):
                 n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                 m.weight.data.normal_(0, math.sqrt(2. / n))
             elif isinstance(m, nn.BatchNorm3d):
                 m.weight.data.fill_(1)
                 m.bias.data.zero_()

    def forward(self, input1, inp_res=(128, 128, 128), inpSt2=False):
        dim0 = input1.size(2)
        dim1 = input1.size(3)
        dim2 = input1.size(4)

        if self.training or inp_res is None:
            # input resolution should be divisible by 8
            inp_res = (math.ceil(dim0 / 8) * 8, math.ceil(dim1 / 8) * 8,
                       math.ceil(dim2 / 8) * 8)
        if inp_res:
            input1 = F.adaptive_avg_pool3d(input1, output_size=inp_res)

        out_l0 = self.level0(input1)

        for i, layer in enumerate(self.level1): #64
            if i == 0:
                out_l1 = layer(out_l0)
            else:
                out_l1 = layer(out_l1)

        out_l2_down = self.level2(out_l1) #32
        for i, layer in enumerate(self.level_2):
            if i == 0:
                out_l2 = layer(out_l2_down)
            else:
                out_l2 = layer(out_l2)
        del out_l2_down

        out_l3_down = self.level3_0(out_l2) #16
        for i, layer in enumerate(self.level_3):
            if i == 0:
                out_l3 = layer(out_l3_down)
            else:
                out_l3 = layer(out_l3)
        del out_l3_down

        # print(out_l3.shape)
        dec_l3_l2 = self.up_l3_l2(out_l3)
        merge_l2 = self.merge_l2(torch.cat([dec_l3_l2, out_l2], 1))
        for i, layer in enumerate(self.dec_l2):
            if i == 0:
                dec_l2 = layer(merge_l2)
            else:
                dec_l2 = layer(dec_l2)

        dec_l2_l1 = self.up_l2_l1(dec_l2)
        merge_l1 = self.merge_l1(torch.cat([dec_l2_l1, out_l1], 1))
        for i, layer in enumerate(self.dec_l1):
            if i == 0:
                dec_l1 = layer(merge_l1)
            else:
                dec_l1 = layer(dec_l1)

        psp_outs = dec_l1.clone()
        for layer in self.pspModules:
            out_psp = layer(dec_l1)
            psp_outs = torch.cat([psp_outs, out_psp], 1)

        decoded = self.classifier(psp_outs)

        return SegOut(
            enc_out=out_l3,
            # mask_out=F.upsample(decoded, size=(dim0, dim1, dim2), mode='trilinear')
            mask_out=nn.functional.interpolate(decoded, size=(dim0, dim1, dim2), mode='trilinear', align_corners=True)
        )




class SegModel(nn.Module):
    def __init__(self, best_model_loc, inp_res=(64, 64, 64)):
        super().__init__()

        self.inp_res= inp_res
        self.espnet = ESPNet(classes=4, channels=4)
        if os.path.isfile(best_model_loc):
            self.espnet.load_state_dict(torch.load(best_model_loc, map_location='cpu'))
        else:
            print('ERROR')

        self.espnet = self.espnet

        del self.espnet.up_l3_l2
        del self.espnet.merge_l2
        del self.espnet.dec_l2
        del self.espnet.up_l2_l1

        del self.espnet.merge_l1
        del self.espnet.dec_l1
        del self.espnet.pspModules
        del self.espnet.classifier

        self.lin = nn.Linear(256, 3)
        # initialize linear layer weights


    def forward(self, input1):

        inp_res = self.inp_res
        batch_size = input1.shape[0]
        dim0 = input1.size(2)
        dim1 = input1.size(3)
        dim2 = input1.size(4)


        if self.training or inp_res is None:
            # input resolution should be divisible by 8
            inp_res = (math.ceil(dim0 / 8) * 8, math.ceil(dim1 / 8) * 8,
                       math.ceil(dim2 / 8) * 8)
        if inp_res:
            input1 = F.adaptive_avg_pool3d(input1, output_size=inp_res)

        out_l0 = self.espnet.level0(input1)

        for i, layer in enumerate(self.espnet.level1): #64
            if i == 0:
                out_l1 = layer(out_l0)
            else:
                out_l1 = layer(out_l1)

        out_l2_down = self.espnet.level2(out_l1) #32
        for i, layer in enumerate(self.espnet.level_2):
            if i == 0:
                out_l2 = layer(out_l2_down)
            else:
                out_l2 = layer(out_l2)
        del out_l2_down

        out_l3_down = self.espnet.level3_0(out_l2) #16
        for i, layer in enumerate(self.espnet.level_3):
            if i == 0:
                out_l3 = layer(out_l3_down)
            else:
                out_l3 = layer(out_l3)
        del out_l3_down

        out_l3 = F.adaptive_avg_pool3d(out_l3, output_size=1)
        out_l3 = out_l3.view(batch_size, -1)

        final_out = self.lin(out_l3)
        return final_out

if __name__ == '__main__':
    channels = 4
    bSz = 1
    classes = 4
    input = torch.FloatTensor(bSz, channels, 80, 80, 80)
    input_var = Variable(input).cuda()
    model = ESPNet(classes=classes, channels=channels).eval().cuda()
    out = model(input_var)
    print(out.size())
