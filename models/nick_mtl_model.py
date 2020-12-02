import torch
from torch import nn
import math
from models.ESPNet import ESPNet, SegOut
import os
from torch.nn import functional as F
import torch
from typing import List, NamedTuple, Optional
from torch import Tensor
import numpy as np
import copy

import sys

sys.path.append("../")

# from beibin_surv.surv_model import LogSurvLoss
from utils import get_bb_3D_torch

import pdb

GBMOut = NamedTuple(
    "GBMOut",
    [
        ("seg_out", Tensor),  # N x C x H/4 x W/4
        ("class_out", Tensor),  # N x C x H x W
        ('seg_loss', Tensor),
        ('class_loss', Tensor),
        ('surv_loss', Tensor),
        ('surv_risk', Tensor),
        ('ci', Tensor)
    ],
)


class GenomeNet(nn.Module):

    def __init__(self, in_features, out_features, proj_factor=4):
        super(GenomeNet, self).__init__()

        proj_features = in_features * proj_factor
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=proj_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=proj_features, out_features=out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

# --------
class GBMNetMTL(nn.Module):
    def __init__(self,
                 g_in_features,
                 g_out_features,
                 n_classes,
                 n_volumes=4,
                 seg_classes=4,
                 pretrained=None,
                 class_loss_weights=None,
                 seg_4class_weights=None,
                 seg_2class_weights=None,
                 seg_loss_scale=1,
                 surv_loss_scale=1,
                 device='cpu',
                 brats_seg_ids=None,
                 standard_unlabled_loss=True,
                 fusion_net_flag=True,
                 modality=None,
                 take_surv_loss=False):

        super().__init__()
        print('GBMNet!')
        self.mask_net = ESPNet(classes=4, channels=4)
        self.channels = n_volumes
        # self.teacher = ESPNet(classes=seg_classes, channels=n_volumes)
        # print("seg loss weights", seg_loss_weights)
        # print("cls loss weights", class_loss_weights)
        self.take_surv_loss = take_surv_loss
        self.standard_unlabled_loss = standard_unlabled_loss
        self.seg_loss_weights = seg_4class_weights
        self.seg_4class_weights = seg_4class_weights
        self.seg_2class_weights = seg_2class_weights
        self.class_loss_weights = class_loss_weights
        self.seg_loss_scale = seg_loss_scale
        self.surv_loss_scale = surv_loss_scale
        self.brats_seg_ids = brats_seg_ids
        self.fusion_net_flag = fusion_net_flag
        self.modality = modality

        if pretrained is not None:
            assert os.path.isfile(pretrained), 'Pretrained file does not exist. {}'.format(pretrained)
            weight_dict = torch.load(pretrained, map_location=device)
            self.mask_net.load_state_dict(weight_dict)
            # self.teacher.load_state_dict(weight_dict)

            if self.channels == 1:
                print('Segmentation model will only use one modality (channel)')
                level0_weight = self.mask_net.level0.conv.weight[:, 0].unsqueeze(1)
                self.mask_net.level0.conv = nn.Conv3d(1, 16, kernel_size=(7, 7, 7),
                                                         stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
                self.mask_net.level0.conv.weight = nn.Parameter(level0_weight)


        self.mask_net = self.mask_net.train()


        self.genome_net = GenomeNet(in_features=g_in_features, out_features=g_out_features)

        fusion_in_dim = g_out_features + self.mask_net.config[2]  # 256

        self.num_classes = n_classes
        self.lin = nn.Linear(256, self.num_classes)

        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_in_dim, g_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=g_out_features, out_features=n_classes)
        )

        self.surv_net = nn.Sequential(
            nn.Linear(fusion_in_dim, g_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=g_out_features, out_features=1)
        )

        self.surv_net_no_genomic = nn.Sequential(
            nn.Linear(256, g_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=g_out_features, out_features=1)
        )

        # self.surv_criterion = LogSurvLoss()

        self.global_avg = nn.AdaptiveAvgPool3d(output_size=1)
        self.kl_loss = nn.KLDivLoss()
        self.softmax = nn.Softmax(dim=1)
        self.criterion_4class = nn.CrossEntropyLoss(weight=self.seg_4class_weights, ignore_index = 255)
        self.criterion_2class = nn.CrossEntropyLoss(weight=self.seg_2class_weights, ignore_index = 255)
        self.criterion_core = nn.CrossEntropyLoss(weight=self.seg_4class_weights, ignore_index = 2)
        # self.seg_loss = nn.CrossEntropyLoss(weight=self.seg_loss_weights, ignore_index = 255) # TO DO
        self.class_loss = nn.CrossEntropyLoss(weight=self.class_loss_weights, ignore_index=255)
        self.device = device


    def forward_masknet(self, image_data, seg_probs=None, seg_gt=None, compute_loss=True, bratsIDs=None): # we don't always want to compute the loss (i.e., if we want to evaluate)
        # segment the MR volumes using pretrained network
        # mask_out : SegOut = self.mask_net(x_vol) # mask_out is of TYPE SegOut

        # compute loss (kl and maybe cross entropy)

        # segment with the student network (that we are contining to train)
        mask_out_student = self.mask_net(image_data)
        loss = 0.0

        if compute_loss:
            # pdb.set_trace()
            # take the softmax of both segmentation outputs
            mask_out_student_sm = self.softmax(mask_out_student.mask_out)

            if self.modality == 't1ce' or self.modality == 't1' or self.modality == 't1ce-t1':

                core_loss = self.criterion_core(mask_out_student_sm, seg_gt.type(torch.int64))
                loss = loss + core_loss
            elif self.modality == 'flair' or self.modality == 't2':

                second_softmax = self.softmax(mask_out_student_sm)
                channel1 = second_softmax[:, 0, :, :, :].unsqueeze(1)
                channel2 = torch.sum(second_softmax[:, 1:, :, :, :], axis=1).unsqueeze(1)
                probs_2class_seg = torch.cat((channel1, channel2), 1)
                loss_2class_seg = self.criterion_2class(probs_2class_seg, seg_probs) # seg_probs is the 2class seg_wt (I haven't changed the name yet)

                loss = loss + loss_2class_seg

            else:
                gt_seg_idx = [i for i, x in enumerate(bratsIDs) if x in self.brats_seg_ids]
                pseudo_seg_idx = [i for i, x in enumerate(bratsIDs) if x not in self.brats_seg_ids]

                if len(gt_seg_idx) > 0:
                    # scans with ground truth
                    gt_probs = mask_out_student_sm[gt_seg_idx]
                    gt_segs = seg_gt[gt_seg_idx]

                    gt_ce_loss = self.criterion_4class(gt_probs, gt_segs.type(torch.int64))
                    loss = loss + gt_ce_loss

                if len(pseudo_seg_idx) > 0:

                    # without ground truth (average these losses?)
                    pseudo_outputs = mask_out_student_sm[pseudo_seg_idx] # pseudo_probs
                    pseudo_segs = seg_gt[pseudo_seg_idx]
                    pseudo_wt_seg = seg_probs[pseudo_seg_idx] # this is from the teacher (passed in) # teacher_probs


                    # # # regular loss
                    if self.standard_unlabled_loss:
                        seg = copy.deepcopy(pseudo_segs)
                        seg[pseudo_segs > 1] == 255
                        gt_loss = self.criterion_4class(pseudo_outputs, seg.type(torch.int64))
                        loss = loss + gt_loss

                    else:
                        pseudo_softmax = self.softmax(pseudo_outputs)
                        channel1 = pseudo_softmax[:, 0, :, :, :].unsqueeze(1)
                        channel2 = torch.sum(pseudo_softmax[:, 1:, :, :, :], axis=1).unsqueeze(1)
                        pseudo_probs_seg = torch.cat((channel1, channel2), 1)
                        pseudo_thresh_loss = self.criterion_2class(pseudo_probs_seg, pseudo_wt_seg)

                        loss = loss + pseudo_thresh_loss

            # mask_out_teacher_sm = self.softmax(seg_probs)
            # loss_kl = self.kl_loss(input=mask_out_student_sm, target=mask_out_teacher_sm)

            # if

            # loss_ce = self.criterion_4class(input=mask_out_student_sm, target=seg_gt.long())
            # loss = loss + loss_ce
            #return mask_out_student, loss_ce #+ loss_kl

        # else:

        return mask_out_student, loss

    def forward(self, image_data, genome_data, seg_probs=None, seg_gt=None, class_labels=None, compute_loss=True, event=None, OS=None, bratsIDs=None):

        bsz = image_data.size(0) # ?

        # Segmentatin Part
        seg_object, seg_loss = self.forward_masknet(image_data=image_data,
                                                    seg_probs=seg_probs,
                                                    seg_gt=seg_gt,
                                                    compute_loss=compute_loss,
                                                    bratsIDs=bratsIDs) ## mask_out is SegOut type.

        enc_out_pool = self.global_avg(seg_object.enc_out).contiguous().view(bsz, -1)

        # Genomic Part
        # pdb.set_trace()
        # try:
        #     assert(torch.sum(genome_data).item() == 0)
        # except:
        #     print("ERROR on genomic input:", torch.sum(genome_data).item())

        # pdb.set_trace()

        if self.fusion_net_flag:
            genome_out = self.genome_net(genome_data.float())
            assert genome_out.dim() == enc_out_pool.dim()
            # Fusion Image and Genome
            out = self.fusion_net(torch.cat([genome_out, enc_out_pool], dim=-1))

            # Survival Loss
            surv_risk = self.surv_net(torch.cat([genome_out, enc_out_pool], dim=-1))
            if self.take_surv_loss:
                # loss_surv, ci = self.surv_criterion(surv_risk, event, OS)
                temp = 0 # placeholder
            else:
                # k = 0 ## this is a continue statemment
                loss_surv, ci = torch.tensor(0).to(self.device), -1

        else:
            out = self.lin(enc_out_pool)
            # Survival Loss
            surv_risk = self.surv_net_no_genomic(enc_out_pool)
            if self.take_surv_loss:
                # loss_surv, ci = self.surv_criterion(surv_risk, event, OS)
                temp = 0 # placeholder
            else:
                loss_surv, ci = torch.tensor(0).to(self.device), torch.tensor(0).to(self.device)



        # Classification Loss
        class_loss = self.class_loss(out, class_labels.long())

        loss_surv, ci = torch.tensor(0).to(self.device), -1 ## hack should leave this out.

        return GBMOut(
            seg_out=seg_object.mask_out,
            class_out=out,
            seg_loss=seg_loss * self.seg_loss_scale,
            class_loss=class_loss,
            # surv_loss=loss_surv * self.surv_loss_scale,
            surv_loss=torch.tensor(0).to(self.device),
            surv_risk=surv_risk,
            ci=ci
        )


if __name__ == '__main__':
    bsz = 1
    genome_in = 32
    g_out = 128
    n_volumes = 4
    device = 'cuda'
    input_vol = torch.Tensor(bsz, n_volumes, 64, 64, 64).to(device=device)
    input_genome = torch.Tensor(bsz, genome_in).to(device=device)
    model = GBMNet(g_in_features=genome_in, g_out_features=g_out, n_classes=2, n_volumes=n_volumes, seg_classes=4)
    model = model.to(device=device)

    out = model(x_vol=input_vol, x_genome=input_genome)
