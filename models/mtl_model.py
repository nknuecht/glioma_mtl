## Nicholas Nuechterlein

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
    # input branch (1 layer) takes SCNA data and prepares it to be concatenated with the output of the network's encoder
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



        self.channels = n_volumes # number of MRI modalities
        self.take_surv_loss = take_surv_loss # include survival loss in network loss function (yet to be implemented)
        self.standard_unlabled_loss = standard_unlabled_loss # take segmentation loss for weak labels with hard labels rather than probabilies
        self.seg_4class_weights = seg_4class_weights # loss weights for different segmentation classes (assuming 4 segmentation classes)
        self.seg_2class_weights = seg_2class_weights # loss weights for different segmentation classes (assuming 2 segmentation classes)
        self.class_loss_weights = class_loss_weights # classification loss weights
        self.seg_loss_scale = seg_loss_scale # weight of segmentation loss
        self.surv_loss_scale = surv_loss_scale # weight of survival loss
        self.brats_seg_ids = brats_seg_ids # IDs of MRI scans that have ground truth segmentation labels
        self.fusion_net_flag = fusion_net_flag # if we are training with SCNA data, we attach a second input branch
        self.modality = modality # if we have 1 channel input, we name the modality
        self.num_classes = n_classes

        # segmentation network
        self.mask_net = ESPNet(classes=4, channels=4)
        if pretrained is not None:
            assert os.path.isfile(pretrained), 'Pretrained file does not exist. {}'.format(pretrained)
            weight_dict = torch.load(pretrained, map_location=device)
            self.mask_net.load_state_dict(weight_dict)

            if self.channels == 1: # modify input layer if input has only one channel
                print('Segmentation model will only use one modality (channel)')
                level0_weight = self.mask_net.level0.conv.weight[:, 0].unsqueeze(1)
                self.mask_net.level0.conv = nn.Conv3d(1, 16, kernel_size=(7, 7, 7),
                                                         stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
                self.mask_net.level0.conv.weight = nn.Parameter(level0_weight)

        self.mask_net = self.mask_net.train()

        # SCNA input branch
        self.genome_net = GenomeNet(in_features=g_in_features, out_features=g_out_features)
        fusion_in_dim = g_out_features + self.mask_net.config[2]  # 256

        self.lin = nn.Linear(256, self.num_classes)

        # classification branch
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_in_dim, g_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=g_out_features, out_features=n_classes)
        )

        # survival branch with genomic data (not implemented yet)
        self.surv_net = nn.Sequential(
            nn.Linear(fusion_in_dim, g_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=g_out_features, out_features=1)
        )

        # survival branch without genomic data (not implemented yet)
        self.surv_net_no_genomic = nn.Sequential(
            nn.Linear(256, g_out_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=g_out_features, out_features=1)
        )



        self.global_avg = nn.AdaptiveAvgPool3d(output_size=1)
        # self.kl_loss = nn.KLDivLoss() # for further experiments
        self.softmax = nn.Softmax(dim=1)
        self.criterion_4class = nn.CrossEntropyLoss(weight=self.seg_4class_weights, ignore_index = 255)
        self.criterion_2class = nn.CrossEntropyLoss(weight=self.seg_2class_weights, ignore_index = 255)
        self.criterion_core = nn.CrossEntropyLoss(weight=self.seg_4class_weights, ignore_index = 2)
        self.class_loss = nn.CrossEntropyLoss(weight=self.class_loss_weights, ignore_index=255)
        # self.surv_criterion = LogSurvLoss() # for further experiments
        self.device = device


    def forward_masknet(self, image_data, seg_probs=None, seg_gt=None, compute_loss=True, bratsIDs=None): # we don't always want to compute the loss (i.e., if we want to evaluate)
        # get segmenation mask
        mask_out_student = self.mask_net(image_data)
        loss = 0.0

        if compute_loss:
            # take the softmax of predicted segmentation mask
            mask_out_student_sm = self.softmax(mask_out_student.mask_out)
            # if we have 1-channel input, we take different loss based on which modality is being considered
            if self.channels == 1:
                if self.modality == 't1ce' or self.modality == 't1' or self.modality == 't1ce-t1':
                    # with t1-based modalities, we ignore the edema label
                    core_loss = self.criterion_core(mask_out_student_sm, seg_gt.type(torch.int64))
                    loss = loss + core_loss
                elif self.modality == 'flair' or self.modality == 't2':
                    # for T2 and FLAIR modalities, we ignore the edema label
                    second_softmax = self.softmax(mask_out_student_sm)
                    channel1 = second_softmax[:, 0, :, :, :].unsqueeze(1)
                    channel2 = torch.sum(second_softmax[:, 1:, :, :, :], axis=1).unsqueeze(1)
                    probs_2class_seg = torch.cat((channel1, channel2), 1)
                    loss_2class_seg = self.criterion_2class(probs_2class_seg, seg_probs)
                    loss = loss + loss_2class_seg

            else:
                # find samples with ground truth vs. weak labels in the batch
                gt_seg_idx = [i for i, x in enumerate(bratsIDs) if x in self.brats_seg_ids]
                pseudo_seg_idx = [i for i, x in enumerate(bratsIDs) if x not in self.brats_seg_ids]

                if len(gt_seg_idx) > 0:
                    # consider scans with ground truth segmenation labels
                    gt_probs = mask_out_student_sm[gt_seg_idx]
                    gt_segs = seg_gt[gt_seg_idx]

                    gt_ce_loss = self.criterion_4class(gt_probs, gt_segs.type(torch.int64))
                    loss = loss + gt_ce_loss

                if len(pseudo_seg_idx) > 0:
                    # consider scans with weak segmenation labels
                    pseudo_outputs = mask_out_student_sm[pseudo_seg_idx] # pseudo_probs
                    pseudo_segs = seg_gt[pseudo_seg_idx]

                    # take loss between predicted segmenation mask and a weak segmenation mask
                    if self.standard_unlabled_loss: # regular loss
                        seg = copy.deepcopy(pseudo_segs)
                        seg[pseudo_segs > 1] == 255
                        gt_loss = self.criterion_4class(pseudo_outputs, seg.type(torch.int64))
                        loss = loss + gt_loss

                    else:
                        pseudo_wt_seg = seg_probs[pseudo_seg_idx] # this is from the teacher (passed in) # teacher_probs
                        pseudo_softmax = self.softmax(pseudo_outputs)
                        channel1 = pseudo_softmax[:, 0, :, :, :].unsqueeze(1)
                        channel2 = torch.sum(pseudo_softmax[:, 1:, :, :, :], axis=1).unsqueeze(1)
                        pseudo_probs_seg = torch.cat((channel1, channel2), 1)
                        pseudo_thresh_loss = self.criterion_2class(pseudo_probs_seg, pseudo_wt_seg)

                        loss = loss + pseudo_thresh_loss

        return mask_out_student, loss

    def forward(self, image_data, genome_data, seg_probs=None, seg_gt=None, class_labels=None, compute_loss=True, event=None, OS=None, bratsIDs=None):

        bsz = image_data.size(0) # batch size

        # get segmentation output
        seg_object, seg_loss = self.forward_masknet(image_data=image_data,
                                                    seg_probs=seg_probs,
                                                    seg_gt=seg_gt,
                                                    compute_loss=compute_loss,
                                                    bratsIDs=bratsIDs) ## mask_out is SegOut type.

        # get pooled encoder output
        enc_out_pool = self.global_avg(seg_object.enc_out).contiguous().view(bsz, -1)

        if self.fusion_net_flag:
            genome_out = self.genome_net(genome_data.float()) # pass SCNA data though input branch
            assert genome_out.dim() == enc_out_pool.dim()

            # Survival risk (yet to implement)
            surv_risk = self.surv_net(torch.cat([genome_out, enc_out_pool], dim=-1))

            # Fusion MR data and SCNA data
            out = self.fusion_net(torch.cat([genome_out, enc_out_pool], dim=-1))
        else:
            # Survival risk (yet to implement)
            surv_risk = self.surv_net_no_genomic(enc_out_pool)

            # Fusion MR data and SCNA data
            out = self.lin(enc_out_pool)



        if self.take_surv_loss:
            temp = 0 # placeholder
        else:
            loss_surv, ci = torch.tensor(0).to(self.device), -1 # placeholder

        # Classification Loss
        class_loss = self.class_loss(out, class_labels.long())

        loss_surv, ci = torch.tensor(0).to(self.device), torch.tensor(0).to(self.device) #-1 ## hack should leave this out.

        return GBMOut(
            seg_out=seg_object.mask_out,
            class_out=out,
            seg_loss=seg_loss * self.seg_loss_scale,
            class_loss=class_loss,
            # surv_loss=loss_surv * self.surv_loss_scale, (yet to implement)
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
