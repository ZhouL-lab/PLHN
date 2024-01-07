from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
#from lib.loss.loss_helper import FSAuxCELoss, FSAuxRMILoss, FSCELoss
from Model.modelv5.logger import Logger as Log

class FSweightCELoss(nn.Module):
    def __init__(self, ):
        super(FSCELoss, self).__init__()

        weight =[0,1]
        weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'

        ignore_index = -1

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)
            print('len(inputs):', len(inputs))
            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0].squeeze(1), (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()

class FSCELoss(nn.Module):
    def __init__(self, ):
        super(FSCELoss, self).__init__()


        reduction = 'elementwise_mean'

        ignore_index = -1

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)
            print('len(inputs):', len(inputs))
            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3),inputs.size(4)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3),inputs.size(4)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0].squeeze(1), (inputs.size(2), inputs.size(3),inputs.size(4)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class PPC(nn.Module, ABC):
    def __init__(self, ):
        super(PPC, self).__init__()
        self.ignore_label = -1

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)

        return loss_ppc


class PPD(nn.Module, ABC):
    def __init__(self, ):
        super(PPD, self).__init__()

        self.ignore_label = -1

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd


class PixelPrototypeCELoss(nn.Module, ABC):
    def __init__(self, ):
        super(PixelPrototypeCELoss, self).__init__()

        ignore_index = -1
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_ppc_weight = 0.01
        self.loss_ppd_weight = 0.001
        self.seg_criterion = FSCELoss()

        self.ppc_criterion = PPC()
        self.ppd_criterion = PPD()

    def forward(self, preds, target):
        d,h, w = target.size(2), target.size(3),target.size(4)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            pred = F.interpolate(input=seg, size=(d ,h, w), mode='trilinear', align_corners=True)

            loss = self.seg_criterion(pred, target)
            return loss+self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd

        seg = preds
        pred = F.interpolate(input=seg, size=(d,h, w), mode='trilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return loss





