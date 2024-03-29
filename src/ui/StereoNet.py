"""
Classes and functions to instantiate, and train, a StereoNet model (https://arxiv.org/abs/1807.08865).

StereoNet model is decomposed into a feature extractor, cost volume creation, and a cascade of refiner networks.

Loss function is the Robust Loss function (https://arxiv.org/abs/1701.03077)
"""

from typing import Dict, Any
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class StereoNet(pl.LightningModule):

    def __init__(self, k_downsampling_layers = 3,
                 k_refinement_layers = 3,
                 candidate_disparities = 256,
                 feature_extractor_filters = 32,
                 cost_volumizer_filters = 32,
                 mask: bool = True):
        
        super().__init__()
        self.save_hyperparameters()

        self.k_downsampling_layers = k_downsampling_layers
        self.k_refinement_layers = k_refinement_layers
        self.candidate_disparities = candidate_disparities
        self.mask = mask
        self.feature_extractor_filters = feature_extractor_filters
        self.cost_volumizer_filters = cost_volumizer_filters
        self._max_downsampled_disps = \
            (candidate_disparities+1) // (2**k_downsampling_layers)

        self.feature_extractor = FeatureExtractor(
            in_channels=3, out_channels=self.feature_extractor_filters, 
            k_downsampling_layers=self.k_downsampling_layers
            )
        self.cost_volumizer = CostVolume(
            in_channels=self.feature_extractor_filters, 
            out_channels=self.cost_volumizer_filters, 
            max_downsampled_disps=self._max_downsampled_disps
            )

        self.refiners = nn.ModuleList()
        for _ in range(self.k_refinement_layers):
            self.refiners.append(Refinement())

    def forward_pyramid(self, sample, side = 'left'):

        if side == 'left':
            reference = sample['left']
            shifting = sample['right']
        elif side == 'right':
            reference = sample['right']
            shifting = sample['left']

        reference_embedding = self.feature_extractor(reference)
        shifting_embedding = self.feature_extractor(shifting)

        cost = self.cost_volumizer(
            (reference_embedding, shifting_embedding), 
            side=side
            )

        disparity_pyramid = [soft_argmin(cost, self.candidate_disparities)]

        for idx, refiner in enumerate(self.refiners, start=1):
            scale = (2**self.k_refinement_layers) / (2**idx)
            new_h = int(reference.size()[2]//scale)
            new_w = int(reference.size()[3]//scale)
            reference_rescaled = F.interpolate(
                reference, [new_h, new_w], 
                mode='bilinear', align_corners=True
                )
            disparity_low_rescaled = F.interpolate(
                disparity_pyramid[-1], [new_h, new_w], 
                mode='bilinear', align_corners=True
                )
            refined_disparity = F.relu(
                refiner(
                    torch.cat((reference_rescaled, disparity_low_rescaled), 
                              dim=1)) + disparity_low_rescaled
                )
            disparity_pyramid.append(refined_disparity)

        return disparity_pyramid

    def forward(self, sample): 

        disparities = self.forward_pyramid(sample, side='left')
        return disparities[-1]

    def training_step(self, batch, _):

        left = batch['left']
        right = batch['right']
        disp_gt_left = batch['disp_left']
        disp_gt_right = batch['disp_right']

        sample = {'left': left, 'right': right}

        disp_pred_left_nonuniform = self.forward_pyramid(sample, side='left')
        disp_pred_right_nonuniform = self.forward_pyramid(sample, side='right')

        for idx, (disparity_left, disparity_right) in \
            enumerate(zip(
            disp_pred_left_nonuniform, 
            disp_pred_right_nonuniform
            )):
            disp_pred_left_nonuniform[idx] = F.interpolate(
                disparity_left, [left.size()[2], left.size()[3]], 
                mode='bilinear', align_corners=True
                )
            disp_pred_right_nonuniform[idx] = F.interpolate(
                disparity_right, [left.size()[2], left.size()[3]], 
                mode='bilinear', align_corners=True
                )

        disp_pred_left = torch.stack(disp_pred_left_nonuniform, dim=0)
        disp_pred_right = torch.stack(disp_pred_right_nonuniform, dim=0)

        def _tiler(tensor, matching_size = None):
            if matching_size is None:
                matching_size = [disp_pred_left.size()[0], 1, 1, 1, 1]
            return tensor.tile(matching_size)

        disp_gt_left = _tiler(disp_gt_left)
        disp_gt_right = _tiler(disp_gt_right)

        if self.mask:
            left_mask = (disp_gt_left < self.candidate_disparities).detach()
            right_mask = (disp_gt_right < self.candidate_disparities).detach()

            loss_left = torch.mean(robust_loss(
                disp_gt_left[left_mask] - disp_pred_left[left_mask], 
                alpha=1, c=2
                ))
            loss_right = torch.mean(robust_loss(
                disp_gt_right[right_mask] - disp_pred_right[right_mask], 
                alpha=1, c=2
                ))
        else:
            loss_left = torch.mean(robust_loss(
                disp_gt_left - disp_pred_left, 
                alpha=1, c=2
                ))
            loss_right = torch.mean(robust_loss(
                disp_gt_right - disp_pred_right, 
                alpha=1, c=2
                ))

        loss = (loss_left + loss_right) / 2

        self.log("train_loss_step", loss, on_step=True, 
                 on_epoch=False, prog_bar=True, logger=True
                 )
        self.log("train_loss_epoch", 
                 F.l1_loss(disp_pred_left[-1], disp_gt_left[-1]), 
                 on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True
                 )
        return loss

    def validation_step(self, batch, batch_idx):

        left = batch['left']
        right = batch['right']
        disp_gt = batch['disp_left']

        sample = {'left': left, 'right': right}

        disp_pred = self(sample)

        loss = F.l1_loss(disp_pred, disp_gt)
        self.log("val_loss_epoch", loss, on_epoch=True, logger=True)

    def configure_optimizers(self) -> Dict[str, Any]:

        optimizer = torch.optim.RMSprop(
            self.parameters(), lr=1e-3, weight_decay=0.0001
            )
        lr_dict = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.9, last_epoch=-1
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "ExponentialDecayLR"
            }
        config = {"optimizer": optimizer, "lr_scheduler": lr_dict}
        return config


class FeatureExtractor(torch.nn.Module):

    def __init__(self, in_channels, out_channels, k_downsampling_layers):

        super().__init__()
        self.k = k_downsampling_layers
        net = OrderedDict()

        for block_idx in range(self.k):
            net[f'segment_0_conv_{block_idx}'] = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, 
                kernel_size=5, stride=2, padding=2
                )
            in_channels = out_channels

        for block_idx in range(6):
            net[f'segment_1_res_{block_idx}'] = ResBlock(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=3, padding=1
                )

        net['segment_2_conv_0'] = nn.Conv2d(
            in_channels=32, out_channels=32, 
            kernel_size=3, padding=1
            )

        self.net = nn.Sequential(net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class CostVolume(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int, max_downsampled_disps: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self._max_downsampled_disps = max_downsampled_disps

        net: OrderedDict[str, nn.Module] = OrderedDict()

        for block_idx in range(4):
            net[f'segment_0_conv_{block_idx}'] = nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels, 
                kernel_size=3, padding=1
                )
            net[f'segment_0_bn_{block_idx}'] = nn.BatchNorm3d(
                num_features=out_channels
                )
            net[f'segment_0_act_{block_idx}'] = nn.LeakyReLU(
                negative_slope=0.2
                )

            in_channels = out_channels

        net['segment_1_conv_0'] = nn.Conv3d(
            in_channels=out_channels, out_channels=1, 
            kernel_size=3, padding=1
            )

        self.net = nn.Sequential(net)

    def forward(self, x, side = 'left'):
        reference_embedding, target_embedding = x

        cost = compute_volume(
            reference_embedding, target_embedding, 
            max_downsampled_disps=self._max_downsampled_disps, side=side
            )

        cost = self.net(cost)
        cost = torch.squeeze(cost, dim=1)

        return cost


def compute_volume(reference_embedding: torch.Tensor, target_embedding: torch.Tensor, max_downsampled_disps: int, side: str = 'left') -> torch.Tensor:
    """
    Refer to the doc string in CostVolume.forward.
    Refer to https://github.com/meteorshowers/X-StereoLab/blob/9ae8c1413307e7df91b14a7f31e8a95f9e5754f9/disparity/models/stereonet_disp.py

    This difference based cost volume is also reflected in an implementation of the popular DispNetCorr:
        Line 81 https://github.com/wyf2017/DSMnet/blob/b61652dfb3ee84b996f0ad4055eaf527dc6b965f/models/util_conv.py
    """
    batch, channel, height, width = reference_embedding.size()
    cost = torch.Tensor(batch, channel, max_downsampled_disps, height, width).zero_()
    cost = cost.type_as(reference_embedding)  # PyTorch Lightning handles the devices
    cost[:, :, 0, :, :] = reference_embedding - target_embedding
    for idx in range(1, max_downsampled_disps):
        if side == 'left':
            cost[:, :, idx, :, idx:] = reference_embedding[:, :, :, idx:] - target_embedding[:, :, :, :-idx]
        if side == 'right':
            cost[:, :, idx, :, :-idx] = reference_embedding[:, :, :, :-idx] - target_embedding[:, :, :, idx:]
    cost = cost.contiguous()

    return cost


class Refinement(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

        dilations = [1, 2, 4, 8, 1, 1]

        net: OrderedDict[str, nn.Module] = OrderedDict()

        net['segment_0_conv_0'] = nn.Conv2d(
            in_channels=4, out_channels=32, 
            kernel_size=3, padding=1
            )

        for block_idx, dilation in enumerate(dilations):
            net[f'segment_1_res_{block_idx}'] = ResBlock(
                in_channels=32, out_channels=32, 
                kernel_size=3, padding=dilation, dilation=dilation
                )

        net['segment_2_conv_0'] = nn.Conv2d(
            in_channels=32, out_channels=1, 
            kernel_size=3, padding=1
            )

        self.net = nn.Sequential(net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class ResBlock(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, stride=stride, 
            padding=padding, dilation=dilation, bias=False
            )
        self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels)
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv_2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, 
            kernel_size=kernel_size, stride=stride, 
            padding=padding, dilation=dilation, bias=False
            )
        self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels)
        self.activation_2 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        res = self.conv_1(x)
        res = self.batch_norm_1(res)
        res = self.activation_1(res)
        res = self.conv_2(res)
        res = self.batch_norm_2(res)
        out = res + x
        out = self.activation_2(out)

        return out


def soft_argmin(cost: torch.Tensor, max_downsampled_disps: int) -> torch.Tensor:
    """
    Soft argmin function described in the original paper.  The disparity grid creates the first 'd' value in equation 2 while
    cost is the C_i(d) term.  The exp/sum(exp) == softmax function.
    """
    disparity_softmax = F.softmax(-cost, dim=1)
    disparity_grid = torch.linspace(0, max_downsampled_disps, disparity_softmax.size(1)).reshape(1, -1, 1, 1)
    disparity_grid = disparity_grid.type_as(disparity_softmax)

    disp = torch.sum(disparity_softmax * disparity_grid, dim=1, keepdim=True)

    return disp


def robust_loss(x: torch.Tensor, alpha, c) -> torch.Tensor:  # pylint: disable=invalid-name
    """
    A General and Adaptive Robust Loss Function (https://arxiv.org/abs/1701.03077)
    """
    f: torch.Tensor = (abs(alpha - 2) / alpha) * (torch.pow(torch.pow(x / c, 2)/abs(alpha - 2) + 1, alpha/2) - 1)  # pylint: disable=invalid-name
    return f
