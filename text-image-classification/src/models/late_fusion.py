#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert import BertEncoder,BertClf
from .image import ImageEncoder,ImageClf


class ConcatFusion_AUXI(nn.Module):
    """Auxiliary fusion module: outputs unimodal logits (via zero-masking) + fusion logits."""
    def __init__(self, input_dim=1024, output_dim=100):
        super().__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        x_out = self.fc_out(torch.cat((x, torch.zeros_like(y)), dim=1))
        y_out = self.fc_out(torch.cat((torch.zeros_like(x), y), dim=1))
        return x_out, y_out, output


class MultimodalLateFusionUDML(nn.Module):
    """UDML: PE inside encoder + variance-guided fusion.

    Forward returns 11 values:
      fusion_logits, txt_logits, img_logits,
      txt_var, img_var, weight_t, weight_v,
      txt_mu, txt_std, img_mu, img_std
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        from udml import BertClf_UDML, ImageClf_UDML, ConcatFusion_AUXI
        self.txtclf = BertClf_UDML(args)
        self.imgclf = ImageClf_UDML(args)
        fusion_dim = getattr(args, 'fusion_dim', 512)

        self.txt_variance_estimator = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.Dropout(0.1), nn.Linear(256, 1))
        self.img_variance_estimator = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, txt, mask, segment, img):
        txt_logits, txt_s, txt_mu, txt_std = self.txtclf(txt, mask, segment)
        img_logits, img_s, img_mu, img_std = self.imgclf(img)

        txt_var = (self.txt_variance_estimator(txt_std.detach()) * 0.5).exp()
        img_var = (self.img_variance_estimator(img_std.detach()) * 0.5).exp()

        # Raw weights from variance
        raw_wt = 2 * img_var ** 2 / (img_var ** 2 + txt_var ** 2 + 1e-8)
        raw_wv = 2 * txt_var ** 2 / (img_var ** 2 + txt_var ** 2 + 1e-8)

        # Scale by modality depend (logit magnitude), matching original UDML
        t_dep = getattr(self.args, 'text_depend', 1.0)
        v_dep = getattr(self.args, 'visual_depend', 1.0)
        wt = raw_wt / t_dep
        wv = raw_wv / v_dep
        ws_ = wt + wv + 1e-8
        wt, wv = 2 * wt / ws_, 2 * wv / ws_

        # Warmup: equal weights
        if self.training and hasattr(self.args, 'current_epoch') and \
           self.args.current_epoch < self.args.cylcle_epoch + 10:
            wt = torch.ones_like(wt)
            wv = torch.ones_like(wv)

        # Feature reweighting + fusion (return all 3 fusion outputs)
        fuse_out=txt_logits*wt+img_logits*wv

        return fuse_out, txt_logits, img_logits, txt_var, img_var, wt, wv, txt_mu, txt_std, img_mu, img_std


class MultimodalLateFusionClf(nn.Module):
    def __init__(self, args):
        super(MultimodalLateFusionClf, self).__init__()
        self.args = args

        self.txtclf = BertClf(args)
        self.imgclf= ImageClf(args)


    def forward(self, txt, mask, segment, img):
        txt_out = self.txtclf(txt, mask, segment)
        img_out = self.imgclf(img)

        txt_energy = torch.log(torch.sum(torch.exp(txt_out), dim=1))
        img_energy = torch.log(torch.sum(torch.exp(img_out), dim=1))

        txt_conf = txt_energy / 10
        img_conf = img_energy / 10
        txt_conf = torch.reshape(txt_conf, (-1, 1))
        img_conf = torch.reshape(img_conf, (-1, 1))

        if self.args.df:
            txt_img_out = (txt_out * txt_conf.detach() + img_out * img_conf.detach())
        else:
            txt_conf.detach()
            img_conf.detach()
            txt_img_out=0.5*txt_out+0.5*img_out

        return txt_img_out, txt_out, img_out, txt_conf, img_conf
