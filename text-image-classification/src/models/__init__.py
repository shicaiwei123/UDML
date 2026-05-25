#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from .late_fusion import MultimodalLateFusionClf, MultimodalLateFusionUDML
MODELS = {
    'latefusion': MultimodalLateFusionClf,
    'latefusion_udml': MultimodalLateFusionUDML,
}


def get_model(args):
    return MODELS[args.model](args)
