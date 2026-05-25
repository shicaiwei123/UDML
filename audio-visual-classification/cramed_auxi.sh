#!/bin/bash

python main_auxi_weight_udml.py --ckpt_path ./results/cramed/udml --modality full --dataset CREMAD --gpu_ids 1 --modulation Normal  --train --num_frame 1 --pe 1 --beta 1e-5 --gamma 4.0
python main_auxi_weight_udml.py --ckpt_path ./results/cramed/udml_origin_schdue --modality full --dataset CREMAD --gpu_ids 0 --modulation Normal  --train --num_frame 1 --pe 1 --beta 1e-5 --gamma 4.0
