#!/bin/bash

python main_auxi_weight_udml.py --ckpt_path /root/autodl-tmp/results/ks/udml_35 --modality full --dataset KineticSound --gpu_ids 0 --modulation Normal  --train --num_frame 3 --pe 1 --beta 0 --gamma 2.5
