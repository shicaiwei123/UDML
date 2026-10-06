#!/bin/bash

# Clean UDML
python main_auxi_weight_udml.py --ckpt_path ./results/cramed/udml_clean --dataset CREMAD --gpu_ids 0 --modulation Normal --train --num_frame 1 --pe 1 --noise_type None --beta 1e-5 --gamma 4.0

# Gaussian noise UDML
python main_auxi_weight_udml.py --ckpt_path ./results/cramed/udml_noise_cycle50_gaussian_0_11 --dataset CREMAD --gpu_ids 0 --modulation Normal --train --num_frame 1 --pe 1 --noise_type Gaussian --beta 1e-5 --gamma 4.0
