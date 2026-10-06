#!/bin/bash

# Clean UDML
python main_auxi_weight_udml.py --ckpt_path ./results/ks/udml_clean --dataset KineticSound --gpu_ids 0 --modulation Normal --train --num_frame 3 --pe 1 --noise_type None --beta 0 --gamma 2.5

# Gaussian noise UDML
python main_auxi_weight_udml.py --ckpt_path ./results/ks/udml_noise_cycle50_gaussian_0_11 --dataset KineticSound --gpu_ids 0 --modulation Normal --train --num_frame 3 --pe 1 --noise_type Gaussian --beta 0 --gamma 2.5
