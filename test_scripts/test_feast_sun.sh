#!/bin/bash
python ../test.py --multigpu 1 --lastgpu 1 --checkpoint_path_file ./../checkpoints/sunrgbd/feast.pth.tar --dataset_path ./../dataset/sunrgbd --dataset_folder h5_geometric --exp_name ./../results/test/cm_feast_sun.npy --nfeatures 3 --coordnode 1 --model_config 'ggrad_0.1_100_0,feast_16_8_0, b_0, r_0, prnn_max_0.1_0.15_100_0, rfeast_16_8_0, b_0, r_0, rfeast_16_8_0, b_0, r_0, prnn_max_0.15_0.25_100_0, rfeast_32_8_0, b_0, r_0, rfeast_32_8_0, b_0, r_0, prnn_max_0.25_0.35_100_0, rfeast_64_8_0, b_0, r_0, rfeast_64_8_0, b_0, r_0, prnn_max_0.35_0.55_100_0, rfeast_128_8_1, b_1, r_1, rfeast_128_8_1, b_1,  r_1, prnn_max_0.55_0.55_100_1, gp_avg_1, f_128_1, b_1, r_1, d_0.2_1, f_19_1'