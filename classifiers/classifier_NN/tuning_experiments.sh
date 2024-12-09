#!/bin/bash

### Execute this Shell script if you want to re-run the Fine-Tuning experiments

# Fine-Tuning on "CEDAR-Letter"
echo "Fine-Tuning on 'CEDAR-Letter'..."
python ft_CEDAR_Letter_5.py -test_id CEDAR-Letter-New_AdamW_0.03 -crop_size 380 -opt adamw -lr 0.03 -train_replicas 15 -random_seed 24

# Fine-Tuning on "CVL"
echo "Fine-Tuning on 'CVL'..."
python ft_CVL_1.py -test_id CVL-New_SGD_0.09 -crop_size 380 -opt sgd -lr 0.09 -train_replicas 10 -random_seed 24