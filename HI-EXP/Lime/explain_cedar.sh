#!/bin/bash

### Execute this Shell script if you want to compute the Explanations for the 'CEDAR' instances
echo "Explaining 'CEDAR-Letter-0005_AdamW_0.03' with 'Ridge' as Surrogate Model"
python explain_removing_patches_cedar5train.py -test_id CEDAR-Letter-0005_AdamW_0.03 -model NN -block_width 75 -block_height 75 -crop_size 380 -surrogate_model Ridge -lime_iters 3
python explain_removing_patches_cedar5test.py -test_id CEDAR-Letter-0005_AdamW_0.03 -model NN -block_width 75 -block_height 75 -crop_size 380 -surrogate_model Ridge -lime_iters 3