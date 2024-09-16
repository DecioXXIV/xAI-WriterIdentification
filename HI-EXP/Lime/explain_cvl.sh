#!/bin/bash

### Execute this Shell script if you want to compute the Explanations for the 'CVL' instances
echo "Explaining 'CVL-0001_SGD_0.09' with 'Ridge' as Surrogate Model"
python explain_removing_patches_cvl1train.py -test_id CVL-0001_SGD_0.09 -model NN -block_width 75 -block_height 75 -crop_size 380 -surrogate_model Ridge -lime_iters 3
python explain_removing_patches_cvl1test.py -test_id CVL-0001_SGD_0.09 -model NN -block_width 75 -block_height 75 -crop_size 380 -surrogate_model Ridge -lime_iters 3