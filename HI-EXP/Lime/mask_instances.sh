#!/bin/bash

### Execute this Shell script if you want to mask the "full" instances by using what computed in the Explanation process
# Remember to put the "full" instances in the 'mask_images' folder

# CEDAR-Letter-0005_AdamW_0.03 with 'Ridge' as surrogate model
python image_masker_from_exp.py -instance 0001a -test_id CEDAR-Letter-0005_AdamW_0.03 -model NN -timestamp 2024.08.09_22.08.27.878021 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0001b -test_id CEDAR-Letter-0005_AdamW_0.03 -model NN -timestamp 2024.08.09_22.08.27.878021 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0002a -test_id CEDAR-Letter-0005_AdamW_0.03 -model NN -timestamp 2024.08.09_22.08.27.878021 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0002b -test_id CEDAR-Letter-0005_AdamW_0.03 -model NN -timestamp 2024.08.09_22.08.27.878021 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0003a -test_id CEDAR-Letter-0005_AdamW_0.03 -model NN -timestamp 2024.08.09_22.08.27.878021 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0003b -test_id CEDAR-Letter-0005_AdamW_0.03 -model NN -timestamp 2024.08.09_22.08.27.878021 -surrogate_model Ridge

# CVL-0001_SGD_0.09 with 'Ridge' as surrogate model
python image_masker_from_exp.py -instance 0002-1 -test_id CVL-0001_SGD_0.09 -model NN -timestamp 2024.08.22_21.14.56.132395 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0002-6 -test_id CVL-0001_SGD_0.09 -model NN -timestamp 2024.08.22_21.14.56.132395 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0002-8 -test_id CVL-0001_SGD_0.09 -model NN -timestamp 2024.08.22_21.14.56.132395 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0017-1 -test_id CVL-0001_SGD_0.09 -model NN -timestamp 2024.08.22_21.14.56.132395 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0017-6 -test_id CVL-0001_SGD_0.09 -model NN -timestamp 2024.08.22_21.14.56.132395 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0017-8 -test_id CVL-0001_SGD_0.09 -model NN -timestamp 2024.08.22_21.14.56.132395 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0023-1 -test_id CVL-0001_SGD_0.09 -model NN -timestamp 2024.08.22_21.14.56.132395 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0023-6 -test_id CVL-0001_SGD_0.09 -model NN -timestamp 2024.08.22_21.14.56.132395 -surrogate_model Ridge
python image_masker_from_exp.py -instance 0023-8 -test_id CVL-0001_SGD_0.09 -model NN -timestamp 2024.08.22_21.14.56.132395 -surrogate_model Ridge