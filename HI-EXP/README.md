# xAI-WriterIdentification/HI-EXP 👨🏻‍🏫

Here you have a brief guide to execute the whole process.

## 1. Dataset Creation 📄📄📄
Create a new directory inside the [dataset](https://github.com/DecioXXIV/xAI-WriterIdentification/tree/main/datasets) one and put here the data you want to use for your experiments.
*Be careful!* The instances must be labeled and you have to create a subfolder for each label.

## 2. Fine-Tuning 🔧
Pick a model to fine-tune ([classifier_NN](https://github.com/DecioXXIV/xAI-WriterIdentification/tree/main/HI-EXP/classifier_NN), [classifier_SVM](https://github.com/DecioXXIV/xAI-WriterIdentification/tree/main/HI-EXP/classifier_SVM), [classifier_GB](https://github.com/DecioXXIV/xAI-WriterIdentification/tree/main/HI-EXP/classifier_GB)) and create a new script following the model of `ft_CEDAR_Letter_5.py` and `ft_CVL_1.py`: make sure to use the Dataset created in _Step 1_.

## 3. Explanations Computation 🧠📚
Employ what is stored within the [Lime]() directory to evaluate the model fine-tuned on your data.

Create new scripts following the models of `explain_removing_patches_cedar5train.py` or explain_removing_patches_cedar5test.py: make sure to use the Model fine-tuned in _Step 2_.

## 4. Instance Masking 📄🤬
We decided to exploit the results of the explanations on the training instances perturbing them by removing the 75x75 patches that proved most relevant to the performance obtained by the model.

You can perform what described by using what is implemented in `image_masker_from_exp.py` and scripted in `mask_instances.sh`: don't forget to put the instances you want to mask into the [mask_images](https://github.com/DecioXXIV/xAI-WriterIdentification/tree/main/HI-EXP/Lime/mask_images) folder before running the Shell script.

## 5. Dataset re-Creation 📄📄📄🔄
You can employ the masked Instances (computed in _Step 4_) to create a new Dataset: follow what described under the _Step 1_ section.

## 6. Fine-Tuning (again!) 🔧🔄
You can employ the new Dataset (with the masked Instances) computer in _Step 5_ to perform a new Fine-Tuning of the original System.

## 7. Re-Explaining 🧠📚🔄 and Outcome Comparisons 
You can start from the Model obtained in Step 6 to re-compute the Explanations: at this point, we suggest to compare the Explanations generated on the Test Set instances in _Step 3_ with those generated in this step on the same instances.
