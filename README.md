# xAI-WriterIdentification 🧠🤖📄

This repository allows the user to solve the task of Handwriting Identification. Here you can find:
- [HI-SSL](https://github.com/DecioXXIV/xAI-WriterIdentification/tree/main/HI-SSL): original HI system, developed by Lorenzo Lastilla ([L9L4](https://github.com/L9L4))
- [HI-EXP](https://github.com/DecioXXIV/xAI-WriterIdentification/tree/main/HI-EXP): code to fine-tune the classifiers and generate the explanations using [LIME](https://github.com/marcotcr/lime)

## About the Explainable AI Process
### Datasets
The [dataset]() folder contains the Dataset employed for the Experiments: _CEDAR-Letter_ and _CVL_.

You can execute the `prepare_pages.py` inside each folder to create the instance which can be employed to fine-tune and then test the HI System.

### Fine-Tuning the original System to obtain a Classifier
Inside the [HI-EXP](https://github.com/DecioXXIV/xAI-WriterIdentification/tree/main/HI-EXP) you can find three folders: _classifier_NN_, _classifier_GB_ and _classifier_SVM_. 

Each of them contains a checkpoint of the System's pre-trained backbone, the "model.py" and "utils.py" files and, finally, the Python scripts to perform the System's Fine-Tuning. You can inspect the `tuning_experiments.sh` inside the [classifier_NN](https://github.com/DecioXXIV/xAI-WriterIdentification/tree/main/HI-EXP/classifier_NN) folder in order to see how to setup and run the tuning process.

### Explanations with LIME
The [Lime](https://github.com/DecioXXIV/xAI-WriterIdentification/tree/main/HI-EXP/Lime) folder contains:
* Code to compute the explanations for a fine-tuned model: inspect the `explain_cedar.sh` and the `explain_cvl.sh` Shell scripts to see how to setup and run the explanation process
* Code to employ the computed explanations with the aim of create a perturbed version of the employed instances: inspect the mask_instances.sh Shell script to see how to setup and run the instance-masking-from-computed-explanations process
