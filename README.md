# xAI-WriterIdentification 🧠🤖📄
This Repository stores the code to:
- Fine-Tune a CNN Model (_ResNet18_) in order to solve the Handwriting Identification task on modern manuscript documents
- Generate the Explanations (through [_LIME_](https://github.com/marcotcr/lime)) for the results provided by the classifier

### Tip!
The Pipeline is entirely executable using a single Shell script, to be created in this directory.
Each Step is executed by a Python script, to be launched using the syntax below:

`python -m folder_name/script_name.py`

*Be Careful!* All the Python scripts require parameters to be passed. Use the syntax below:

`python -m folder_name/script_name.py -param_1 value_1 -param_2 value_2 -... -param_n value_n`

## Guidelines 👨🏻‍🏫
Here you have a brief guide to execute the whole process.

### 1. Baseline Experiment Setup 🔧
Choose a _test_id_ and execute `log\setup_exp.py` to create a new JSON file which will store a collection of useful metadata for the current experiment.

### 2. Dataset Creation 📄📄📄
Given the _test_id_ and its metadata, execute `datasets\prepare_pages.py` to create the Train and Test Instances.

### 3. Model Fine-Tuning 🛠️🧠🤖
Given the _test_id_ and its metadata, execute `classifiers\classifier_NN\fine_tune.py` to perform the Model Fine-Tuning using the Dataset generated in the Step 2.

### 4. Explanations Computation 🧠🤖📚
Given the _test_id_ and its _Fine-Tuned Model_ (Step 3), execute `xai\explain.py` to compute the Explanations both for Training and Test Instances.

### 5. Instance Masking 📄🤬
Given the _test_id_, its _Fine-Tuned Model_ (Step 3), and the Explanations (Step 4), execute `xai/mask_instances.py` to generate the Train Instances for the Re-Trained Experiment.

### 6. Re-Trained Experiment Setup 🔧🔄
Choose a _retrained_test_id_ and execute again `log\setup_exp.py` to create a new JSON file which will store a collection of useful metadata for the current (re-trained) experiment.

### 7. Dataset re-Creation 📄📄📄🔄
Given the _retrained_test_id_ and its metadata, execute `datasets\prepare_pages.py` to create the Train and Test Instances. The Train Instances will be created using the masked Instances generated starting from the Baseline Experiment Explanations (Step 4).

### 8. Model Fine-Tuning (again!) 🛠️🧠🤖🔄
Given the _retrained_test_id_ and its metadata, execute `classifiers\classifier_NN\fine_tune.py` to perform the Model Fine-Tuning using the Dataset generated in the Step 7.

### 9. Re-Explaining 🧠🤖📚🔄
Given the _retrained_test_id_ and its _Fine-Tuned Model_ (Step 8), execute `xai\explain.py` to compute the Explanations only for the Test Instances.

### 10. Outcome Comparisons (Baseline VS Re-Trained) 🧠🤖🆚🧠🤖🔄
Given a _test_id_ and a _retrained_test_id_, execute `bvr_comparisons\compare.py` in order to produce the Comparation Reports (_Confidence_ and _Explanations_).
