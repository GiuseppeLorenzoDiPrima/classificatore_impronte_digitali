## Project for "Autenticazione biometrica per la sicurezza dei sistemi informatici"
### PolyU HRF DBII Datasets

This is a guide to assist in reading the “PolyU HRF DBII” project related to the exam of "Autenticazione biometrica per la sicurezza dei sistemi informatici" for the "Kore" University of Enna.

| | |
| --- | --- |
| **Description** | Project PolyU HRF DBII |
| **Authors** | Di Prima Giuseppe Lorenzo |
| **Course** | [Autenticazione biometrica per la sicurezza dei sistemi informatici @ UniKore](https://unikore.it/cdl-insegnamento/autenticazione-biometrica-per-la-sicurezza-dei-sistemi-informatici-ing-inf-05-9-cfu-ingegneria-dellintelligenza-artificiale-e-della-sicurezza-informatica-pds-2024-2025-ii-anno/) |
| **License** | [MIT](https://opensource.org/licenses/MIT) |

---

### Table of Contents

- [Project PolyU HRF DBII](#PolyU-HRF-DBII-Datasets)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Code structure](#code-structure)
  - [Automatic classification](#automatic-classification)
  - [Use of base_config.yaml file](#Use-of-base_config.yaml-file)
  - [Documentation](#Documentation)
  - [License](#license)

---

### Introduction

The objective of this project is to develop an automatic fingerprint classification system into 8 classes: Plain Arch, Tented Arch, Ulnar Loop, Radial Loop, Double Loop Whorl, Plain Whorl, Central Pocket Loop Whorl and Accidental Whorl. To achieve this goal, a proposal using an advanced deep learning techniques has been made. To train the model, the HRF DBII dataset provided by the Hong Kong Polytechnic University was used.

PolyU HRF DBII consists of a small high resolution fingerprint (HRF) dataset. The images of the same finger were collected in two sessions separated by about two weeks. Each image is named as “ID_S_X”, where “ID” represents the identity of the person, “S” represents the session of the captured image and “X” represents the image number of each session. DBII contains 1.480 images from 148 fingers.

To ensure that the performance during the testing phase was as objective as possible, after classifying the images into the 8 classes, they were divided by user, ensuring that only fingerprints from users not used during the training phase were used during testing.

In terms of model, a residual convolutional neural network (ResNet) was used. The performance of the model may vary between training sessions due to random initialization.

Before you can proceed with the classification of fingerprints, it is necessary to train the neural network. Two scripts are provided for this purpose:
- `train.py` for training the model.
- `test.py` for testing the model.

The dataset is managed by the `manage_dataset.py` class, while the custom model is defined in the `resnet_model.py`.

> [!IMPORTANT]  
> To reproduce the project, you need to run the following commands to include the configuration file:
>
>\>>> python -u train.py -c config/base_config.yaml
>
> Replace "train.py" with "test.py" to evaluate performance on the test dataset after training the model

The main idea is that, the project can be reproduced by running the following commands:

```bash
git clone https://github.com/GiuseppeLorenzoDiPrima/classificatore_impronte_digitali.git
cd classificatore_impronte_digitali
bash prepare.sh
python train.py
python test.py
```

> [!CAUTION]
>  You must have a version of git installed that is compatible with your operating system to perform the git clone operation.

After training and testing the neural network, you can proceed with the automatic classification of images. Within the repository, there is a folder named `Fingerprint_to_classify`. Simply move the images you want to classify into this folder and execute the command:

```bash
python classify.py -c config/base_config.yaml
```
---
Upon completion, the classification results will be visible inside the `Fingerprint_classified` folder. 

### Requirements

The project is based on **Python 3.12.3** - one of the latest versions of Python at the time of writing.

The `prepare.sh` script is used to install the requirements for the project and to set up the environment (e.g. download the dataset)

- The requirements are listed in the `requirements.txt` file and can be installed using `pip install -r requirements.txt`.

This project is based on the following libraries:
- `torch` for PyTorch.
- `torchvision` for PyTorch vision.
- `yaml_config_override` for configuration management.
- `addict` for configuration management.
- `tqdm` for progress bars.
- `scikit-learn` as a newer version of sklearn
- `numpy` for numerical computing
- `matplotlib` for data visualization
- `pandas` for data manipulation
- `shutil` for file and directory manipulation
- `seaborn` for confusion matrix
---

### Code structure

The code is structured as follows:

```
main_repository/
|
├── config/
|   ├── base_config.yaml
|
|
├── data_classes/
|   ├── manage_dataset.py
|
|
├── docs
|   ├── _modules/..
|   ├── _sources/..
|   ├── _static/..
|   ├── .buildinfo
|   ├── data_classes.html
|   ├── genindex.html
|   ├── index.html
|   ├── model_classes.html
|   ├── modules.html
|   ├── objects.inv
|   ├── py-modindex.html
|   ├── search.html
|   ├── searchindex.js
|   ├── test.html
|   ├── train.html
|   ├── utils.html
|
|
├── model_classes/
|   ├── resnet_model.py
|
├── classify.py
|
├── LICENCE
├── prepare.sh
├── README.md
├── requirememts.txt
├── utils.py
|
├── train.py
├── test.py
```

- `config/` contains the configuration parameters.
- `data_classes/` contains the class for managing the dataset.
- `docs/` contains project documentation.
- `model_classes/` contains the classes for the model design.
- `classify.py` is the script for classification.
- `LICENCE/` contains the license to use the project.
- `prepare.sh` is a script for setting up the environment installing the requirements.
- `README.md` is the file you are currently reading.
- `requirements.txt` contains the list of dependencies for the project.
- `utils.py` is the script that evaluates the performance metrics, print them and contain other useful functions.
- `train.py` is the script for training the model.
- `test.py` is the script for testing the model.


> [!IMPORTANT]  
> After executing the script train.py to the entire local running directory, additional folders will be generated: checkpoints, dataset and graph

- `checkpoints/` this folder could contain a file in .pt format that represent the saving of the weights of the best model found during training. 
- `dataset/` this folder contains the dataset already divided into train, val and test.
- `graph/` this folder contains the graphs that originated from the manipulation of the dataset, the training and the testing phases.

### Automatic classification

> [!IMPORTANT]  
> The project was provided with two empty folders, respectively named 'Fingerprint_to_classify' and 'Fingerprint_classified,' which are used for the automatic classification of images. <br>Specifically:
- `Fingerprint_to_classify/` represents the folder where you should place the fingerprint images you want to classify using the trained neural residual network.
- `Fingerprint_classified/` represents the folder where the images are moved once they have been classified. At the end of the classification process, a subfolder for each necessary class is created, ensuring that the corresponding image is restored in that class.

> [!CAUTION]
>  If the "Fingerprint_to_classify" folder is accidentally removed, it will be automatically recreated during the next classification attempt. You will need to reinsert the images you want to classify into the new 'Fingerprint_to_classify' folder.
---

### Use of base_config.yaml file
Through the use of the base_config.yaml file, it is possible to modify the configuration parameters related to the training of the model. Here are just a few of the most common examples:

- `create_dataset_graph` you can choose to create [TRUE] or not [FALSE] dataset graph.
- `view_dataset_graph` you can choose to view [TRUE] or not [FALSE] dataset graph during execution.
- `create_model_graph` you can choose to create [TRUE] or not [FALSE] model graph.
- `view_model_graph` you can choose to view [TRUE] or not [FALSE] model graph.
- `metric_plotted_during_traininig` you can select the only performance metrics you prefer to plot.
- `epochs` you can choose number of epochs based on your device computing capacity.
- `early_stopping_metric` you can choose the metric against which you want to check for performance improvement according to early stopping.
- `earling_stopping_max_no_valid_epochs` you can choose the max value of epochs that do not produce performance improvement.
- `evaluation_metric` you can choose the performace metric by which you want to evaluate your performance.
- `many other...`
---

### Documentation
Inside the docs folder you can open the index.html file to access the documentation of this project. The file will be opened through the default browser set by the user.

---

### License
This project is licensed under the terms of the MIT license. You can find the full license in the `LICENSE` file.
