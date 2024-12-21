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

The objective of this project is to develop an automatic fingerprint classification system into 8 classes: Plain Arch, Tented Arch, Ulnar Loop, Radial Loop, Double Loop Whorl, Plain Whorl, Central Pocket Loop Whorl and Accidental Whorl. To achieve this goal, multiple proposals have been made, using advanced deep learning and machine learning techniques. For training the model, the HRF DBII dataset provided by the Hong Kong Polytechnic University was used.

PolyU HRF DBII consists of a small high resolution fingerprint (HRF) dataset. The images of the same finger were collected in two sessions separated by about two weeks. Each image is named as “ID_S_X”, where “ID” represents the identity of the person, “S” represents the session of the captured image and “X” represents the image number of each session. DBII contains 1.480 images from 148 fingers.

To ensure that the performance during the testing phase was as objective as possible, after classifying the images into the 8 classes, they were divided by user, ensuring that only fingerprints from users not used during the training phase were used during testing.

In terms of networks, within deep learning, a residual convolutional neural network (ResNet) and a convolutional neural network were used, while in the field of machine learning, the well-known SVM algorithm was used. The performance of the networks may vary between training sessions due to random initialization.

Before you can proceed with the classification of fingerprints, it is necessary to train the neural network. Two scripts are provided for this purpose:
- `train.py` for training the models.
- `test.py` for testing the models.

The dataset is managed by the `manage_dataset.py` class, while the two custom models are defined in the `resnet_model.py` and `CNN_model.py`. A standard machine learning model, specifically the `SVM`, is also used.

> [!IMPORTANT]  
> To reproduce the project, you need to run the following commands to include the configuration file:
>
>\>>> python -u train.py -c config/base_config.yaml
>
> Replace "train.py" with "test.py" to evaluate performance on the test dataset after training the models

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

After training and testing the neural networks, you can proceed with the automatic classification of images. Within the repository, there is a folder named `Fingerprint_to_classify`. Simply move the images you want to classify into this folder and execute the command:

```bash
python classify.py
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
- `sklearn` for machine learning algorithms
- `numpy` for numerical computing
- `matplotlib` for data visualization
- `pandas` for data manipulation
- `shutil` for file and directory manipulation
- `seaborn` for confusion matrix
- `transformers` for embeddings
- `logging` to hide warnings
- `joblib` to save and load models
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
|   ├── extract_representations.html
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
├── extract_representations/
|   ├── vision_embeddings.py
|
|
├── model_classes/
|   ├── resnet_model.py
|   ├── CNN_model.py
|
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
- `data_classes/` contains the classe for managing the dataset.
- `docs/` contains project documentation.
- `extract_representations/` contains classes to manage embeddings for the SVM model.
- `model_classes/` contains the classes for the models design.
- `classify.py` is the script for classification.
- `LICENCE/` contains the license to use the project.
- `prepare.sh` is a script for setting up the environment installing the requirements.
- `README.md` is the file you are currently reading.
- `requirements.txt` contains the list of dependencies for the project.
- `utils.py` is the script that evaluates the performance metrics, print them and contain other useful functions.
- `train.py` is the script for training the models.
- `test.py` is the script for testing the models.


> [!IMPORTANT]  
> After executing the script train.py to the entire local running directory, additional folders will be generated: checkpoints, dataset and graph

- `checkpoints/` this folder could contain two files in .pt format that represent the saving of the weights of the best models found during training. It also could contain .joblib and .pkl files used by SVM model. 
- `dataset/` this folder contains the dataset already divided into train, val and test.
- `graph/` this folder contains the graphs that originated from the manipulation of the dataset, the training, comparison and testing phases.

### Automatic classification

> [!IMPORTANT]  
> The project was provided with two empty folders, respectively named 'Fingerprint_to_classify' and 'Fingerprint_classified,' which are used for the automatic classification of images. <br>Specifically:
- `Fingerprint_to_classify/` represents the folder where you should place the fingerprint images you want to classify using the trained neural network.
- `Fingerprint_classified/` represents the folder where the images are moved once they have been classified. At the end of the classification process, a subfolder for each necessary class is created, ensuring that the corresponding image is restored in that class.

> [!CAUTION]
>  If the folders are accidentally removed, they will be automatically recreated during the next classification attempt. You will need to reinsert the images you want to classify into the 'Fingerprint_to_classify' folder.
---

### Use of base_config.yaml file
Through the use of the base_config.yaml file, it is possible to modify the configuration parameters related to the training of the model. Here are just a few of the most common examples:

- `create_dataset_graph` you can choose to create [TRUE] or not [FALSE] dataset graph.
- `view_dataset_graph` you can choose to view [TRUE] or not [FALSE] dataset graph during execution.
- `model_to_train` you can choose the model you want to train.
- `model_to_test` you can choose the model you want to test if they have been correct trained yet.
- `create_model_graph` you can choose to create [TRUE] or not [FALSE] models graph.
- `view_model_graph` you can choose to view [TRUE] or not [FALSE] models graph.
- `create_compare_graph` you can choose to create [TRUE] or not [FALSE] compare graph.
- `view_compare_graph` you can choose to view [TRUE] or not [FALSE] compare graph.
- `metric_plotted_during_traininig` you can select the only performance metrics you prefer to view.
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
