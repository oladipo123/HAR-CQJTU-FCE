<<<<<<< HEAD
# HAR-CQJTU-FCE
A Self Supervised Learning Model for Human Activity Recognition Based on IMU

## File Overview
This project contains following folders and files.
- [`config`](./config) : config json files of models and training hyper-parameters.
- [`dataset`](./dataset) : the scripts for preprocessing four open datasets and a config file of key attributes of those datasets.
- [`classifier.py`](./classifier.py) : run LIMU-GRU that inputs representations learned by LIMU-BERT and output labels for target applications.
- [`config.py`](./config.py) : some helper functions for loading settings.
- [`embedding.py`](./embedding.py) : generates representation or embeddings for raw IMU readings given a pre-trained LIMU-BERT.
- [`models.py`](./models.py) : the implementations of LIMU-BERT, LIMU-GRU, and other baseline models.
- [`plot.py`](./plot.py) : some helper function for plotting IMU sensor data or learned representations.
- [`pretrain.py`](./pretrain.py) : pretrain LIMU-BERT.
- [`statistic.py`](./statistic.py) : some helper functions for evaluation.
- [`train.py`](./train.py) : several helper functions for training models.
- [`utils.py`](./utils.py) : some helper functions for preprocessing data or separating dataset.

## Git Clone
Assume that we use ```~/Repos``` as the working directory. Git clone this repo:
```
$ cd ~/Repos
$ git clone https://github.com/oladipo123/HAR-CQJTU-FCE
```

## Setup
### Option: pip
This repository has be tested for Python 3.7.7/3.8.5 and Pytorch 1.5.1/1.7.1. To install all dependencies, use the following command:
```
$ pip install -r requirements.txt
```

## Dataset
In the [`dataset`](./dataset) folder, we provide four scripts that preprocess the corresponding datasets. Those datasets are widely utilized in the previous studies:
- [HHAR](http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition)
- [UCI](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions)
- [MotionSense](https://github.com/mmalekzadeh/motion-sense)
- [CQJTU-FEC](13529310738@163.com)

Each script has a kernel function which reads the raw IMU data and output preprocessed data and label. You can set the sampling rate and window size (sequence length) in the functiono. It retures two values:
- Data: a numpy array with the shape of (N\*W\*F), N is the number of samples, W is the windows size, and F is the number of features (6 or 9).
- Label: a numpy array with the shape of (N\*W\*L), N is the number of samples, W is the windows size, and L is the number of label types (e.g., activity and user labels). The detailed label information is provided in [`data_config.json`](./dataset/data_config.json).

The two numpy arrays are saved as "data_X_Y.npy" and "label_X_Y.npy" in each dataset folder, where X represents the sampling rate and Y is the window size. 
For example, all data and label are saved as "data_20_120.npy" and "label_20_120.npy" in the settings of LIMU-BERT. And the data and label arrays of HHAR dataset are saved in the _dataset/hhar_ folder.

## Framework
In our framework, there are two phases:
- Self-supervised training phase: train feature model with unlabeled IMU data.
- Supervised training phase: train GRU based on the learned representations.

In implementation, there are three steps to run the codes:
- [`pretrain.py`](./pretrain.py) : pretrain feature model and the decoder.
- [`embedding.py`](./embedding.py) : generates and save representations learned by feature model.
- [`classifier.py`](./classifier.py) : load representations and train a task-specific classifier.

## Usage
[`pretrain.py`](./pretrain.py), [`embedding.py`](./embedding.py), [`classifier.py`](./classifier.py), 
```
usage: xxx.py [-h] [-g GPU] [-f MODEL_FILE] [-t TRAIN_CFG] [-a MASK_CFG]
                   [-l LABEL_INDEX] [-s SAVE_MODEL]
                   model_version {hhar,motion,uci,myowner} {10_100,20_120}

positional arguments:
  model_version         Model config, e.g. v1
  {hhar,motion,uci,shoaib}
                        Dataset name
  {10_100,20_120}       Dataset version

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU     Set specific GPU
  -f MODEL_FILE, --model_file MODEL_FILE
                        Pretrain model file, default: None
  -t TRAIN_CFG, --train_cfg TRAIN_CFG
                        Training config json file path
  -a MASK_CFG, --mask_cfg MASK_CFG
                        Mask strategy json file path, default: config/mask.json
  -l LABEL_INDEX, --label_index LABEL_INDEX
                        Label Index setting the task, default: -1
  -s SAVE_MODEL, --save_model SAVE_MODEL
                        The saved model name, default: 'model'
```
### [`pretrain.py`](./pretrain.py)
Example:
```
python pretrain.py v1 uci 20_120 -s limu_v1 
```
For this command, we will train a feature model, whose settings are defined in the _based_v1_ of [`limu_bert.json`](./config/limu_bert.json),
with the UCI dataset "data_20_120.npy" and "label_20_120.npy". The trained model will be saved as "limu_v1.pt" in the _saved/pretrain_base_uci_20_120_ folder.
The mask and train settings are defined in the [`mask.json`](./config/mask.json) and [`pretrain.json`](./config/pretrain.json), respectively.

In the main function of [`pretrain.py`](./pretrain.py), you can set following parameters:
- _training_rate_: float, defines the proportion of unlabeled training data we want to use. The default value is 0.8.
### [`embedding.py`](./embedding.py)
Example:
```
python embedding.py v1 uci 20_120 -f limu_v1
```
For this command, we will load the pretrained feature model from file "limu_v1.pt" in the _saved/pretrain_base_uci_20_120_ folder.
And embedding.py will save the learned representations as "embed_limu_v1_uci_20_120.npy" in the _embed_ folder.

### [`classifier.py`](./classifier.py)
Example:
```
python classifier.py v2 uci 20_120 -f limu_v1 -s limu_gru_v1 -l 0
```
For this command, we will load the embeddings or representations from "embed_limu_v1_uci_20_120.npy" and train the GRU classifier
, whose settings are defined in the _gru_v2_ of [`classifier.json`](./config/classifier.json). 
The trained GRU classifier will be saved as "limu_gru_v1.pt" in the _saved/classifier_base_uci_20_120_ folder.
The target task corresponds to the first label in "label_20_120.npy" of UCI dataset, which is a human activity recognition task defined in [`data_config.json`](./dataset/data_config.json). The train settings are defined in the [`train.json`](./config/train.json).

In the main function of [`classifier.py`](./classifier.py), you can set following parameters:
- _training_rate_: float, defines the proportion of unlabeled data that the pretrained LIMU-BERT uses. The default value is 0.8. 
Note that this value must be equal to the _training_rate_ in the [`pretrain.py`](./pretrain.py).
- _label_rate_: float, defines the proportion of labeled data to the unlabeled training data that the GRU classifier uses.
- _balance_: bool, defines whether it should use balanced labeled samples among the multiple classes. Default: True.
- _method_: str, defines the classifier type from {gru, lstm, cnn1, cnn2, attn}. Default: gru.

Note: The myowner dataset in the code is the self collected CQJTU-FCE dataset.

## Contact
13529310738@163.com



