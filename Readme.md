This repository is built upon https://github.com/zhang-guangyi/t-udeepsc.

## Usage
### Clone
Clone this repository and enter the directory using the commands below:
```bash
git clone https://github.com/zhang-guangyi/t-udeepsc.git
cd t-udeepsc/
```

### Requirements
`Python 3.8.5` is recommended.

Install the required packages with:
```bash
pip install -r requirements.txt (Not provided yet)
```
If you're having issues with installing PyTorch compatible with your CUDA version, we strongly recommend related documentation page](https://pytorch.org/get-started/previous-versions/).


In our work, we use the bert model to initialize the text encoder, the pretrained weights should be placed at ./pretrain_models. The weights can be downloaded in the [huggingface websites](https://huggingface.co/prajjwal1/bert-small).

## Dataset Preparation
### CIFAR10
Use the torchvision, the datasets will be dowmloaded automatically. Then, place the dataset in path ./data/cifar

### MOSEI and MOSI
Download the CMU-MOSI and CMU-MOSEI dataset from Google [Drive](https://drive.google.com/drive/folders/1IBwWNH0XjPnZWaAlP1U2tIJH6Rb3noMI?usp=sharing) and place the contents inside ```./data/msadata``` folder. Note that these are (pre-computed splits).   

### SST2
This dataset is used for text sentiment analysis and text reconstruction. As we use pytreebank in our implementation, the SST2 dataset will also be downloaded automatically. The dataset will be placed at the .cache folder, you can also move it to your required place.

### VQAv2
We use the image features are extracted using the bottom-up-attention strategy, with each image being represented as an 2048-D features. The features for each image are stored in a `.npz` file. You can prepare the visual features by yourself or download the extracted features from [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EsfBlbmK1QZFhCOFpr4c5HUBzUV0aH2h1McnPG1jWAxytQ?e=2BZl8O) or [BaiduYun](https://pan.baidu.com/s/1C7jIWgM3hFPv-YXJexItgw#list/path=%2F). The downloaded files contains three files: **train2014.tar.gz, val2014.tar.gz, and test2015.tar.gz**, corresponding to the features of the train/val/test images for *VQA-v2*, respectively. You should place them as follows:

```angular2html
|-- ./data/vqa_datasets
	|-- coco_extract
	|  |-- train2014.tar.gz
	|  |-- val2014.tar.gz
	|  |-- test2015.tar.gz
```

Besides, we use the VQA samples from the [visual genome dataset](http://visualgenome.org/) to expand the training samples. The processed vg questions and annotations files can be found in [OneDrive](https://awma1-my.sharepoint.com/:f:/g/personal/yuz_l0_tn/EmVHVeGdck1IifPczGmXoaMBFiSvsegA6tf_PqxL3HXclw) or [BaiduYun](https://pan.baidu.com/s/1QCOtSxJGQA01DnhUg7FFtQ#list/path=%2F), and place them as follow:


```angular2html
|-- ./data/vqa_datasets
	|-- vqa
	|  |-- VG_questions.json
	|  |-- VG_annotations.json
```


Then, you can run the following script to setup all the needed configurations for the experiments.
```bash
$ bash vqa_setup.sh
```

Running the script will: 

1. Download the QA files for [VQA-v2](https://visualqa.org/download.html).
2. Unzip the bottom-up features

Finally, the `./data/vqa_datasets` folders will have the following structure:

```angular2html
|-- ./data/vqa_datasets
	|-- coco_extract
	|  |-- train2014
	|  |  |-- COCO_train2014_...jpg.npz
	|  |  |-- ...
	|  |-- val2014
	|  |  |-- COCO_val2014_...jpg.npz
	|  |  |-- ...
	|  |-- test2015
	|  |  |-- COCO_test2015_...jpg.npz
	|  |  |-- ...
	|-- vqa
	|  |-- v2_OpenEnded_mscoco_train2014_questions.json
	|  |-- v2_OpenEnded_mscoco_val2014_questions.json
	|  |-- v2_OpenEnded_mscoco_test2015_questions.json
	|  |-- v2_OpenEnded_mscoco_test-dev2015_questions.json
	|  |-- v2_mscoco_train2014_annotations.json
	|  |-- v2_mscoco_val2014_annotations.json
	|  |-- VG_questions.json
	|  |-- VG_annotations.json

```
