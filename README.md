# Mumpy: Multilateral Temporal-view Pyramid Transformer for Video Inpainting Detection

![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch) ![last commit](https://img.shields.io/github/last-commit/yuxiaoxiangyong/Mumpy)![GitHub Repo stars](https://img.shields.io/github/stars/yuxiaoxiangyong/Mumpy)[![Contact me!](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://GitHub.com/yuxiaoxiangyong)

This repo contains an official PyTorch implementation of our paper: [Multilateral Temporal-view Pyramid Transformer for Video Inpainting Detection.](https://arxiv.org/abs/2404.11054)

## ⏰Update
- **2024-12-04:** Trainging and Test code is uploaded.
- **2024-07-23:** DVI-VI/OP/CP is uploaded.
- **2024-07-21:** 📢 Mumpy is accepted by **BMVC2024**!
- **2024-05-08**: The repo is created.

## ⏳Todo
- [x] The pre-trained weights will be uploaded soon.
- [x] Make model and code available.
- [x] Make user guideline available.
- [ ] Provide YTVI, ~~DVI~~ and FVI dataset.
- [ ] More analysis on why multiple-pyramid decoder.

## 🌏Overview

<img src=".\images\overview.png" style="zoom: 25%;" />

## 🌄Get Started


### Quick Start of Source Code

```
./Mumpy
    ├── scripts (train or test scripts folder)
    │      ├── train_davis.sh (script of training on DVI)
    │      ├── train_youtube.sh(script of trainging on YTVI)
    │      ├── measure.sh(script of measuring F1 and IoU)
    │      └── test.sh(script of generating localization)
    ├── configs (training or test configuration folder)
    │      ├── davis (folder)
    │      │      ├── config.py (davis inpainting dataset configuration file)
    │      │      ├── db_info.yaml (davis dataset basic information file)	    
    │      ├── youtube (folder)
    │      │      ├── config.py (youtube inpainting dataset configuration file)
    │      │      ├── youtubevos_2018.yaml (youtube 2018 dataset basic information file)
    ├── dataloaders (dataloader folder)
    │      ├── base.py (functional class)
    │      ├── universaldataloader.py
    │      └── universaldataset.py
    ├── models (encoder, decoder, model factory, modules)
    │      ├── encoder (folder)
    │      │      ├── encoder.py (overall encoder structure of mumpy, baseline)
    │      │      ├── multiTemporalViewEncoder.py (implementation of multiTemporalViewEncoder)	    
    │      ├── decoder (folder)
    │      │      ├── decoder.py (multi pyramid decoder)
    │      ├── factory (folder)
    │      │      ├── modelFactory.py (encoder factory) 
    │      ├── modules (folder)
    │      │      ├── blocks.py (vit block) 
    │      │      ├── dct.py  
    │      │      ├── deformableAttention.py  
    │      │      └── swinTransformer.py
    ├── utils(util folder)
    │      ├── optimizer (folder)
    │      │      ├── scheduler.py 
    │      │      ├── factory.py	    
    │      ├── dataset_utils.py 
    │      ├── io_aux.py 
    │      ├── loss.py 
    │      ├── randaugment.py (data augmentation) 
    │      ├── utils.py (load or save model, ....)
    ├── train.py (training files)
    ├── test.py (generate localization results)
    ├── measure.py (localization evaluation)
    ├── weights(imagenet pretrain weight folder)
    ├── results (trained model folder)
    ├── weights (put the pre-trained weights in.)
    └── images(images folder)
```

### Preparation

The code was running on:

* Ubuntu 22.04 LTS, Python 3.9,  CUDA 11.7, GeForce RTX 3090ti

- To create your environment by

```python
conda create --n mumpy python=3.9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

- Go to [mumpy_weights_link](https://drive.google.com/file/d/1gpxMheyI8KfeSgMwye_J8TONuk8A-AIe/view?usp=drive_link) to download the weights from, and then put them in `weights`.

#### Training

Mumpy is trained on DVI or YTVI. Taking training from DVI as an example.

- First replace `__C.PATH.SEQUENCES` , `__C.PATH.SEQUENCES2` and  `__C.PATH.SEQUENCES3` in `config.py` to be the training data path.
- The global variables related to data in the configuration file also need to be modified(e.g., `__C.PATH.ANNOTATIONS`, `__C.PATH.DATA`, `__C.FILES.DB_INFO`).
- The training of Mumpy requires changing the parameters in the mentationed script. You can adjust it as needed.

#### Test

Mumpy is tested on DVI, FVI and YTVI. Taking testing on DVI as an example.

- run `./scripts/test.sh`， as you can change `model_name` and `test_epoch`.
- When testing with different datasets, you only need to change the dataset parameter. For example, DVI corresponds to `davis`  and YTVI corresponds to`youtubevos` All the commands have already been provided in the file.
- run `./scripts/measure.sh`, the `input` means generated localization result and `mask_dir` means corresponding ground truth.


## 📑Dataset

1. Since the size of the generated image by inpainting methods can influence the richness of the provided information, for fair comparision, we generated 224x224 inpainted images on OP and CP. However, as VI only supports 256x256 and 512x512 images, we resized them accordingly. All the experiments on [DVI](https://drive.google.com/file/d/1bEtMe4lGwKhIjT9CYEfCyBohxU-DrGOj/view?usp=drive_link) follow the above principle.
2. YTVI.
3. FVI.

## 💬More Analysis

1. What is inpainting detection? And What is the solvement of former research?

## 📧Contact

If you have any questions, please feel free to reach me out at yingzhang@stu.ouc.edu.cn.

## Citation

If you find our repo useful for your research, please consider citing our paper:

```latex
@article{zhang2024multilateral,
  title={Multilateral Temporal-view Pyramid Transformer for Video Inpainting Detection},
  author={Zhang, Ying and Peng, Bo and Zhou, Jiaran and Zhou, Huiyu and Dong, Junyu and Li, Yuezun},
  journal={arXiv preprint arXiv:2404.11054},
  year={2024}
}
```

---
