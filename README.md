# Mumpy: Multilateral Temporal-view Pyramid Transformer for Video Inpainting Detection

![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch) ![last commit](https://img.shields.io/github/last-commit/yuxiaoxiangyong/Mumpy)![GitHub Repo stars](https://img.shields.io/github/stars/yuxiaoxiangyong/Mumpy)[![Contact me!](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://GitHub.com/yuxiaoxiangyong)

This repo contains an official PyTorch implementation of our paper: [Multilateral Temporal-view Pyramid Transformer for Video Inpainting Detection.](https://arxiv.org/abs/2404.11054)

## ⏰Update

- **2024-12-04:** Trainging and Test code is uploaded.
- **2024-07-23:** DVI-VI/OP/CP is uploaded.
- **2024-07-21:** 📢 Mumpy is accepted by **BMVC2024**!
- **2024-05-08**: The repo is created.

## ⏳Todo

- [ ] Provide the generation code for YTVI.
- [x] The pre-trained weights will be uploaded soon.
- [x] Make model and code available.
- [x] Make user guideline available.
- [x] Provide ~~YTVI~~, ~~DVI~~ and ~~FVI~~ dataset.
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
- run ./scripts/train_davis.sh

#### Test

Mumpy is tested on DVI, FVI and YTVI. Taking testing on DVI as an example.

- run `./scripts/test.sh`， as you can change `model_name` and `test_epoch`.
- When testing with different datasets, you only need to change the dataset parameter. For example, DVI corresponds to `davis`  and YTVI corresponds to`youtubevos` All the commands have already been provided in the file.
- run `./scripts/measure.sh`, the `input` means generated localization result and `mask_dir` means corresponding ground truth.

## 📑Dataset

#### Davis Inpainting Dataset (DVI)

Since the size of the generated image by inpainting methods can influence the richness of the provided  information, for fair comparision, we generated 224x224 inpainted images on OP and CP. However, as VI only supports 256x256 and 512x512 images, we resized them accordingly. All the experiments on [DVI](https://drive.google.com/file/d/1bEtMe4lGwKhIjT9CYEfCyBohxU-DrGOj/view?usp=drive_link) follow the above principle.

#### YouTube-VOS Video Inpainting dataset (YTVI)

- YTVI is built upon Youtube-vos 2018, which contains 3471 videos with 5945 object instances in its training set. Since only the training set of this dataset is fully annotated, we use it to construct YTVI.
- Specifically, with the goal of further improving the comprehensiveness, we adopt many more recent video inpainting methods on this dataset, including EG2 [1], FF [2], PP [3], and ISVI [4], together with VI, OP and CP. These inpainting methods are applied to the object regions annotated by ground truth masks.
- The file `./configs/youtube/youtubevos_2018.yaml` contains the videos selected for YTVI. Simply follow the instructions for each method to generate the corresponding inpainting videos, and you will obtain the YTVI dataset.

[1] **CVPR 2022** [Towards an end-to-end framework for flow-guided video inpainting.](https://github.com/MCG-NKU/E2FGVI)

[2] **ICCV 2021** [Fuseformer: Fusing fine-grained informationin transformers for video inpainting.](https://github.com/ruiliu-ai/FuseFormer)

[3] **ICCV 2023** [ProPainter:Improving propagation and transformer for video inpainting.](https://github.com/sczhou/ProPainter)

[4] **CVPR 2022** [Inertia-guided flow completion and style fusion for video inpainting.](https://github.com/hitachinsk/ISVI)

#### Free-from Video Inpainting dataset (FVI)

FVI dataset contains 100 test videos that are processed by object removal, and are usually used for demonstrating detection generalization. The download link is [here](https://drive.google.com/file/d/1V7CIZWwt2RV2m0-s5fnkZDtRayRK2Dfv/view?usp=drive_link).

## 💬More Analysis

- What is inpainting detection?
- More details of former research?

## 📧Contact

If you have any questions, please feel free to reach me out at yingzhang@stu.ouc.edu.cn.

## Citation

If you find our repo useful for your research, please consider citing our paper:

```latex
@article{zhang2024multilateral,
  title={Multilateral Temporal-view Pyramid Transformer for Video Inpainting Detection},
  author={Zhang, Ying and Peng, Bo and Zhou, Jiaran and Zhou, Huiyu and Dong, Junyu and Li, Yuezun},
  journal={BMVC},
  year={2024}
}
```

---

```

```
