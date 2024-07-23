# Mumpy: Multilateral Temporal-view Pyramid Transformer for Video Inpainting Detection

![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch) ![last commit](https://img.shields.io/github/last-commit/yuxiaoxiangyong/Mumpy)![GitHub Repo stars](https://img.shields.io/github/stars/yuxiaoxiangyong/Mumpy)[![Contact me!](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://GitHub.com/yuxiaoxiangyong) 

This repo contains an official PyTorch implementation of our paper: [Multilateral Temporal-view Pyramid Transformer for Video Inpainting Detection.](https://arxiv.org/abs/2404.11054)

## ⏰Update

- **2024-07-23:** DVI-VI/OP/CP is uploaded. 
- **2024-07-21:** 📢 Mumpy is accepted by **BMVC2024**! 
- **2024-05-08**: The repo is created. 


## ⏳Todo 
- [ ] Make model and code available. 
- [ ] Make user guideline available. 
- [ ] Provide YTVI, ~~DVI~~ and FVI dataset. 
- [ ] More analysis on why multiple-pyramid decoder.

## 🌏Overview

<img src=".\images\overview.png" style="zoom: 25%;" />

##  🌄Get Started



##  📑Dataset

1. Since the size of the generated image by inpainting methods can influence the richness of the provided information, for fair comparision, we generated 224x224 inpainted images on OP and CP. However, as VI only supports 256x256 and 512x512 images, we resized them accordingly. 

​		[DVI-VI/OP/CP]( https://drive.google.com/file/d/1bEtMe4lGwKhIjT9CYEfCyBohxU-DrGOj/view?usp=drive_link)

2. YTVI.
3. FVI. 

##  💬More Analysis

1. What is inpainting detection? And What is the solvement of former research?


##  📧Contact
If you have any questions, please feel free to reach me out at yingzhang@stu.ouc.edu.cn.

##  Citation
If you find our repo useful for your research, please consider citing our paper:
```latex
@article{zhang2024multilateral,
  title={Multilateral Temporal-view Pyramid Transformer for Video Inpainting Detection},
  author={Zhang, Ying and Peng, Bo and Zhou, Jiaran and Zhou, Huiyu and Dong, Junyu and Li, Yuezun},
  journal={arXiv preprint arXiv:2404.11054},
  year={2024}
}
```

****