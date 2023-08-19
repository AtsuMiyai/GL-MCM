# Zero-Shot In-Distribution Detection in Multi-Object Settings Using Vision-Language Foundation Models 

This codebase provides a Pytorch implementation for the paper "Zero-Shot In-Distribution Detection in Multi-Object Settings Using Vision-Language Foundation Models".

## Novel Task: Zero-Shot ID detection
![ID_detection](readme_figs/ID_detection.png)

### Abstract
Extracting in-distribution (ID) images from noisy images scraped from the Internet is an important preprocessing for constructing datasets, which has traditionally been done manually. Automating this preprocessing with deep learning techniques presents two key challenges. First, images should be collected using only the name of the ID class without training on the ID data. Second, as we can see why COCO was created, it is crucial to identify images containing not only ID objects but also both ID and OOD objects as ID images to create robust recognizers. In this paper, we propose a novel problem setting called zero-shot in-distribution (ID) detection, where we identify images containing ID objects as ID images, even if they contain OOD objects, and images lacking ID objects as out-of-distribution (OOD) images without any training. To solve this problem, we present a simple and effective approach, Global-Local Maximum Concept Matching (GL-MCM), based on both global and local visual-text alignments of CLIP features. Extensive experiments demonstrate that GL-MCM outperforms comparison methods on both multi-object datasets and single-object ImageNet benchmarks.

### Illustration
#### Global-Local Maximum Concept Matching (GL-MCM)
![Arch_figure](readme_figs/framework.png)



# Set up

## Required Packages
We have done the codes with a single Nvidia A100 (or V100) GPU.
We follow the environment in [MCM](https://github.com/deeplearning-wisc/MCM).

Our experiments are conducted with Python 3.8 and Pytorch 1.10.
Besides, the following commonly used packages are required to be installed:
```bash
$ pip install ftfy regex tqdm scipy matplotlib seaborn tqdm scikit-learn
```

# Data Preparation
## In-distribution Datasets
We use following datasets as ID:
- `COCO_single`: used in Table1, Table3, and Table4 (Each Image in COCO_single has single-class ID objects and one or more OOD objects)
- `VOC_single`: used in Table1 and Table3 (Each Image in VOC_single has single-class ID objects and one or more OOD objects)
- `ImageNet`: used in Table2
- `COCO_multi`: used in supplementary (Each Image in COCO_multi has multi-class ID objects and one or more OOD objects)    

We provide our curated ID and OOD datasets via [this url](https://drive.google.com/file/d/1he4jKi2BfyGT6rkcbFYlez7PbLMXTBMR/view?usp=sharing).   
For ImageNet-1k, we use the validation partion of the [official provided dataset](https://image-net.org/challenges/LSVRC/2012/index.php#).    
After downloads and, please set the datasetes to `./datasets` 
<!-- For other datasets, please download them via [this url](https://drive.google.com/file/d/1Wn5zGQQzadsvza86shO_ydpyCu5-k2eN/view?usp=share_link).         
After downloads, please set the datasetes to `./datasets`     -->

## Out-of-Distribution Datasets
We use the large-scale OOD datasets [iNaturalist](https://arxiv.org/abs/1707.06642), [SUN](https://vision.princeton.edu/projects/2010/SUN/), [Places](https://arxiv.org/abs/1610.02055), and [Texture](https://arxiv.org/abs/1311.3618) curated by [Huang et al. 2021](https://arxiv.org/abs/2105.01879). We follow instruction from the this [repository](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) to download the subsampled datasets. For ImageNet-22K, we use this [url](https://drive.google.com/drive/folders/1BGMRQz3eB_npaGD46HC6K_uzt105HPRy) in this [repository](https://github.com/deeplearning-wisc/multi-label-ood) curated by [Wang et al. 2021](https://arxiv.org/abs/2109.14162)

In addition, we also use ood_coco and ood_voc in [this url](https://drive.google.com/file/d/1he4jKi2BfyGT6rkcbFYlez7PbLMXTBMR/view?usp=sharing). 

The overall file structure is as follows:

```
GL-MCM
|-- datasets
    |-- ImageNet
    |-- ID_COCO_single
    |-- ID_VOC_single
    |-- ID_COCO_multi
    |-- OOD_COCO
    |-- OOD_VOC
    |-- iNaturalist
    |-- SUN
    |-- Places
    |-- Texture
    |-- ImageNet-22K
    ...
```

# Quick Start

The main script for evaluating OOD detection performance is `eval_id_detection.py`. Here are the list of arguments:

- `--name`: A unique ID for the experiment, can be any string
- `--score`: The OOD detection score, which accepts any of the following:
  - `MCM`: [Maximum Concept Matching score](https://arxiv.org/pdf/2211.13445.pdf)
  - `L-MCM`: Local MCM (ours)
  - `GL-MCM`: Global-Local MCM (ours)
- `--seed`: A random seed for the experiments
- `--gpu`: The index of the GPU to use. For example `--gpu=0`
- `--in_dataset`: The in-distribution dataset
  - Accepts: `ImageNet`, `COCO_single`, `COCO_multi`, `VOC_single`
- `-b`, `--batch_size`: Mini-batch size
- `--CLIP_ckpt`: Specifies the pretrained CLIP encoder to use
  - Accepts: `RN50`, `RN101`, `ViT-B/16`.
- `--num_ood_sumple`: the number of OOD samples

The OOD detection results will be generated and stored in  `results/in_dataset/score/CLIP_ckpt_name/`. 

We provide bash scripts:

```sh
sh scripts/eval_coco_single.sh
```

# Acknowledgement 
This code is based on the implementations of [MCM](https://github.com/deeplearning-wisc/MCM)


# Citaiton
If you find our work interesting or use our code/models, please cite:
```bibtex
@article{miyai2023zero,
  title={Zero-Shot In-Distribution Detection in Multi-Object Settings Using Vision-Language Foundation Models},
  author={Miyai, Atsuyuki and Yu, Qing and Irie, Go and Aizawa, Kiyoharu},
  journal={arXiv preprint arXiv:2304.04521},
  year={2023}
}