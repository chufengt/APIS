# Active Pointly-Supervised Instance Segmentation (APIS)
Code for the paper "Active Pointly-Supervised Instance Segmentation", ECCV 2022.

## Preparation

This project is based on the open-source toolbox [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) (as well as [Detectron2](https://github.com/facebookresearch/detectron2)).

Please refer to [INSTALL.md](https://github.com/aim-uofa/AdelaiDet/blob/master/README.md) for installation and dataset (MS-COCO) preparation.

The expected folder structure:

```text
ROOT_PATH
├── AdelaiDet
│   ├── datasets
│   │   ├── coco
│   │   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
├── detectron2
├── APIS
│   ├── scripts
│   ├── src
```

Note that only the newly added source files are included in `APIS/src`.

Modify `$ROOT_PATH`  in `APIS/scripts/prepare.sh` and run:

```bash
# copy source files and prepare random point annotations
sh APIS/scripts/prepare.sh
```

## Usage

We provide the **one-click scripts** to reproduce the main results in the paper, including the results of the `Random Sampling` and `Entropy` strategies mentioned in Tables 1,2,3,6 and Figures 2,3,4.

#### 1. model initialization (P0)

```bash
python APIS/scripts/initialization.py
```

#### 2. random sampling (P1~P9)

```bash
python APIS/scripts/random.py
```

#### 3. active selection with the Entropy metric (P1~P9)

```bash
python APIS/scripts/entropy.py
```
