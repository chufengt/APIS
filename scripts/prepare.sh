#!/bin/bash

export ROOT_PATH = your_root_path # modify
cd $ROOT_PATH

# copy source files
cp APIS/src/train_net_point.py AdelaiDet/tools/
cp APIS/src/register_point_annotations.py detectron2/projects/PointSup/point_sup/
cp APIS/src/condinst/* AdelaiDet/adet/modeling/condinst/
mkdir -p AdelaiDet/configs/CondInst/APIS
cp APIS/src/configs/* AdelaiDet/configs/CondInst/APIS/

cd detectron2 && export DETECTRON2_DATASETS=$ROOT_PATH/AdelaiDet/datasets && cd ..

# generate random points (see PointSup for details)
python detectron2/projects/PointSup/tools/prepare_coco_point_annotations_without_masks.py 10

mkdir -p AdelaiDet/models