import os
import torch
import json
from tqdm import trange
import numpy as np
import copy
from pycocotools.coco import COCO

os.system('cd ./AdelaiDet')

# step=2 is P1 in the paper
for step in range(2, 11):
    pre_step = step - 1
    
    #########################################################################################################
    # point selection
    #########################################################################################################

    os.make_dirs('points')
    strs = ('cp adet/modeling/condinst/Entropy/* adet/modeling/condinst')
    os.system(strs)

    strs = (f'OMP_NUM_THREADS=1 python ./tools/train_net_point.py \
            --config-file configs/CondInst/APIS/point_selection_from_CondInst.yaml \
            --num-gpus 8 \
            MODEL.WEIGHTS models/model_{pre_step}.pth \
            OUTPUT_DIR sample_log')
    os.system(strs)

    strs = ('cp adet/modeling/condinst/standard/* adet/modeling/condinst')
    os.system(strs)

    #########################################################################################################
    # generate json files for training
    #########################################################################################################
    
    coco_json = json.load(open('datasets/coco/annotations/instances_train2017.json', 'r'))
    pre_json = json.load(open(f'datasets/coco/annotations/instances_train2017_n{pre_step}.json', 'r'))
    step_random_json = json.load(open(f'datasets/coco/annotations/instances_train2017_n{step}_v1_without_masks.json', 'r'))

    coco = COCO('datasets/coco/annotations/instances_train2017.json')

    img_anns = {}
    for i in range(len(coco_json['annotations'])):
        ann = coco_json['annotations'][i]
        image_id = ann['image_id']
        if image_id not in img_anns:
            img_anns[image_id] = []
        img_anns[image_id].append(i)

    def load_preds(path):
        preds = dict()
        if not os.path.exists(path): return preds
        load_preds = torch.load(path)
        for (ins_idx, points) in load_preds:
            preds[int(ins_idx)] = points
        return preds

    points_for_ann = dict()

    for i_img_id in range(len(img_anns)):
        img_id = list(img_anns.keys())[i_img_id]
        anns = img_anns[img_id]
        
        points_path = 'points/{}.pt'.format(str(img_id).zfill(12))
        image_points = load_preds(points_path)

        for ins_i in range(len(anns)):
            if ins_i in image_points:
                points = image_points[ins_i].tolist()
                points_for_ann[anns[ins_i]] = points
                
    new_json = copy.deepcopy(pre_json)

    not_predicted_cnt = 0
    for i in range(len(coco_json['annotations'])):
        if i in points_for_ann:
            ann = coco_json['annotations'][i]
            mask = coco.annToMask(ann)
            points = points_for_ann[i]
            label = mask[points[0],points[1]]
            new_json['annotations'][i]['point_coords'].append([points[1], points[0]])
            new_json['annotations'][i]['point_labels'].append(int(label))
        else:
            new_json['annotations'][i]['point_coords'].append(step_random_json['annotations'][i]['point_coords'][pre_step])
            new_json['annotations'][i]['point_labels'].append(step_random_json['annotations'][i]['point_labels'][pre_step])
            not_predicted_cnt += 1
    print(f'{not_predicted_cnt} instances not predicted.')

    json.dump(new_json, open(f'datasets/coco/annotations/instances_train2017_n{step}.json', 'w'))
    os.system('rm -rf points')

    #########################################################################################################
    # train with the selected points
    #########################################################################################################

    strs = (f'OMP_NUM_THREADS=1 python ./tools/train_net_point.py \
            --config-file configs/CondInst/APIS/CondInst_fine_tuning_30k.yaml \
            --num-gpus 8 \
            DATASETS.TRAIN "\'coco_2017_train_points_n{step}_active\'," \
            MODEL.WEIGHTS models/model_{pre_step}.pth \
            MODEL.BOXINST.POINT_LOSS_WEIGHT 1.0 \
            OUTPUT_DIR training_dir/entropy_logs_n{step}')
    os.system(strs)
    os.system(f'cp training_dir/entropy_logs_n{step}/model_final.pth models/model_{step}.pth')