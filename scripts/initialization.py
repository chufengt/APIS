import os
import json
import copy

os.system('cd ./AdelaiDet')

# generate random points for p1~p9
# p(i) should be the subset of p(j) if i<j
for n in range(1, 10):
    json_file = json.load(open('datasets/coco/annotations/instances_train2017_n10_v1_without_masks.json', 'r'))
    for ann in json_file['annotations']:
        ann['point_coords'] = ann['point_coords'][0:n]
        ann['point_labels'] = ann['point_labels'][0:n]
    json.dump(json_file, open(f'datasets/coco/annotations/instances_train2017_n{n}_v1_without_masks.json', 'w'))

# p0 is randomly selected
os.system('cp datasets/coco/annotations/instances_train2017_n1_v1_without_masks.json\
			datasets/coco/annotations/instances_train2017_n1.json')

strs = ('OMP_NUM_THREADS=1 python tools/train_net_point.py \
		--config-file configs/CondInst/APIS//CondInst_P0_1x.yaml \
		--num-gpus 8 \
		MODEL.BOXINST.POINT_LOSS_WEIGHT 0.1 \
		OUTPUT_DIR training_dir/P0')
os.system(strs)

os.system('cp training_dir/P0/model_final.pth models/model_1.pth')