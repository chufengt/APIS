import os

os.system('cd ./AdelaiDet')

# step=2 is P1 in the paper
for step in range(2, 11):
    pre_step = step - 1

    #########################################################################################################
    # train with the selected points
    #########################################################################################################

    strs = (f'OMP_NUM_THREADS=1 python tools/train_net_point.py \
            --config-file configs/CondInst/APIS/CondInst_fine_tuning_30k.yaml \
            --num-gpus 8 \
            DATASETS.TRAIN "\'coco_2017_train_points_n{step}_random\'," \
            MODEL.WEIGHTS models/model_{pre_step}.pth \
            MODEL.BOXINST.POINT_LOSS_WEIGHT 1.0 \
            OUTPUT_DIR training_dir/random_logs_n{step}')
    os.system(strs)
    os.system(f'cp training_dir/random_logs_n{step}/model_final.pth models/model_{step}.pth')