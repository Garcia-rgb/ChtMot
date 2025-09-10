cd src
python train.py \
  --task mot \
  --gpus 0 \
  --exp_id mot20_def \
  --load_model '../src/lib/models/ctdet_coco_dla_2x.pth' \
  --data_cfg '../src/lib/cfg/mot20.json' \
  --batch_size 12 \
  --num_epochs 20 \
  --lr_step 10
cd ..
