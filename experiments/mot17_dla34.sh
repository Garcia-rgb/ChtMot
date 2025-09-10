cd src
python train.py \
  --task mot \
  --exp_id mot17_100_base \
  --load_model '../src/lib/models/ctdet_coco_dla_2x.pth' \
  --data_cfg '../src/lib/cfg/mot17.json'
cd ..
