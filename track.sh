cd src
python track.py \
  --task mot \
  --exp_id mot17base \
  --test_mot17 True \
  --load_model /root/autodl-tmp/my_model_save/mot17_50_base/model_last.pth \
  --conf_thres 0.4
cd ..
