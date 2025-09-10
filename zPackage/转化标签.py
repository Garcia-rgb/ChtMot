import os
import argparse
import numpy as np
from tqdm import tqdm

def convert_sequence(seq_path, output_root, split='train'):
    gt_path = os.path.join(seq_path, 'gt/gt.txt')
    if not os.path.exists(gt_path):
        print(f"❌ 找不到 gt.txt: {gt_path}")
        return

    seq_name = os.path.basename(seq_path)
    out_dir = os.path.join(output_root, split, seq_name, 'img1')
    os.makedirs(out_dir, exist_ok=True)

    gt_data = np.loadtxt(gt_path, delimiter=',')
    for frame_id in tqdm(np.unique(gt_data[:, 0]).astype(int), desc=f'Processing {seq_name}'):
        frame_data = gt_data[gt_data[:, 0] == frame_id]
        lines = []
        for row in frame_data:
            fid, track_id, x, y, w, h, conf, cls_id, vis = row
            if int(cls_id) != 1 or track_id < 0:  # 仅保留行人
                continue
            x_center = x + w / 2
            y_center = y + h / 2
            lines.append(f"{int(track_id)} 0 {x_center:.2f} {y_center:.2f} {w:.2f} {h:.2f}")
        if lines:
            label_file = os.path.join(out_dir, f"{frame_id:06d}.txt")
            with open(label_file, 'w') as f:
                f.write('\n'.join(lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_path', type=str, required=True, help='/home/package/G2EMOT/MOT_dataset/MOT17/train/MOT17-02-DPM')
    parser.add_argument('--output_root', type=str, default='labels_with_ids', help='/home/package/G2EMOT/MOT_dataset/MOT17/train/MOT17-02-DPM')
    parser.add_argument('--split', type=str, default='train', help='train 或 test')
    args = parser.parse_args()

    convert_sequence(args.seq_path, args.output_root, args.split)
