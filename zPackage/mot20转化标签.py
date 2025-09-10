import os
import numpy as np
from tqdm import tqdm

def convert_sequence(seq_path, output_root, split='train'):
    gt_path = os.path.join(seq_path, 'gt/gt.txt')
    if not os.path.exists(gt_path):
        print(f"❌ 找不到 gt.txt: {gt_path}")
        return

    seq_name = os.path.basename(seq_path.rstrip('/'))
    out_dir = os.path.join(output_root, split, seq_name, 'img1')
    os.makedirs(out_dir, exist_ok=True)

    gt_data = np.loadtxt(gt_path, delimiter=',')
    for frame_id in tqdm(np.unique(gt_data[:, 0]).astype(int), desc=f'Processing {split}/{seq_name}'):
        frame_data = gt_data[gt_data[:, 0] == frame_id]
        lines = []
        for row in frame_data:
            fid, track_id, x, y, w, h, conf, cls_id, vis = row
            if int(cls_id) != 1 or track_id < 0:  # 仅保留行人
                continue
            x_center = x + w / 2
            y_center = y + h / 2
            lines.append(f"{int(track_id)} 0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        if lines:
            label_file = os.path.join(out_dir, f"{frame_id:06d}.txt")
            with open(label_file, 'w') as f:
                f.write('\n'.join(lines))

def convert_all_sequences(base_dir):
    for split in ['train', 'test']:
        root_path = os.path.join(base_dir, split)
        output_root = os.path.join(base_dir, 'labels_with_ids')
        if not os.path.exists(root_path):
            print(f"⚠️ 跳过不存在的目录: {root_path}")
            continue
        seq_list = [os.path.join(root_path, d) for d in os.listdir(root_path)
                    if os.path.isdir(os.path.join(root_path, d))]
        print(f"📂 正在处理 {split} 集，共 {len(seq_list)} 个序列：{[os.path.basename(p) for p in seq_list]}")
        for seq_path in seq_list:
            convert_sequence(seq_path, output_root, split)

if __name__ == '__main__':
    base_dir = '/root/autodl-tmp/MOT20'
    convert_all_sequences(base_dir)
