import os
import os.path as osp
import numpy as np

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

# 设置数据路径
seq_root = '/home/package/G2EMOT/MOT_dataset/MOT17/test'
label_root = '/home/package/G2EMOT/MOT_dataset/MOT17/labels_with_ids/test'
mkdirs(label_root)

# 获取所有序列子目录
seqs = [s for s in os.listdir(seq_root) if osp.isdir(osp.join(seq_root, s))]

tid_curr = 0
tid_last = -1

for seq in seqs:
    print('Processing:', seq)
    seq_path = osp.join(seq_root, seq)
    seq_info_path = osp.join(seq_path, 'seqinfo.ini')
    if not osp.exists(seq_info_path):
        print(f"Warning: {seq_info_path} not found. Skipping.")
        continue

    # 读取分辨率信息
    with open(seq_info_path) as f:
        seq_info = f.read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    # 加载 ground truth
    gt_txt = osp.join(seq_path, 'gt', 'gt.txt')
    if not osp.exists(gt_txt):
        print(f"Warning: {gt_txt} not found. Skipping.")
        continue

    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
    idx = np.lexsort(gt.T[:2, :])
    gt = gt[idx, :]

    # 创建输出目录
    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)

    for fid, tid, x, y, w, h, mark, cls, vis in gt:
        if int(mark) != 1 or int(cls) != 1:
            continue  # 跳过未标注或非行人

        fid = int(fid)
        tid = int(tid)
        if tid != tid_last:
            tid_curr += 1
            tid_last = tid

        # 转为中心坐标 + 归一化
        x += w / 2
        y += h / 2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
            tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)

        with open(label_fpath, 'a') as f:
            f.write(label_str)
