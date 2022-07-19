import argparse
import os
import random

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--which_epoch',
                    type=str,
                    default='avg_273_299',
                    help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--train_path', type=str, default='./data/train/imgs', help='training images.')
parser.add_argument('--input_path', type=str, default='./data/test/labels', help='testing labels.')

opt = parser.parse_args()

with open("temp_jittor.txt", "w") as f:
    pass
for _ in range(50):
    seed = random.randint(0, 4096)
    with open("temp_jittor.txt", "a") as f:
        f.write(f'{seed}\t')
    os.system(f"python util/fid.py \
        {opt.train_path} \
        {opt.input_path} \
        ./checkpoints/label2img \
        False \
        {seed} \
        {opt.which_epoch}")

with open("temp_jittor.txt", "r") as f:
    fid_list = f.read().splitlines()
fid_list = [line.split('\t') for line in fid_list]
fid_list = sorted(fid_list, key=lambda x: x[1], reverse=False)
print(f'choose seed {fid_list[0][0]}, with fid {fid_list[0][1]}')
seed = fid_list[0][0]
os.system(f"python util/fid.py \
    {opt.train_path} \
    {opt.input_path} \
    ./checkpoints/label2img \
    True \
    {seed} \
    {opt.which_epoch}")
