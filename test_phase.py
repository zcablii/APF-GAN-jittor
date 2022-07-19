import jittor as jt
import os
from collections import OrderedDict
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import numpy as np
from util.util import mkdir, save_image, get_gray_label

jt.flags.use_cuda = 1

import ntpath

opt = TestOptions().parse()

if opt.seed!=-1:
    seed = opt.seed
    jt.set_global_seed(seed)
    np.random.seed(0)
    print(f'set seed to {seed}')
else:
    print('using random seed')

if opt.USE_AMP:
    jt.flags.auto_mixed_precision_level = 5

opt.label_dir = get_gray_label(opt.input_path,
                               for_test=True,
                               temp_dir='test_post_label')

dataloader = data.create_dataloader(opt)
model = Pix2PixModel(opt)
model.eval()

# test
def np_multi(a, b):
    for x in range(b.shape[-1]):
        b[:, :, x] = a * b[:, :, x]
    return b

mkdir(opt.out_path)
if opt.use_pure:
    ref_dic = np.load(os.path.join(os.path.join(opt.checkpoints_dir, opt.name),
                                   'pure_img.npy'),
                      allow_pickle=True)[0]
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    generated = model(data_i, mode='inference')
    img_path = data_i['path']
    for b in range(generated.shape[0]):
        generated_img = generated[b].detach().float().numpy()
        generated_img = (np.transpose(generated_img,
                                      (1, 2, 0)) + 1) / 2.0 * 255.0
        generated_img = generated_img.astype(np.uint8)
        if opt.use_pure:
            label_map = np.transpose(
                np.array(data_i['label'][b]).astype("uint8"), (1, 2, 0))
            if len(label_map.shape) > 2:
                label_map = label_map[:, :, 0]
            img_shape = label_map.shape
            label_map = label_map.flatten()
            max_label = np.argmax(np.bincount(label_map))
            max_per = np.count_nonzero(label_map == max_label) / len(label_map)

            if max_per > 0.98:
                label_map = label_map.reshape(img_shape)
                train_img_mask = np.ones(img_shape)
                train_gen_mask = np.ones(img_shape)
                train_img_mask[label_map != max_label] = 0
                train_gen_mask[label_map == max_label] = 0

                ref_img = ref_dic[max_label].pop()
                ref_img = np.array(ref_img).astype("uint8")
                generated_img = np_multi(train_gen_mask,
                                         generated_img) + np_multi(
                                             train_img_mask, ref_img)

        short_path = ntpath.basename(img_path[b:(b + 1)][0])
        name = os.path.splitext(short_path)[0]
        image_name = os.path.join('%s.jpg' % (name))
        save_path = os.path.join(opt.out_path, image_name)
        save_image(generated_img, save_path, create_dir=True, is_img=True)
