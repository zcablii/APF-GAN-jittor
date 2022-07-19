
import jittor as jt
from jittor.dataset.dataset import Dataset
from jittor import init
from jittor import nn
from PIL import Image
import os
import cv2
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_rec(dir, images):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)


def make_dataset(dir, recursive=False, read_cache=False, write_cache=False, remove_hard_imgs=False, is_image=True,remove_img_txt_path=''):
    images = []
    gray_scale_imgs = []
    if remove_hard_imgs:
        assert len(remove_img_txt_path)>0
        f=open(remove_img_txt_path,'r')
        gray_scale_imgs =[each.strip('\n') for each in f.readlines()]
        f.close()
    
    # gray_scale_imgs = ['11353622803_0de2b7b088_b',
    #                     '115758430_d061c87b5a_b',
    #                     '12984815745_65b85ac750_b',
    #                     '143192080_6f625f9395_b',
    #                     '143192081_c18fd910ef_b',
    #                     '14449828172_952d9c1ccc_b',
    #                     '15389204151_392009672e_b',
    #                     '15699264089_f051f2f5b0_b',
    #                     '16043835394_4512120627_b',
    #                     '17300502775_4887bd6968_b',
    #                     '17756857269_249d0baf82_b',
    #                     '18303741595_7110c954ef_b',
    #                     '190934991_4b2f916259_b',
    #                     '21205038811_bbe4f046a2_b',
    #                     '2230377496_dd602938ab_b',
    #                     '23427149724_2461c39798_b',
    #                     '2404318197_ac0494c5a0_b',
    #                     '24444852147_d3a866e78e_b',
    #                     '24717951516_3c16c0f417_b',
    #                     '24916631769_cf56300050_b',
    #                     '25343803572_f3100a110f_b',
    #                     '25530291740_1537abf7ef_b',
    #                     '27815964045_ce4a18d1a1_b',
    #                     '27866308613_dc1d3fb568_b',
    #                     '28088321880_91f66e75be_b',
    #                     '30028685815_2234b40f57_b',
    #                     '30473275945_2004171927_b',
    #                     '30581959281_e4b2e1365b_b',
    #                     '30830526070_b92ffd8dc6_b',
    #                     '31272725728_9f356583be_b',
    #                     '33711975325_28d332df24_b',
    #                     '34747817584_798b7a5177_b',
    #                     '3498123046_47b94083ec_b',
    #                     '3498206776_fa841c44bf_b',
    #                     '3769198222_e46daf27de_b',
    #                     '38067729362_2a54019de5_b',
    #                     '40359461253_2971d838f5_b',
    #                     '40632196493_c97ffedc9c_b',
    #                     '4407077267_ce8387564b_b',
    #                     '440820962_9117c8be51_b',
    #                     '4734017177_cc3364968b_b',
    #                     '4836260860_04386539a6_b',
    #                     '49552286648_a47a82e86a_b',
    #                     '5327756166_c4b6118948_b',
    #                     '5588119616_035763822d_b',
    #                     '6106072019_ed4f40d295_b',
    #                     '6204319732_47b5743e3a_b',
    #                     '6228744576_3c79c3075a_b',
    #                     '6283928494_ef867ddfe5_b',
    #                     '6940995391_a10f28ebb0_b',
    #                     '7153595509_ba346c0a33_b',
    #                     '7263857212_a5973c363c_b',
    #                     '7277064838_5d99d7f50e_b',
    #                     '7290338614_96a1183c38_b',
    #                     '7637257794_4b3c1783ef_b',
    #                     '7659955716_6c1b96be16_b',
    #                     '7755287220_f390495a31_b',
    #                     '7882917054_ab970d6c70_b',
    #                     '8155487010_889e0df0d8_b',
    #                     '8578161796_032fe137e7_b',
    #                     '9838480145_22f35818f7_b', #61 gray images
    #                     '120926019_8c0466d52a_b',
    #                     '13968358004_37dcd3497f_b',
    #                     '14392817340_3f77e4ec00_b',
    #                     '16286115311_203be0c6ae_b',
    #                     '17013629930_35a107c40c_b',
    #                     '2164473007_a358e94558_b',
    #                     '2165266318_109fc633ea_b',
    #                     '22514082109_2838139115_b',
    #                     '2289396338_bb0294327e_b',
    #                     '2488216777_99b8e50d39_b',
    #                     '27065943994_6dcf274854_b',
    #                     '27357102244_b0efe3ea0a_b',
    #                     '27464753_c24b5aa361_b',
    #                     '27729487482_06ebd5485e_b',
    #                     '28262860692_3c5919e1ae_b',
    #                     '31846031504_ae69f40c41_b',
    #                     '334489980_56e8f8a2bb_b',
    #                     '3498123046_47b94083ec_b',
    #                     '354087535_03df5d8697_b',
    #                     '3578919486_471205ae2c_b',
    #                     '358463634_8842e8cbb4_b',
    #                     '3652843024_c5afe2a357_b',
    #                     '44757414361_7221816b7b_b',
    #                     '4527908178_598b41321c_b',
    #                     '45935190865_352ddfb7fa_b',
    #                     '4728217649_33a5279f6e_b',
    #                     '475311092_4775c340ba_b',
    #                     '4787996130_3afa6bd023_b',
    #                     '48432_b67ec6cd63_b',
    #                     '505744099_705c930855_b',
    #                     '50871636396_42426bf844_b',
    #                     '509229127_8c1ab2b050_b',
    #                     '5728517799_141d187804_b',
    #                     '5941497734_de76835a99_b',
    #                     '5979188779_26e3bf2509_b',
    #                     '6206377271_a4de595b8f_b',
    #                     '6215218893_416d73df52_b',
    #                     '6538371727_841b7c2c0c_b',
    #                     '6915057706_90b91a4700_b',
    #                     '7424498436_e83c8a1511_b',
    #                     '7787425268_b3d3b2693b_b',
    #                     '7886904976_88e56fdc8f_b',
    #                     '7886905532_f5d9152b25_b',
    #                     '8002028435_f8e8a152fa_b',
    #                     '8589472522_e361bf90f2_b',
    #                     '8668205295_af5a44a8a6_b', #46
    #                     '4692382550_7e9e9d36a5_b', 
    #                     '4474881130_66bea9f429_b', 
    #                     '3689315739_12d10db38e_b', 
    #                     '3648375640_7695fb0b26_b', 
    #                     '2853193459_2c73c058b6_b', 
    #                     '50017718417_304fe59ef0_b', 
    #                     '49678520981_2c61f9952d_b', 
    #                     '50555942066_4e706bbd3b_b', 
    #                     '31712457640_b9beca1bd4_b', 
    #                     '6893674060_a7da19caf4_b', 
    #                     '6701719487_a45a29c31a_b', 
    #                     '5408611299_11cd8424bc_b', 
    #                     '5100446432_8c83422237_b', 
    #                     '2589791569_d188d06503_b', 
    #                     '2341006632_252327b4c8_b', 
    #                     '2288598123_91d9d63f1c_b', 
    #                     '448013412_6613ed0cd2_b', 
    #                     '366878902_807a2eb6a5_b', 
    #                     '59731975_2663259fe1_b',
    #                     '50959067_b9d1fe1cb4_b', 
    #                     '50958346_8593546c32_b', 
    #                     '45979216_5044a6f815_b', 
    #                     '41315555_d5e654f195_b'] #23

    gray_scale_imgs_path = []
    extension = '.jpg' if is_image else '.png'
    for each_img in gray_scale_imgs:
        gray_scale_imgs_path.append(os.path.join(dir, each_img+extension))

    if read_cache:
        possible_filelist = os.path.join(dir, 'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist, 'r') as f:
                images = f.read().splitlines()
                return images

    if recursive:
        make_dataset_rec(dir, images)
    else:
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

        for root, dnames, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        
        if remove_hard_imgs:
            images = [x for x in images if x not in gray_scale_imgs_path]

    if write_cache:
        filelist_cache = os.path.join(dir, 'files.list')
        with open(filelist_cache, 'w') as f:
            for path in images:
                f.write("%s\n" % path)
            print('wrote filelist cache at %s' % filelist_cache)

    return images


def default_loader(path):
    img_B = Image.open(path)
    img_B = np.array(img_B).astype("uint8")
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
    print('img_B',type(img_B),img_B)
    return img_B


class ImageFolder(Dataset):

    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)
        if (len(imgs) == 0):
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if (self.transform is not None):
            img = self.transform(img)
        if self.return_paths:
            return (img, path)
        else:
            return img

    def __len__(self):
        return len(self.imgs)
