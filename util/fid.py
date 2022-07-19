import jittor as jt
from jittor import nn
from jittor import models
import os
import numpy as np
from scipy import linalg
import pathlib
import sys
from imageio import imread
from skimage.transform import resize
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x):
        return x
import shutil
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
jt.flags.use_cuda = 1

class InceptionV3(nn.Module):
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {64: 0, 192: 1, 768: 2, 2048: 3}

    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert (self.last_needed_block <= 3), 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.Pool(3, stride=2, op='maximum')]
        self.blocks.append(nn.Sequential(*block0))
        if (self.last_needed_block >= 1):
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.Pool(3, stride=2, op='maximum')]
            self.blocks.append(nn.Sequential(*block1))
        if (self.last_needed_block >= 2):
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if (self.last_needed_block >= 3):
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def execute(self, inp):
        outp = []
        x = inp
        if self.resize_input:
            x = nn.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = ((2 * x) - 1)
        for (idx, block) in enumerate(self.blocks):
            x = block(x)
            if (idx in self.output_blocks):
                outp.append(x)
            if (idx == self.last_needed_block):
                break
        return outp

def fid_inception_v3():
    inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = jt.load('./util/net_params.pkl')
    jt_dict = {}
    for k in state_dict.keys():
        jt_dict[k] = jt.Var(state_dict[k])
    inception.load_state_dict(jt_dict)
    return inception

class FIDInceptionA(models.inception.InceptionA):

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def execute(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = nn.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return jt.contrib.concat(outputs, dim=1)

class FIDInceptionC(models.inception.InceptionC):

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def execute(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = nn.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return jt.contrib.concat(outputs, dim=1)

class FIDInceptionE_1(models.inception.InceptionE):

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def execute(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = jt.contrib.concat(branch3x3, dim=1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = jt.contrib.concat(branch3x3dbl, dim=1)
        branch_pool = nn.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return jt.contrib.concat(outputs, dim=1)

class FIDInceptionE_2(models.inception.InceptionE):

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def execute(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = jt.contrib.concat(branch3x3, dim=1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = jt.contrib.concat(branch3x3dbl, dim=1)
        branch_pool = nn.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return jt.contrib.concat(outputs, dim=1)


def get_activations(files, model, batch_size=50, dims=2048,verbose=False):
    model.eval()
    if ((len(files) % batch_size) != 0):
        print('Warning: number of images is not a multiple of the batch size. Some samples are going to be ignored.')
    if (batch_size > len(files)):
        print('Warning: batch size is bigger than the data size. Setting batch size to data size')
        batch_size = len(files)
    print(len(files),batch_size)
    n_batches = (len(files) // batch_size)
    n_used_imgs = (n_batches * batch_size)
    pred_arr = np.empty((n_used_imgs, dims))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = (i * batch_size)
        end = (start + batch_size)
        images = np.array([resize(imread(str(f)).astype(np.float32), (256, 256, 3)) for f in files[start:end]])
        images = images.transpose((0, 3, 1, 2))
        images /= 255
        batch = jt.Var(images)
        pred = model(batch)[0]
        if ((pred.shape[2] != 1) or (pred.shape[3] != 1)):
            pred = nn.AdaptiveAvgPool2d(output_size=(1, 1))(pred)
        pred_arr[start:end] = pred.numpy().reshape(batch_size, -1)
        # raise RuntimeError('origin source: <pred.cpu().data.numpy().reshape(batch_size, (- 1))>,There are only 1 args in Pytorch reshape function, but you provide 2')
    if verbose:
        print(' done')
    print(pred_arr.shape)
    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-06):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert (mu1.shape == mu2.shape), 'Training and test mean vectors have different lengths'
    assert (sigma1.shape == sigma2.shape), 'Training and test covariances have different dimensions'
    diff = (mu1 - mu2)
    (covmean, _) = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if (not np.isfinite(covmean).all()):
        offset = (np.eye(sigma1.shape[0]) * eps)
        covmean = linalg.sqrtm((sigma1 + offset).dot((sigma2 + offset)))
    if np.iscomplexobj(covmean):
        if (not np.allclose(np.diagonal(covmean).imag, 0, atol=0.001)):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (((diff.dot(diff) + np.trace(sigma1)) + np.trace(sigma2)) - (2 * tr_covmean))

def calculate_activation_statistics(files, model, batch_size=50, dims=2048, verbose=False):
    act = get_activations(files, model, batch_size, dims, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return (mu, sigma)

def _compute_statistics_of_path(path, model, batch_size, dims):
    if path.endswith('.npz'):
        f = np.load(path)
        (m, s) = (f['mu'][:], f['sigma'][:])
        f.close()
    else:
        path = pathlib.Path(path)
        files = (list(path.glob('*.jpg')) + list(path.glob('*.png')))
        (m, s) = calculate_activation_statistics(files, model, batch_size, dims)
    return (m, s)

def calculate_fid_given_paths(train_path, test_path, stat_path, batch_size, dims, for_train=False):
    # if not os.path.exists(test_path):
    #     raise RuntimeError(('Invalid path: %s' % test_path))
    # if not os.path.exists(train_path):
    #     raise RuntimeError(('Invalid path: %s' % train_path))
    # if not os.path.exists(stat_path):
    #     raise RuntimeError(('Invalid path: %s' % stat_path))
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])

    train_m_path = os.path.join(stat_path,'train_fid_m.npy')
    train_s_path = os.path.join(stat_path,'train_fid_s.npy')
    if for_train:
        (m1,s1) = _compute_statistics_of_path(train_path, model, batch_size, dims)
        np.save(train_m_path,m1)
        np.save(train_s_path,s1)
        return
    if os.path.exists(train_m_path) and os.path.exists(train_s_path):
        m1 = np.load(train_m_path)
        s1 = np.load(train_s_path)
    else:
        (m1,s1) = _compute_statistics_of_path(train_path, model, batch_size, dims)
        np.save(train_m_path,m1)
        np.save(train_s_path,s1)

    (m2, s2) = _compute_statistics_of_path(test_path, model, batch_size, dims)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def get_offline_fid(train_path, test_path, stat_path='checkpoints/label2img'):
    batch_size = 10
    dims = 2048
    fid_value = calculate_fid_given_paths(train_path, test_path,stat_path, batch_size, dims)
    # print(test_path, 'fid_value: ', fid_value)
    return fid_value

if __name__=='__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    ckpts_path = sys.argv[3]
    use_pure = sys.argv[4]
    seed = sys.argv[5]
    ckpts_to_test_fid = []
    for i in range(6, len(sys.argv)):
        ckpt = str(sys.argv[i])
        print(ckpt.split('-'))
        if len(ckpt.split('-'))>1:
            ckpts = list(range(eval(ckpt.split('-')[0]),eval(ckpt.split('-')[1])+1))
            ckpts = [str(x) for x in ckpts]
            ckpts_to_test_fid = ckpts_to_test_fid+ckpts
            continue
        ckpts_to_test_fid.append(ckpt)
    checkpoints_dir,name = os.path.split(ckpts_path)

    out_path = './results'
    # out_path = f'./results_{seed}'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for ep in ckpts_to_test_fid:
        which_epoch = os.path.join(ckpts_path,ep)
        if not os.path.exists(which_epoch+'_net_G.pkl'):
            continue
        print(("python test_phase.py --input_path=%s --checkpoints_dir=%s --name=%s --out_path=%s --which_epoch=%s --use_pure=%s --seed=%s" % (test_path,checkpoints_dir,name,out_path,ep,use_pure,seed)))
        os.system("python test_phase.py --input_path=%s --checkpoints_dir=%s --name=%s --out_path=%s --which_epoch=%s --use_pure=%s --seed=%s" % (test_path,checkpoints_dir,name,out_path,ep,use_pure,seed))
        fid = get_offline_fid(train_path, out_path)
        # save fid
        with open("temp_jittor.txt", "a") as f:
            f.write(f'{fid}\n')
        print('=========================================================================')
        print(which_epoch,' fid: ', fid)
        print('=========================================================================')
    # shutil.rmtree('./temp') 
