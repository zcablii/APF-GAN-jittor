
## This repository is the official implementation of CVM "APF-GAN: Exploring asymmetric pre-training and fine-tuning strategy for conditional generative adversarial network" at: [https://www.sciopen.com/article/pdf/10.1007/s41095-023-0357-1.pdf?stage=5](https://www.sciopen.com/article/pdf/10.1007/s41095-023-0357-1.pdf?stage=5).

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@article{Li2024, 
author = {Yuxuan Li and Lingfeng Yang and Xiang Li},
title = {APF-GAN: Exploring asymmetric pre-training and fine-tuning strategy for conditional generative adversarial network},
year = {2024},
journal = {Computational Visual Media},
volume = {10},
number = {1},
pages = {187-192},
url = {https://www.sciopen.com/article/10.1007/s41095-023-0357-1},
doi = {10.1007/s41095-023-0357-1},
}
```

# 第二届计图(Jittor)人工智能挑战赛--赛道一

## 赛题说明

图像生成任务一直以来都是十分具有应用场景的计算机视觉任务，从语义分割图生成有意义、高质量的图片仍然存在诸多挑战，如保证生成图片的真实性、清晰程度、多样性、美观性等。

清华大学计算机系图形学实验室从Flickr官网收集了1万两千张高清（宽1024、高768）的风景图片，并制作了它们的语义分割图。其中，1万对图片被用来训练。训练数据集10000张风景图可以从[这里](https://cloud.tsinghua.edu.cn/f/1d734cbb68b545d6bdf2/?dl=1)下载。其中 label 是值在 0~28 的灰度图。本赛题将会提供1000张测试分割图片，参赛选手需要根据测试图片生成符合标签含义的风景图片。可在这里下载[A榜测试数据集](https://cloud.tsinghua.edu.cn/f/70195945f21d4d6ebd94/?dl=1) 和 [B榜试数据集](https://cloud.tsinghua.edu.cn/f/980d8204f38e4dfebbc8/?dl=1)。

## 赛题评测指标

赛题分为A、B两个榜单。A榜结束后将按排名筛选、审核让若干支队伍进入B榜。

A榜评分公式：mask accuary * (美学评分 / 10 * 50% + (100 - FID) / 100 * 50%)

mask accuary：根据用户生成的1000张图片，使用 SegFormer 模型[1]对图片进行分割，然后计算分割图和gt分割图的mask accuary=(gt_mask == pred_mask).sum() / (H * W)，确保生成的图片与输入的分割图相对应。mask accuary 越大越好，其数值范围是0~1。

美学评分：由深度学习美学评价模型为图片进行美学评分，大赛组委会参考论文中的实现自动美学评分。该分数将归一化将到 0~1。

FID（Frechet Inception Distance score）：计算生成的 1000 张图与训练图片的FID，该指标越小越好，将FID的100到0线性映射为 0 到 1。由于 baseline 代码的 FID 在 100 以内，所以 FID 大于 100 的将置为 100。
B榜评分公式：mask accuary * (美学评分 / 10 * 25% + (100 - FID) / 100 * 25%) + 15% 大众投票 + 15% 专家投票 + 20% 答辩分数 

用户提交B榜生成图片 1000 张，mask accuary、美学评分和 FID 计算方式参考 A 榜指标。此外，选手需要额外自己选择 3 张高质量图片进行投票评选。该3张图片将与进入B组的若干组选手共同接受大众以及专家的投票。关于投票具体细节将在A榜结束时公布。

## 数据集文件结构

```bash
--data
    --train
        --imgs
            --train_img0.jpg
            --train_img1.jpg
            ...
        --lables
            --train_img0.png
            --train_img1.png
            ...
    --test
        --labels
            --test_img0.png
            --test_img1.png
            ...
```

## 安装

This code requires python 3+ and Jittor 1.3. Please install dependencies by

```bash
sudo apt install python3.9-dev libomp-dev  
python3.9 -m pip install jittor  
pip install -r requirements.txt
```

## 模型训练
我们训练包括**渐进尺寸预训练阶段**180epoch和**增广式微调阶段**130epoch共计310epoch, 可以通过执行如下命令直接运行完成两个阶段的训练，其中训练数据全部采用官方提供的训练集，不采用任何额外的数据集和预训练模型。input_path参数传入官方提供的数据集路径入口。训练过程同时会遍历训练数据统计出一个非常小的纯色语义图字典`./checkpoints/label2img/pure_img.npy`，该字典将用于后续测试中融合使用。为了便于后续测算线下FID，我们会在该阶段获得所有训练图片用于计算FID的统计量，保存于`./checkpoints/label2img/train_fid_m.npy`, `./checkpoints/label2img/train_fid_s.npy`。
```python
python train.py --input_path='./data/train/' 
```

下面对训练的两阶段命令稍加展开：
#### (1) 渐进尺寸预训练阶段, 180epoch
该阶段我们通过渐进尺寸策略(64->128->256->512)，降低网络学习难度，使得网络获得一个较好的直接生成512图像的初始化结构。该阶段的具体命令如下：
```python
python train_phase.py --input_path='./data/train/' --batchSize=10 --niter=180 --pg_niter=180 --pg_strategy=1 --num_D=4
```



#### (2) 增广式微调阶段，130epoch
该阶段我们将上一阶段的最终模型做初始化，并引入大量增广训练操作如可微分的数据增广、语义空间引导的噪音、以及inception感知损失。该阶段的具体命令如下：
```python
python train_phase.py --input_path='./data/train/' --batchSize=5 --niter=310 --pg_niter=180 --pg_strategy=1 --save_epoch_freq=1 --num_D=4 --diff_aug='color,crop,translation' --inception_loss --use_seg_noise --continue_train --which_epoch=180
```


## 模型选择
根据实践我们发现训练过程中不同checkpoint模型得分差异很大，所以不能盲目地选择最终模型或者中间模型做测试。我们构建了一种可行的线下验证方式，能够粗略地评估不同checkpoint的性能：即测算checkpoint的生成图片与官方训练集图片的FID指标，该指标需要越小越好。比赛过程中我们利用[pytorch版本测FID的工具](https://github.com/mseitzer/pytorch-fid)对微调阶段271epoch-310epoch的所有checkpoint的生成图片进行了其与训练集FID的测算，如下是测算记录：
| ckpt | 线下FID       |
| :---:| :---:         |
| 271  | 33.27         |
| 272  | 29.78         |
| **273**  | **28.64** |
| 274  | 30.84         |
| 275  | 31.95         |
| 276  | 30.48         |
| 277  | 30.30         |
| 278  | 32.19         |
| 279  | 34.45         |
| 280  | 34.58         |
| 281  | 33.22         |
| 282  | 31.22         |
| 283  | 31.55         |
| 284  | 31.59         |
| 285  | 32.83         |
| 286  | 31.05         |
| 287  | 31.04         |
| 288  | 29.79         |
| 289  | 33.04         |
| 290  | 30.70         |
| 291  | 30.95         |
| 292  | 28.93         |
| 293  | 30.79         |
| 294  | 32.00         |
| 295  | 33.43         |
| 296  | 30.13         |
| 297  | 29.54         |
| **298**  | **28.59**     |
| **299**  | **28.12**     |
| 300  | 31.99         |
| 301  | 30.15         |
| 302  | 29.98         |
| 303  | 29.02         |
| 304  | 31.24         |
| 305  | 30.94         |
| 306  | 31.76         |
| 307  | 31.79         |
| 308  | 31.29         |
| 309  | 31.35         |

为了便于复现，我们的repo中提供了**jittor版本的测算FID工具**，可以用来自动评测所有checkpoints的FID指标便于模型选择，相关命令如下：
```python
python util/fid.py ./data/train/imgs/ ./data/test/labels ./checkpoints/label2img 270-310
```
因为和pytorch模型的差异，其测算的FID分数范围会不同于表格中的分数(以及下文的所有线下FID分数)，但是相对关系是一致的，能够帮助我们找到FID最优的多个模型。


接下来，我们选择得分最优的3个checkpoint，例如本次比赛中我们训练得到的273,298,299模型。我们采用权重平均的均值专家方法融合2个或多个模型，并再次利用FID进行线下测试评估，结果如下：
| ckpt_merge | 线下FID |
| :---:| :---:         |
| 273 + 298 | 27.37 |
| **273 + 299** | **26.93** |
| 298 + 299 | 27.72 |
| 273 + 298 + 299 | 27.20 |

我们发现，融合模型性能有了非常明显的提升，最终采用avg_273_299模型作为主体预测模型，相关权重我们提供在了zip的checkpoints文件夹中。我们无法保证新机器环境下训练过程中的checkpoints也有和我们完全一致的FID测试结果，但选择模型的逻辑是相对客观且稳定的，复现方一定能够利用我们的工具和策略选择得到与我们最终模型性能相当的模型。以273和299模型为例，融合模型的相关代码如下：
```python
python util/merge_ckpt.py ./checkpoints/label2img 273 299
```

## 模型测试
`注意！根据我们在两台不同机器相同环境的测试情况，使用命令: python test_phase.py --input_path='./data/test/labels' --which_epoch=avg_273_299 --seed=2433 应该能得到与最高分非常接近的结果，如果举办方（因机器原因）未能得出最高分结果，那么请采用如下赛方要求的默认方式: python test.py --input_path='./data/test/labels' 一定可以复现最高结果`

我们已将模型打包，可以直接测试。模型在GoogleDrive上也可以下载 [avg_273_299_net_E](https://drive.google.com/file/d/1c9lpUK_Z1B77WD7H24PRSQJscSl_RQDf/view?usp=sharing), [avg_273_299_net_G](https://drive.google.com/file/d/1mI2FAi7UaEg0nUzQOspxlpSZKeUvcSIl/view?usp=sharing),下载后的模型请放于./checkpoints/label2img/文件夹下。
```python
python test.py --input_path='./data/test/labels'
```

由于我们的模型带有随机噪音层，不同噪音对应的生成结果实际测试中性能上也有一定的浮动(线下FID浮动在0.4以内)。我们对avg_273_299模型随机运行了20次，并测算其线下FID：

26.99, 26.95, 26.88, 26.91, 27.09, 26.82, 27.07, 27.13, **26.76**, 26.93, 27.05, 27.01, 27.02, 27.00, 26.91, 26.86, 27.04, 26.99, 26.97, 27.04

最终我们选择线下FID最低的结果26.76提交线上，再结合纯色语义图字典融合策略，得到最终综合56.76 rank1得分。注意，上述test命令已经实现了多次随机生成(我们代码中默认20次)，并测算与训练集图片的FID，并保留最优的FID结果的整套自动化流程。如果需要得到单次随机噪音的结果，可以使用如下命令：
```python
python test_phase.py --input_path='./data/test/labels' --which_epoch=avg_273_299
```
注意到复现机器环境随机性的不同，根据我们整理复现的情况，按照```python test.py --input_path='./data/test/labels' ```的测试流程大概率能够得到比目前本队伍top1得分56.76更高的复现分数。

