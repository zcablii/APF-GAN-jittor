import jittor as jt
from jittor import init
from jittor import nn
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            if self.opt.encode_mask:
                # self.fc = nn.Conv2d(8 * nf, 16 * nf, 3, padding=1)
                self.fc = nn.Identity()
            else:
                # In case of VAE, we will sample from random z vector
                self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(opt.semantic_nc, 16 * nf, 3, padding=1)

        self.layer_level = 4
        if opt.num_upsampling_layers == 'more':
            self.layer_level = 5
        if opt.use_interFeature_pos:
            W = opt.crop_size // 2**(self.layer_level+1)
            H = int(W / opt.aspect_ratio)

            self.pos_emb_head = nn.Parameter(jt.zeros(1, 16 * nf, H, W), requires_grad=True).cuda()
            self.pos_emb_middle = nn.Parameter(jt.zeros(1, 16 * nf, H*2, W*2), requires_grad=True).cuda()
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        # self.G_middle_2 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        for i in range(self.layer_level):
            if opt.use_interFeature_pos:
                self.register_parameter('pos_emb_%d' % i, nn.Parameter(jt.zeros(1, int(2**(3-i) * nf), H*2**(i+2), W*2**(i+2), device="cuda"), requires_grad=True))
            if i < self.layer_level - self.opt.sr_scale:
                exec('self.up_%d = SPADEResnetBlock(int(2**(4-i) * nf), int(2**(3-i) * nf), opt)'% i)
            else:
                norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
                exec('self.up_%d = nn.Sequential(norm_layer(nn.ConvTranspose2d(int(2**(4-i) * nf), int(2**(3-i) * nf), kernel_size=3, stride=2,padding=1, output_padding=1)), nn.ReLU(False))' % i)
            if i == self.layer_level - self.opt.sr_scale:
                self.res_blocks = nn.Sequential(*[ResnetBlock(int(2**(3-i) * nf),
                                  norm_layer=norm_layer,
                                  activation=nn.ReLU(False),
                                  kernel_size=opt.resnet_kernel_size) for j in range(4)])
            final_nc = int(2**(3-i) * nf)


        if opt.isTrain:
            self.num_mid_supervision_D = opt.num_D - 1
            if self.num_mid_supervision_D > 0 and opt.pg_niter>0:
                self.inter_conv_img = nn.ModuleList([])
                for i in range(1, self.num_mid_supervision_D+1):
                    self.inter_conv_img.append(nn.Conv2d(final_nc*(2**i), 3, 3, padding=1))
        self.out_conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
        self.cur_ep = 0
        return sw, sh
    
    def pg_merge(self, low_res, high_res, alpha):
        up_res =  nn.interpolate(low_res, high_res.shape[-2:])
        return high_res*alpha + up_res*(1-alpha)

    def execute(self, input, epoch=0, z=None):
        seg = input
        print_inf = False
        if self.cur_ep != epoch:
            print_inf = True
            self.cur_ep = epoch
        if self.opt.use_vae:
            if self.opt.encode_mask:
                # assert list(z.shape[-2:]) == [self.sh, self.sw]
                # z = F.interpolate(z, size=(self.sh, self.sw))
                x = self.fc(z)
            else:
                # we sample z from unit normal and reshape the tensor
                if z is None:
                    z = jt.randn(input.size(0), self.opt.z_dim,
                                    dtype=jt.float32, device=input.get_device())
                x = self.fc(z)
                x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = nn.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)
        if self.opt.use_interFeature_pos: x = x + self.pos_emb_head
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        if self.opt.use_interFeature_pos: x = x + self.pos_emb_middle
        x = self.G_middle_1(x, seg) 
        # x = self.G_middle_2(x, seg) 

        # if self.opt.num_upsampling_layers == 'more' or \
        #    self.opt.num_upsampling_layers == 'most':
        #     x = self.up(x)

        # x = self.G_middle_1(x, seg)

        results = []
        for i in range(self.layer_level):
            if i == self.layer_level - self.opt.sr_scale: 
                x = self.res_blocks(x)
            up_conv = eval(f'self.up_{i}')
            if type(up_conv) == SPADEResnetBlock:
                x = self.up(x)
            x = up_conv(x, seg)
            if self.opt.use_interFeature_pos: 
                pos_emb = eval(f'self.pos_emb_{i}')
                x = x + pos_emb
            if self.opt.isTrain and self.opt.pg_strategy==2:
                if self.opt.pg_niter > 0 and self.opt.num_D - 1 > 0:
                    if epoch>=self.opt.pg_niter:
                        continue
                    lowest_D_level = epoch // (self.opt.pg_niter//(self.opt.num_D - 1))
                    if lowest_D_level - 1 + self.num_mid_supervision_D >= self.opt.num_D - 1:
                        self.num_mid_supervision_D = self.num_mid_supervision_D - 1
                        if print_inf:
                            print('del')
                        del self.inter_conv_img[self.opt.num_D - 1- lowest_D_level]
                    if self.layer_level - i - 2< self.num_mid_supervision_D and i < self.layer_level-1:
                        mid_res = self.inter_conv_img[self.layer_level - i - 2](nn.leaky_relu(x, 2e-1))
                        if print_inf:
                            print('lowest_D_level: ', lowest_D_level,'inter D index: ',self.layer_level - i - 2, 'mid_res shape: ',mid_res.shape)
                        results.append(jt.tanh(mid_res))


            if self.opt.isTrain and (self.opt.pg_strategy in [1,3,4]):
                assert self.opt.pg_niter > 0 and self.opt.num_D - 1 > 0
               
                if epoch>=self.opt.pg_niter:
                    if hasattr(self, 'inter_conv_img'):
                        print('del inter_conv_img')
                        del self.inter_conv_img
                    continue
                current_level = epoch // (self.opt.pg_niter//(self.opt.num_D - 1))
                alpha = (epoch % (self.opt.pg_niter//(self.opt.num_D - 1))) / (self.opt.pg_niter//(self.opt.num_D - 1)/2) - 1
                alpha = 0 if alpha<0 else alpha
                relative_level = self.opt.num_D - current_level - 1
                
                if i == self.layer_level - relative_level - 1 and i + 1 - self.layer_level < 0:
                    mid_res = self.inter_conv_img[self.opt.num_D - current_level - 2](nn.leaky_relu(x, 2e-1))
                    results.append(jt.tanh(mid_res))
                    if alpha>0:
                        # print('epoch,alpha', epoch,alpha)
                        if i+1 == self.layer_level - self.opt.sr_scale: 
                            x = self.res_blocks(x)
                        up_conv = eval(f'self.up_{i+1}')
                        if type(up_conv) == SPADEResnetBlock:
                            x = self.up(x)
                        x = up_conv(x, seg)
                        if self.opt.use_interFeature_pos: 
                            pos_emb = eval(f'self.pos_emb_{i+1}')
                            x = x + pos_emb
                        if self.opt.num_D - current_level - 3>=0:
                            mid_res = jt.tanh(self.inter_conv_img[self.opt.num_D - current_level - 3](nn.leaky_relu(x, 2e-1)))
                            results[0] = self.pg_merge(results[0], mid_res, alpha)
                        else:
                            mid_res = jt.tanh(self.out_conv_img(nn.leaky_relu(x, 2e-1)))
                            results[0] = self.pg_merge(results[0], mid_res, alpha)
                    break
        
        if self.opt.isTrain and (self.opt.pg_strategy in [1,3,4]):
            if len(results) > 0:
                return results  # list of rgb from low res to high
            else:
                x = self.out_conv_img(nn.leaky_relu(x, 2e-1))
                x = jt.tanh(x)
                return x

        x = self.out_conv_img(nn.leaky_relu(x, 2e-1))
        x = jt.tanh(x)
        if len(results)>0:
            results.append(x)
            return results
        else: 
            return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def execute(self, input, z=None):
        return self.model(input)
