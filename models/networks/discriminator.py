import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import util.util as util


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        # parser.add_argument('--num_D', type=int, default=3,
        #                     help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_mid_supervision_D = opt.num_D - 1

        if self.opt.pg_strategy == 1:
            for i in range(opt.num_D):
                subnetD = self.create_single_discriminator(opt, i)
                exec('self.multiscale_discriminator_%d = subnetD' % i)

        elif self.opt.pg_strategy == 2:
            for i in range(self.num_mid_supervision_D):
                subnetD = self.create_single_discriminator(opt)
                exec('self.mid_sup_discriminator_%d = subnetD' % i)
            for i in range(opt.num_D):
                subnetD = self.create_single_discriminator(opt)
                exec('self.multiscale_discriminator_%d = subnetD' % i)
        
        elif self.opt.pg_strategy == 3:
            self.multiscale_discriminator= self.create_single_discriminator(opt, 3)

        else:
            for i in range(opt.num_D):
                subnetD = self.create_single_discriminator(opt)
                exec('self.multiscale_discriminator_%d = subnetD' % i)
        self.current_alpha = 0
            

    def create_single_discriminator(self, opt, level = 0):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt, level)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        if self.opt.USE_AMP:
            return jt.float16(nn.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=1,
                            count_include_pad=False))
        return nn.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=1,
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def execute(self, input, epoch):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss

        if self.opt.pg_strategy == 2:
            if type(input) == list:
                assert self.num_mid_supervision_D > 0 and self.opt.pg_niter > 0
                lowest_D_level = epoch // (self.opt.pg_niter//(self.opt.num_D - 1)) - 1
                if lowest_D_level + self.num_mid_supervision_D >= self.opt.num_D - 1 or epoch>=self.opt.pg_niter:
                    self.num_mid_supervision_D = self.num_mid_supervision_D - 1
                    D = eval(f'self.mid_sup_discriminator_{lowest_D_level}')
                    del D
                for i in range(self.num_mid_supervision_D):
                    D = eval(f'self.mid_sup_discriminator_{i}')
                    out = D(input[i+1]) # mid level rgb, not include output size img
                    if not get_intermediate_features:
                        out = [out]
                input = input[0]
                assert input.shape[-1] == self.opt.crop_size

        elif self.opt.pg_strategy == 1:
            assert self.opt.pg_niter > 0 and self.opt.num_D - 1 > 0
            if epoch>=self.opt.pg_niter:
                for i in range(self.opt.num_D): # empirical working version
                    D = eval(f'self.multiscale_discriminator_{i}')
                    out = D(input)
                    if not get_intermediate_features:
                        out = [out]
                    result.append(out)
                    if self.opt.one_pg_D:
                        break
                    input = self.downsample(input)
            else:
                current_level = epoch // (self.opt.pg_niter//(self.opt.num_D - 1))
                alpha = (epoch % (self.opt.pg_niter//(self.opt.num_D - 1))) / (self.opt.pg_niter//(self.opt.num_D - 1)/2) - 1
                alpha = 0 if alpha<0 else alpha
                relative_level = self.opt.num_D - current_level - 1
                if relative_level > 0:
                    if alpha == 0:
                        assert len(input) == 1
                        input = input[0]
                        if self.opt.reverse_map_D:
                            ordered_D = range( current_level+1)
                        else:
                            ordered_D = range( current_level,-1,-1)
                        for i in ordered_D:
                            D = eval(f'self.multiscale_discriminator_{i}')
                            out = D(input)
                            if not get_intermediate_features:
                                out = [out]
                            result.append(out)
                            if self.opt.one_pg_D:
                                break
                            input = self.downsample(input)

                    else:
                        input = input[0]
                        if ((epoch-1) % (self.opt.pg_niter//(self.opt.num_D - 1))) / (self.opt.pg_niter//(self.opt.num_D - 1)/2) - 1 <=0:
                            D = eval(f'self.multiscale_discriminator_{current_level+1}')
                            sub_D = eval(f'self.multiscale_discriminator_{current_level}')
                            D.load_state_dict(sub_D.state_dict())
                            if self.opt.one_pg_D:
                                del sub_D
                           
                        D = eval(f'self.multiscale_discriminator_{current_level+1}')
                        out = D(input, alpha=alpha)
                        if not get_intermediate_features:
                            out = [out]
                        result.append(out)
                        input = self.downsample(input)
                        
                        if self.opt.reverse_map_D:
                            ordered_D = range( current_level+1)
                        else:
                            ordered_D = range( current_level,-1,-1)

                        for i in ordered_D:
                            if self.opt.one_pg_D:
                                break
                            D = eval(f'self.multiscale_discriminator_{i}')
                            out = D(input)
                            if not get_intermediate_features:
                                out = [out]
                            result.append(out)
                            input = self.downsample(input)

        elif self.opt.pg_strategy == 3:
            if type(input)==list:
                input = input[0]
            D = self.multiscale_discriminator
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)

        elif self.opt.pg_strategy == 4:
            assert self.opt.pg_niter > 0 and self.opt.num_D - 1 > 0
            if epoch>=self.opt.pg_niter:
                for i in range(self.opt.num_D-1,-1,-1):
                    D = eval(f'self.multiscale_discriminator_{i}')
                    out = D(input)
                    if not get_intermediate_features:
                        out = [out]
                    result.append(out)
                    if self.opt.one_pg_D:
                        break
                    input = self.downsample(input)
            else:
                current_level = epoch // (self.opt.pg_niter//(self.opt.num_D - 1))
                alpha = (epoch % (self.opt.pg_niter//(self.opt.num_D - 1))) / (self.opt.pg_niter//(self.opt.num_D - 1)/2) - 1
                alpha = 0 if alpha<0 else alpha
                relative_level = self.opt.num_D - current_level - 1
                if relative_level > 0:
                    if alpha == 0:
                        assert len(input) == 1
                        input = input[0]
                        for i in range( current_level,-1,-1):
                            D = eval(f'self.multiscale_discriminator_{i}')
                            out = D(input)
                            if not get_intermediate_features:
                                out = [out]
                            result.append(out)
                            if self.opt.one_pg_D:
                                break
                            input = self.downsample(input)

                    else:
                        input = input[0]
                        if ((epoch-1) % (self.opt.pg_niter//(self.opt.num_D - 1))) / (self.opt.pg_niter//(self.opt.num_D - 1)/2) - 1 <=0:
                            D = eval(f'self.multiscale_discriminator_{current_level+1}')
                            sub_D = eval(f'self.multiscale_discriminator_{current_level}')
                            D.load_state_dict(sub_D.state_dict())
                            if self.opt.one_pg_D:
                                del sub_D
                          
                        for i in range( current_level+1,-1,-1):
                            D = eval(f'self.multiscale_discriminator_{i}')
                            out = D(input)
                            if not get_intermediate_features:
                                out = [out]
                            result.append(out)
                            if self.opt.one_pg_D:
                                break
                            input = self.downsample(input)
        else:
            for i in range(self.opt.num_D):
                D = eval(f'self.multiscale_discriminator_{i}')
                out = D(input)
                if not get_intermediate_features:
                    out = [out]
                result.append(out)
                input = self.downsample(input)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, level=0): # relative resolution level from output size
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf // (2**level)
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = []
        self.level = level
        # self.add_module('toRGB_level' + str(level), nn.Sequential(nn.Conv2d(input_nc, nf, kernel_size=kw,
        #                                         stride=1, padding=padw),
        #                     nn.LeakyReLU(0.2, False)))


        for n in range(level, -1, -1):
            
            exec('self.toRGB_extra%d = nn.Sequential(nn.Conv2d(input_nc, nf, kernel_size=kw,stride=1, padding=padw),nn.LeakyReLU(0.2))' % n)
          
            nf = min(nf * 2, 512)

        nf = opt.ndf // (2**level)
        for n in range(level + opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == level + opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        self.sequence_len = len(sequence)

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(level, 0, -1):
            exec('self.extra%d = nn.Sequential(*sequence[level-n])' % n)
        for n in range(level, len(sequence)):
            exec('self.model%d = nn.Sequential(*sequence[n])' % (n-level))

    def compute_D_input_nc(self, opt):
        input_nc = opt.label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        return input_nc
    

    def execute(self, input, alpha = 0): # id of extra level
        level = self.level
        if alpha > 0:
            low_res = nn.interpolate(input, scale_factor=0.5)
            lowres_toRGB = eval(f'self.toRGB_extra{self.level-1}')
            highres_toRGB = eval(f'self.toRGB_extra{self.level}')
            extra_m = eval(f'self.extra{self.level}')
            level -= 1
            low_res = lowres_toRGB(low_res)
            high_res = extra_m(highres_toRGB(input))
            intermediate_output = low_res*(1-alpha) + high_res*alpha
            results = [intermediate_output]

        else:
            toRGB = eval(f'self.toRGB_extra{self.level}')
            intermediate_output = toRGB(input)
            results = [intermediate_output]

        for i in range(level, 0, -1):
            extra_m = eval(f'self.extra{i}')
            intermediate_output = extra_m(results[-1])
            results.append(intermediate_output)

        for i in range(self.level, self.sequence_len):
            submodel = eval(f'self.model{i-self.level}')
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
