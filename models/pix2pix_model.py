import jittor as jt
from jittor import init
from jittor import nn
import models.networks as networks
import util.util as util
from util.util import DiffAugment

class Pix2PixModel(nn.Module):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = jt.float16
        self.netG, self.netD, self.netE = self.initialize_networks(opt)
        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = nn.L1Loss()
            if not opt.no_vgg_loss:
                if opt.inception_loss:
                    self.criterionVGG = networks.InceptionLoss(self.opt.gpu_ids)
                else:
                    self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def execute(self, data, epoch = 0, mode = None):
        input_semantics, real_image = self.preprocess_input(data)
        # print('execute data is ok', input_semantics[0][0][0])

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, epoch)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, epoch)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with jt.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = jt.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = jt.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)
        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].float_auto()
        
        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
            
        input_label = jt.zeros((bs,nc,h,w)).float_auto()

        input_semantics = input_label.scatter_(1, label_map, jt.array(1.0).float_auto())
 
        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = jt.contrib.concat((input_semantics, instance_edge_map), dim=1)
        
        return jt.float_auto(input_semantics), jt.float_auto(data['image'])
        

    def compute_generator_loss(self, input_semantics, real_image, epoch):

        G_losses = {}
        (fake_image, KLD_loss) = self.generate_fake(input_semantics, real_image, epoch, compute_kld_loss=self.opt.use_vae)
        if (self.opt.use_vae and (KLD_loss is not None)):
            G_losses['KLD'] = KLD_loss
        (pred_fake, pred_real) = self.discriminate(input_semantics, fake_image, real_image, epoch)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        if (not self.opt.no_ganFeat_loss):
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(0)
            for i in range(num_D):
                num_intermediate_outputs = (len(pred_fake[i]) - 1)
                for j in range(num_intermediate_outputs):
                    unweighted_loss = jt.abs(jt.float32(pred_fake[i][j]-pred_real[i][j].detach())).mean().float_auto()
            
                    GAN_Feat_loss += ((unweighted_loss * self.opt.lambda_feat) / num_D)
        
            G_losses['GAN_Feat'] = GAN_Feat_loss
        if (not self.opt.no_vgg_loss):
            if (type(fake_image) == list):
                fake_image = fake_image[(- 1)]
            img_shape = fake_image.shape[(- 2):]
            real_image = nn.interpolate(real_image, img_shape)
            if self.opt.inception_loss:
                G_losses['Inception'] = (self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg)
            else:
                G_losses['VGG'] = (self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg)
        return (G_losses, fake_image)

    def compute_discriminator_loss(self, input_semantics, real_image, epoch):
        D_losses = {}
        with jt.no_grad():
            (fake_image, _) = self.generate_fake(input_semantics, real_image, epoch)
            if (type(fake_image) == list):
                with jt.enable_grad():
                    fake_image = [fake_img.detach()for fake_img in fake_image]
            else:
                with jt.enable_grad():
                    fake_image = fake_image.detach()
        (pred_fake, pred_real) = self.discriminate(input_semantics, fake_image, real_image, epoch)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)
        return D_losses

    def encode_z(self, real_image):
        (mu, logvar) = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return (z, mu, logvar)

    def encode_m(self, mask):
        z = self.netE(mask)

        return z

    def generate_fake(self, input_semantics, real_image, epoch=0, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            if self.opt.encode_mask:
                z = self.encode_m(input_semantics)
            else:
                (z, mu, logvar) = self.encode_z(real_image)
                if compute_kld_loss:
                    KLD_loss = (self.KLDLoss(mu, logvar) * self.opt.lambda_kld)

        fake_image = self.netG(input_semantics, epoch, z=z)
        assert ((not compute_kld_loss) or self.opt.use_vae), 'You cannot compute KLD loss if opt.use_vae == False'
        return (fake_image, KLD_loss)

    def discriminate(self, input_semantics, fake_image, real_image, epoch):
        if (not (type(fake_image) == list)):
            if (len(self.opt.diff_aug) > 0):
                (real_image, fake_image, input_semantics) = DiffAugment(real_image, fake_image, input_semantics, policy=self.opt.diff_aug)
            fake_concat = jt.contrib.concat([input_semantics, fake_image], dim=1)
            real_concat = jt.contrib.concat([input_semantics, real_image], dim=1)
            fake_and_real = jt.contrib.concat([fake_concat, real_concat], dim=0)
        else:
            fake_concat = []
            real_concat = []
            for i in range((len(fake_image) - 1), (- 1), (- 1)):
                if (len(self.opt.diff_aug) > 0):
                    (generated_image, real_image, input_semantics) = DiffAugment(fake_image[i], real_image, input_semantics, policy=self.opt.diff_aug)
                else:
                    img_shape = fake_image[i].shape[(- 2):]
                    input_semantics = nn.interpolate(input_semantics, img_shape)
                    real_image = nn.interpolate(real_image, img_shape)
                    generated_image = fake_image[i]
                fake_concat.append(jt.contrib.concat([input_semantics, generated_image], dim=1))
                real_concat.append(jt.contrib.concat([input_semantics, real_image], dim=1))
            fake_and_real = []
            for i in range(len(fake_concat)):
                fake_and_real.append(jt.contrib.concat([fake_concat[i], real_concat[i]], dim=0))
   
        discriminator_out = self.netD(fake_and_real, epoch)
        (pred_fake, pred_real) = self.divide_pred(discriminator_out)
        return (pred_fake, pred_real)

    def divide_pred(self, pred):
        if (type(pred) == list):
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:(tensor.shape[0] // 2)] for tensor in p])
                real.append([tensor[(tensor.shape[0] // 2):] for tensor in p])
        else:
            fake = pred[:(pred.shape[0] // 2)]
            real = pred[(pred.shape[0] // 2):]
        return (fake, real)

    def get_edges(self, t):
        edge = self.FloatTensor(t.shape).zero_()
        edge[:, :, :, 1:] = (edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :(- 1)]))
        edge[:, :, :, :(- 1)] = (edge[:, :, :, :(- 1)] | (t[:, :, :, 1:] != t[:, :, :, :(- 1)]))
        edge[:, :, 1:, :] = (edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :(- 1), :]))
        edge[:, :, :(- 1), :] = (edge[:, :, :(- 1), :] | (t[:, :, 1:, :] != t[:, :, :(- 1), :]))
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = jt.exp((0.5 * logvar))
        eps = jt.randn_like(std)
        return (eps.mul(std) + mu)

    def use_gpu(self):
        return (len(self.opt.gpu_ids) > 0)

