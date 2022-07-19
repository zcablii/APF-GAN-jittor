from models.pix2pix_model import Pix2PixModel
import jittor as jt

class Pix2PixTrainer():

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        self.generated = None
        if opt.isTrain:
            (self.optimizer_G, self.optimizer_D) = self.pix2pix_model.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, data, epoch):
        self.optimizer_G.zero_grad()
        (g_losses, generated) = self.pix2pix_model(data, epoch, mode='generator')
        g_loss = sum(g_losses.values()).mean().float_auto()
        self.optimizer_G.step(g_loss)
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data, epoch):
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(data, epoch, mode='discriminator')
        d_loss = sum(d_losses.values()).mean().float_auto()
        self.optimizer_D.step(d_loss)
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model.save(epoch)

    def update_learning_rate(self, epoch):
        if ((self.opt.pg_strategy != 0) and ((epoch % (self.opt.pg_niter // (self.opt.num_D - 1))) == 0) and (epoch < (self.opt.pg_niter + 1))):
            new_lr = (self.old_lr * self.opt.pg_lr_decay)
        if (epoch > self.opt.niter):
            if ((epoch - 1) == self.opt.niter):
                self.opt.lr = self.old_lr
            lrd = (self.opt.lr / self.opt.niter_decay)
            new_lr = (self.old_lr - lrd)
        else:
            new_lr = self.old_lr
        if (new_lr != self.old_lr):
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = (new_lr / 2)
                new_lr_D = (new_lr * 2)
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print(('update learning rate: %f -> %f' % (self.old_lr, new_lr)))
            self.old_lr = new_lr

