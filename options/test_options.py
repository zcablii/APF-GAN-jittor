from .base_options import BaseOptions
from util.util import str2bool


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--out_path', type=str, default='./results/', help='saves results here.')
        
        parser.add_argument('--which_epoch', type=str, default='avg_273_299', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--use_pure', type=str2bool, default=True, help='use train set image replacement for pure label images')
        parser.add_argument('--seed', type=int, default=-1, help='set random seed')
        parser.set_defaults(use_seg_noise=True)
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        # parser.set_defaults(use_pure=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
