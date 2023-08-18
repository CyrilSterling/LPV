import torch
import numpy as np
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TokenLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, opt):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.opt = opt
        self.SPACE = '[s]'
        self.GO = '[GO]'

        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(opt.character)

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.max_length = opt.max_length + 2

    def encode(self, text):
        """ convert text-label into text-index.
        """
        batch_text = torch.LongTensor(len(text), self.max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)  # batch_text[:, 0] = [GO] token
        return batch_text.to(device)
    
    def char_encode(self, text):
        """ convert text-label into text-index.
        """
        batch_len = torch.LongTensor(len(text), 1).fill_(self.dict[self.GO])
        batch_text = torch.LongTensor(len(text), self.max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            length = len(t) 
            batch_len[i][0] = torch.LongTensor([length])
            
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)
            
        return batch_len.to(device), batch_text.to(device)

    def char_decode(self, text_index):
        """ convert text-index into text-label. """
        texts = []
        for index in range(text_index.shape[0]):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
    

class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def get_device(verbose=True):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if verbose:
        print("Device:", device)
    return device


def get_args(is_train=True):
    parser = argparse.ArgumentParser(description='STR')

    # for test
    parser.add_argument('--eval_data', help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--show', default='None', help='', choices=['None', 'image', 'attn'])
    parser.add_argument('--no_debug', action='store_true', help='debug mode')
    parser.add_argument('--fast_acc', action='store_true', help='Fast average accuracy computation')

    # for train
    parser.add_argument('--exp_name', default='svtr-tiny-exp', help='Where to store logs and models')
    parser.add_argument('--train_data', default='./dataset', help='path to training dataset')
    parser.add_argument('--valid_data', default='./dataset/evaluation', help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=226, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers. Use -1 to use all cores.', default=12)
    parser.add_argument('--batch_size', type=int, default=96, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=413940, help='number of iterations to train for')
    parser.add_argument('--drop_iter', type=int, default=240000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=1000, help='Interval between each validation')
    parser.add_argument('--show_iter', type=int, default=50, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_path', default='./workdir/', help="path to save")
    parser.add_argument('--char_dic', default='', help="path to save")
    
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0001 for Adam')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='decay rate beta2 for adam. default=0.99')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=20, help='gradient clipping value. default=5')

    # scheduler
    parser.add_argument('--scheduler', action='store_true', help='Use lr scheduler')
    
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', default=True, help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    
    """ Model Architecture """
    choices = ["svtr_tiny", "svtr_small", "svtr_base"]
    parser.add_argument('--backbone', type=str, default='svtr_tiny', help='')
    parser.add_argument('--attention_mode', type=str, default=None, help='')
    parser.add_argument('--d_model', type=int, default=256, help='')
    parser.add_argument('--nhead', type=int, default=8, help='')
    parser.add_argument('--d_inner', type=int, default=1024, help='')
    parser.add_argument('--dropout', type=int, default=0.1, help='')
    parser.add_argument('--activation', type=str, default='gelu', help='')
    parser.add_argument('--trans_ln', type=int, default=2, help='')
    parser.add_argument('--attn', type=int, default=3, help='')
    parser.add_argument('--fuse_mask', action='store_true', help='use diagmask and charmask together')
    parser.add_argument('--mask', action='store_true', help='use diagmask and charmask together')
    
    # selective augmentation 
    # can choose specific data augmentation
    parser.add_argument('--issel_aug', action='store_true', help='Select augs')
    parser.add_argument('--sel_prob', type=float, default=1., help='Probability of applying augmentation')
    parser.add_argument('--pattern', action='store_true', help='Pattern group')
    parser.add_argument('--warp', action='store_true', help='Warp group')
    parser.add_argument('--geometry', action='store_true', help='Geometry group')
    parser.add_argument('--weather', action='store_true', help='Weather group')
    parser.add_argument('--noise', action='store_true', help='Noise group')
    parser.add_argument('--blur', action='store_true', help='Blur group')
    parser.add_argument('--camera', action='store_true', help='Camera group')
    parser.add_argument('--process', action='store_true', help='Image processing routines')

    parser.add_argument('--intact_prob', type=float, default=0.5, help='Probability of not applying augmentation')
    parser.add_argument('--isrand_aug', action='store_true', default=True, help='Use RandAug')
    parser.add_argument('--augs_num', type=int, default=3, help='Number of data augment groups to apply. 1 to 8.')
    parser.add_argument('--augs_mag', type=int, default=None, help='Magnitude of data augment groups to apply. None if random.')

    # for comparison to other augmentations
    parser.add_argument('--issemantic_aug', action='store_true', help='Use Semantic')
    parser.add_argument('--isrotation_aug', action='store_true', help='Use ')
    parser.add_argument('--isscatter_aug', action='store_true', help='Use ')
    parser.add_argument('--islearning_aug', action='store_true', help='Use ')

    # dist train
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # for eval
    parser.add_argument('--eval_img', action='store_true', help='eval imgs dataset')
    parser.add_argument('--model_dir', default='') 
    parser.add_argument('--demo_imgs', default='')
    
    args = parser.parse_args()
    return args
