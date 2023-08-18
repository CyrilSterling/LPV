import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
import random
import string
import argparse
import re

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np
import copy

from utils import Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from models import Model
from test_final import validation
from utils import get_args
import utils_dist as utils

import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import datetime
def format_time(time):
    days = 0
    if time > 86400:
        days = int(time // 86400)
        time = time - days*86400
    date = datetime.datetime.utcfromtimestamp(time)
    output = datetime.datetime.strftime(date, "%H:%M:%S")
    return f'{days:d}days {output}' if days > 0 else output
    
class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, softmax=True):
        if softmax: log_prob = F.log_softmax(input, dim=-1)
        else: log_prob = torch.log(input)
        loss = -(target * log_prob).sum(dim=-1)
        if self.reduction == "mean": return loss.mean()
        elif self.reduction == "sum": return loss.sum()
        else: return loss

def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    opt.eval = False
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'{opt.saved_path}/{opt.exp_name}/log_dataset.txt', 'a')

    val_opt = copy.deepcopy(opt)
    val_opt.eval = True
    
    if opt.sensitive:
        opt.data_filtering_off = True
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=val_opt)
    valid_dataset, _ = hierarchical_dataset(root=opt.valid_data, opt=val_opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
        
    
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
        
    """ model configuration """
    converter = TokenLabelConverter(opt)
        
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)

    print(model)

    # data parallel for multi-GPU
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], find_unused_parameters=False)
    
    model.train()
    
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        model.load_state_dict(torch.load(opt.saved_model, map_location='cpu'), strict=True)

    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
        
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    # setup optimizer
    optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, opt.beta2))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.drop_iter, gamma=0.1)

    """ final options """
    with open(f'{opt.saved_path}/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        #print(opt_log)
        opt_file.write(opt_log)
        total_params = int(sum(params_num))
        total_params = f'Trainable network params num : {total_params:,}\n'
        print(total_params)
        opt_file.write(total_params)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    best_accuracy = -1
    iteration = start_iter
            
    print("LR",scheduler.get_last_lr()[0])
        
    start_time = time.time()
    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        
        len_target, char_target = converter.char_encode(labels)

        char_preds = model(image)
        
        it = char_preds.shape[0] // char_target.shape[0]
        char_target = char_target.repeat(it, 1)
        
        char_preds = char_preds.view(-1, char_preds.shape[-1])
        
        char_loss = criterion(char_preds, char_target.contiguous().view(-1))
        cost = char_loss 

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 20 (Default)
        optimizer.step()

        loss_avg.add(cost)

        if utils.is_main_process() and ((iteration + 1) % opt.show_iter == 0):
            remain_time = format_time((time.time() - start_time) * (opt.num_iter - iteration - 1) / (iteration+1-start_iter))
            with open(f'{opt.saved_path}/{opt.exp_name}/log_train.txt', 'a') as log:
                loss_log = f'[{iteration+1}/{opt.num_iter}]\tLR: {scheduler.get_last_lr()[0]:0.5f}, \tTrain loss: {loss_avg.val():0.5f}, \tRemain Time: {remain_time}'
                print(loss_log)
                log.write(loss_log + '\n')

        # validation part
        if utils.is_main_process() and ((iteration + 1) % opt.valInterval == 0):
            # for log
            print("LR",scheduler.get_last_lr()[0])
            with open(f'{opt.saved_path}/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                
                with torch.no_grad():
                    valid_loss, current_accuracy, char_preds, confidence_score, labels, length_of_data, _ = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                loss_log = f'[{iteration+1}/{opt.num_iter}] LR: {scheduler.get_last_lr()[0]:0.5f}, Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}'
                loss_avg.reset()
                current_model_log = f'{"Word_accuracy":17s}: {current_accuracy:0.3f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'{opt.saved_path}/{opt.exp_name}/best_accuracy.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], char_preds[:5], confidence_score[:5]):
                    pred = pred[:pred.find('[s]')]

                    # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                    if opt.sensitive and opt.data_filtering_off:
                        pred = pred.lower()
                        gt = gt.lower()
                        alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                        out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                        pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                        gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if utils.is_main_process() and (iteration + 1) % 5e+3 == 0:
            # torch.save(
            #     model.state_dict(), f'{opt.saved_path}/{opt.exp_name}/iter_{iteration+1}.pth')
            pass

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1
        if scheduler is not None:
            scheduler.step()

if __name__ == '__main__':

    opt = get_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.backbone}-trans{opt.trans_ln}'

    opt.exp_name += f'-Seed{opt.manualSeed}'

    os.makedirs(f'{opt.saved_path}/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    if opt.char_dic != '':
        character = ''
        with open(opt.char_dic, 'r') as f:
            chars = f.readlines()
        for char in chars:
            character += char.split('\n')[0]
        opt.character = character
    
    utils.init_distributed_mode(opt)

    print(opt)
    
    """ Seed and GPU setting """
    
    seed = opt.manualSeed + utils.get_rank()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    train(opt)

