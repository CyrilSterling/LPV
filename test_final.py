import os
import string
import argparse
import re
import PIL
import validators

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from matplotlib import pyplot as plt
from matplotlib import colors
import cv2
from torchvision import transforms
import torchvision.utils as vutils

from utils import Averager, TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, ImgDataset
from models import Model
from utils import get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def benchmark_all_eval(model, criterion, converter, opt):
    """ evaluation with 10 benchmark evaluation datasets """

    if opt.fast_acc:
    # # To easily compute the total accuracy of six benchmarks.
        eval_data_list = ['IC13_857', 'SVT', 'IIIT5k', 'SVTP', 'IC15_1811', 'CUTE80']
    else:
        eval_data_list = ['IC13_857', 'IC13_1015', 'SVT', 'IIIT5k', 'SVTP', 'IC15_1811', 'IC15_2077', 'CUTE80']

    word_list_accuracy = []
    total_evaluation_data_number = 0
    word_total_correct_number = 0
    log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
    dashed_line = '-' * 80
    print(dashed_line)
    log.write(dashed_line + '\n')
    for eval_data in eval_data_list:

        if opt.eval_img:
            eval_data_path = os.path.join(opt.eval_data, eval_data+'.txt')
            eval_data = ImgDataset(root=eval_data_path, opt=opt)
        else:
            eval_data_path = os.path.join(opt.eval_data, eval_data)
            print(eval_data_path)
            eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)

        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy, _, _, _, length_of_data, accur_numbers = validation(
            model, criterion, evaluation_loader, converter, opt)
        word_list_accuracy.append(f'{accuracy:0.3f}')

        total_evaluation_data_number += len(eval_data)
        word_total_correct_number += accur_numbers
        print(f'Word_Acc {accuracy:0.3f}')
        log.write(f'Word_Acc {accuracy:0.3f}')
        print(dashed_line)
        log.write(dashed_line + '\n')

    word_total_accuracy = round(word_total_correct_number/total_evaluation_data_number*100,3)
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: ' + '\n'
    evaluation_log += 'word_total_Acc:'+str(word_total_accuracy)+'\n'
    evaluation_log += f'# parameters: {params_num/1e6:0.3f}'
    print(evaluation_log)
    log.write(evaluation_log + '\n')
    log.close()

    return word_total_accuracy


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    word_n_correct = 0

    length_of_data = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels, imgs_path) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        if opt.backbone:
            _, target = converter.char_encode(labels)
        else:
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
            text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
        
        if opt.no_debug:
            word_preds = model(image, is_eval=True)
        else:
            attens, word_preds, self_attns = model(image, is_eval=True)

        it = len(word_preds)
        if opt.no_debug:
            word_preds_ = word_preds.view(-1, word_preds.shape[-1])
        else:
            word_preds_ = word_preds[-1].view(-1, word_preds[-1].shape[-1])
            word_preds = word_preds[-1]
        cost = criterion(word_preds_, target.contiguous().view(-1))
        
        max_length = opt.max_length+2
        # pred
        _, word_pred_index = word_preds.topk(1, dim=-1, largest=True, sorted=True)
        word_pred_index = word_pred_index.view(-1, max_length)

        word_preds_str = converter.char_decode(word_pred_index[:,1:])
        word_pred_prob = F.softmax(word_preds, dim=2)
        word_pred_max_prob, _ = word_pred_prob.max(dim=2)
        word_preds_max_prob = word_pred_max_prob[:,1:]
            
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        confidence_score_list = []
        if not opt.no_debug:
            attens = torch.stack(attens).detach().cpu() if attens is not None else None
            self_attns = self_attns.detach().cpu() if self_attns is not None else None
        for index,gt in enumerate(labels):
            word_pred = word_preds_str[index]
            word_pred_max_prob = word_preds_max_prob[index]
            word_pred_EOS = word_pred.find('[s]')
            word_pred = word_pred[:word_pred_EOS]  # prune after "end of sentence" token ([s])
            if word_pred == gt:
                word_n_correct += 1
            word_pred_max_prob = word_pred_max_prob[:word_pred_EOS+1]
            try:
                word_confidence_score = word_pred_max_prob.cumprod(dim=0)[-1]
            except:
                word_confidence_score = 0.0

            confidence_score_list.append(word_confidence_score)
            
            ## draw pic and attn
            if opt.show == 'image':
                os.makedirs(f'./result/{opt.exp_name}/attn', exist_ok=True)
                vutils.save_image([image_tensors[index]], f'./result/{opt.exp_name}/attn/{char_pred}_{gt}_{char_confidence_score:0.3f}.jpg', nrow=1, normalize=True, scale_each=True)
            elif 'attn' in opt.show:
                os.makedirs(f'./result/{opt.exp_name}/attn', exist_ok=True)
                pil = transforms.ToPILImage()
                tensor = transforms.ToTensor()
                resize = transforms.Resize(size=(32,100), interpolation=0)
                
                save_image = draw_atten(image_tensors[index], gt, char_pred, attens[:,index], pil, tensor, resize)
                vutils.save_image(save_image, f'./result/{opt.exp_name}/attn/{i*batch_size+index}_{char_pred}_{gt}_{char_confidence_score:0.3f}.jpg', nrow=6, normalize=True, scale_each=True)
                
    word_accuracy = word_n_correct/float(length_of_data) * 100

    return valid_loss_avg.val(), word_accuracy, word_preds_str, confidence_score_list, labels, length_of_data, word_n_correct

def draw_atten(image, gt, pred, attn, pil, tensor, resize):
    image_np = np.array(pil(image))
    save_image = []

    for i in range(3):
        attn_pil = [pil(a) for a in attn[i, ...]]
        attns = [tensor(resize(a)).repeat(3, 1, 1) for a in attn_pil]
        attn_sum = np.array([np.array(a) for a in attn_pil[:len(gt)]]).sum(axis=0)
        blended_sum = tensor(blend_mask(image_np, attn_sum))
        blended = [tensor(blend_mask(image_np, np.array(a))) for a in attn_pil]
        save_image += [image] + attns  + [blended_sum] + blended
    save_image = torch.stack(save_image)
    save_image = save_image.view(6, -1, *save_image.shape[1:])
    save_image = save_image.permute(1, 0, 2, 3, 4).flatten(0, 1)
    return save_image

def blend_mask(image, mask, alpha=0.5, cmap='jet', color='b', color_alpha=1.0):
    # normalize mask
    mask = (mask-mask.min()) / (mask.max() - mask.min() + np.finfo(float).eps)
    if mask.shape != image.shape:
        mask = cv2.resize(mask,(image.shape[1], image.shape[0]))
    # get color map
    color_map = plt.get_cmap(cmap)
    mask = color_map(mask)[:,:,:3]
    # convert float to uint8
    mask = (mask * 255).astype(dtype=np.uint8)

    # set the basic color
    basic_color = np.array(colors.to_rgb(color)) * 255 
    basic_color = np.tile(basic_color, [image.shape[0], image.shape[1], 1]) 
    basic_color = basic_color.astype(dtype=np.uint8)
    # blend with basic color
    blended_img = cv2.addWeighted(image, color_alpha, basic_color, 1-color_alpha, 0)
    # blend with mask
    blended_img = cv2.addWeighted(blended_img, alpha, mask, 1-alpha, 0)

    return blended_img
    

def test(opt):
    """ model configuration """
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)
    
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    if validators.url(opt.saved_model):
        model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

    """ setup loss """
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    opt.eval = True
    with torch.no_grad():
        if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
            return benchmark_all_eval(model, criterion, converter, opt)
        else:
            log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a')
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
            eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=True)
            _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
                model, criterion, evaluation_loader, converter, opt)
            log.write(eval_data_log)
            print(f'{accuracy_by_best_model[0]:0.3f}')
            log.write(f'{accuracy_by_best_model[0]:0.3f}\n')
            log.close()


if __name__ == '__main__':
    opt = get_args(is_train=False)

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

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    
    opt.saved_model = opt.model_dir
    test(opt)
