# Test fairness for gender and age
# python test.py --dataset NIHchest --resume '/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/popar_adodocar_448_1/best.pth' --device 1
# python test.py --dataset RSNA --resume '/sda1/zhouziyu/ssl/downstream_checkpoints/RSNAPneumonia/popar_cyclic_448_1/best.pth' --device 6
# python test.py --dataset shenzhenCXR --resume '/sda1/zhouziyu/ssl/downstream_checkpoints/shenzhenCXR/popar_cyclic_448_1/best.pth' --device 0


import argparse
import datetime
import json
import os
import random
import sys
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12345'
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from config import get_config
from configs.config_NIHchest import get_config_NIHchest
from configs.config_RSNA import get_config_RSNA
from configs.config_shenzhenCXR import get_config_shenzhenCXR
from configs.config_CheXpert import get_config_CheXpert
from configs.config_vindrcxr import get_config_vindrcxr
from configs.config_SIIM import get_config_SIIM
from configs.config_CovidQuEx import get_config_CovidQuEx
from data import build_loader
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model
from optimizer import build_optimizer
from sklearn.metrics import roc_auc_score
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter, accuracy
from utils import (NativeScalerWithGradNormCount, auto_resume_helper,
                   load_checkpoint, load_pretrained, reduce_tensor,
                   save_checkpoint)
import ipdb


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default='configs/swin/swin_base_patch4_window7_224.yaml', help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')
    # easy config modification
    parser.add_argument('--matadata', type=str, default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/Data_Entry_2017_v2020.csv', help='fairness')
    parser.add_argument('--backbone', type=str, default='convnext', help='swin_base, swin_large, vit_base, vit_base_patchsize16, vit_huge_patchsize14,resnet50,convnext')
    parser.add_argument('--batch-size', type=int,default=128, help="batch size for single GPU") # default=128 for 224 img_size, 32 for 448 img_size
    parser.add_argument('--dataset', type=str,default='RSNA_robust', help="NIHchest_robust, CheXpert_robust, RSNA_robust, CovidQuEx_robust")
    parser.add_argument('--img_size', type=int, default=224, help='image size of downstream task')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
    parser.add_argument('--mode', type=str, default='test', help='mode: train, val or test')
    parser.add_argument('--augmode', type=str, default='CustomGaussianNoise', help='CustomMetalArtifact,CustomBrightnessEnhance,CustomGaussianNoise')

    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/CheXWorldvit_base_patchsize16_224_1/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/Lamps_large_swin_448_1/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/adam-v2convnext_224_1/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/RAD-DINOvit_base_patchsize16_518_1/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/eva-x/checkpoint-best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/FoundationXswin_base_224_1/best.pth", help='test from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/ark+/Ark+SwinL768_ChestX-ray14_ft.pth.tar", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/Ark_plusswin_large_linearprob_768_2/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/FoundationXswin_224_1/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/Lamps_large_swinv1swin_448_1/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/CheXWorldvit_base_patchsize16_224_1/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/RAD-DINOvit_base_patchsize16_linearprob_518_4/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/adam-v2convnext_linearprob_224_5/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/eva-xvit_base_patchsize16_linearprob_224_4/best.pth", help='resume from checkpoint')
    parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CovidQuEx/adam-v2convnext_linearprob_224_1/best.pth", help='resume from checkpoint')

    parser.add_argument('--pretrain_mode', type=str, default='Adamv2', help='FoundationX,Ark_plus,CheXWorld,Lamps,Adamv2,RAD-DINO,eva-x')
    parser.add_argument('--model_type', type=str,default='swin', help="swin,swinv2")

    parser.add_argument("--device", type=str, default='0')

    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')


    args, unparsed = parser.parse_known_args()

    if args.dataset in ['NIHchest_robust']:
        config = get_config_NIHchest(args)
    elif args.dataset in ['RSNA', 'RSNA_robust']:
        config = get_config_RSNA(args)
    elif args.dataset == 'shenzhenCXR':
        config = get_config_shenzhenCXR(args)
    elif args.dataset in ['CheXpert', 'CheXpert_gender', 'CheXpert_age', 'CheXpert_robust']:
        config = get_config_CheXpert(args)
    elif args.dataset == 'vindrcxr':
        config = get_config_vindrcxr(args)
    elif args.dataset == 'SIIM_cls':
        config = get_config_SIIM(args)
    elif args.dataset == 'CovidQuEx_robust':
        config = get_config_CovidQuEx(args)

    return args, config


def main(config):
    device = torch.device(f'cuda:{config.DEVICE}' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = build_model(config)
    model = model.to(device)
    # ipdb.set_trace()
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu', weights_only=False)

    # model.load_state_dict(checkpoint['model'])
    # state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    try:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
    except:
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    # val_max_auc = checkpoint['max_auc'] # max_auc
    # print(f'val_max_auc: {val_max_auc}')

    
    
    results_path = os.path.dirname(args.resume)
    if os.path.exists(os.path.join(results_path, 'test_result.txt')):
        log_writer = open(os.path.join(results_path, 'test_result.txt'), 'a')
    else:
        log_writer = open(os.path.join(results_path, 'test_result.txt'), 'w')
        
    test_dataset_init, test_dataset_metalartifact, test_dataset_brightnessenhance, test_dataset_gaussiannoise = build_loader(config, dataset = config.DATA.DATASET)
    
    sum = 0
    print(f'Testing augmentation of metalartifact:', file=log_writer)
    print(f'Testing augmentation of metalartifact:')
    for idx, (image, label) in enumerate(test_dataset_metalartifact):
    
        if len(image.shape)==3:
            image = torch.unsqueeze(image, dim=0)
        image = image.to(device) # [10,3,224,224]
        
        # print(image.shape)
        # image = image.unsqueeze(0) # (3,512,512) --> (1,3,512,512)
        logits = model(image).mean(0).sigmoid() # ncrop
        # print(logits.shape)
        # sys.exit(1)
        logits = logits.cpu().detach().numpy()
        if config.DATA.DATASET in ['RSNA']:
            pred = logits.argmax()
            if pred == label.argmax():
                sum+=1

        if idx==0:
            output = logits
            target = label
        else:
            output = np.vstack([output, logits])
            target = np.vstack([target, label])
        print(idx)

    if config.DATA.DATASET in ['RSNA']:
        acc = sum*1.0/(idx+1)
        print(f'acc: {acc}')
        print(f'acc: {acc}', file=log_writer)
    else:
        auc = metric_AUROC(target, output, nb_classes=config.MODEL.NUM_CLASSES)
        print(f'auc: {auc}')
        print(f'auc: {auc}', file=log_writer)
        if config.DATA.DATASET == 'CheXpert_robust':
            print((auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5)
            print((auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5, file=log_writer)
        elif config.DATA.DATASET in ['NIHchest_robust', 'CovidQuEx_robust', 'RSNA_robust']:
            print(np.array(auc).mean())
            print(np.array(auc).mean(), file=log_writer)
        

    sum = 0
    print(f'Testing augmentation of brightnessenhance:', file=log_writer)
    print(f'Testing augmentation of brightnessenhance:')
    for idx, (image, label) in enumerate(test_dataset_brightnessenhance):
            
        if len(image.shape)==3:
            image = torch.unsqueeze(image, dim=0)
        image = image.to(device) # [10,3,224,224]
        
        logits = model(image).mean(0).sigmoid() # ncrop
        logits = logits.cpu().detach().numpy()
        if config.DATA.DATASET in ['RSNA']:
            pred = logits.argmax()
            if pred == label.argmax():
                sum+=1

        if idx==0:
            output = logits
            target = label
        else:
            output = np.vstack([output, logits])
            target = np.vstack([target, label])
        print(idx)
        
    if config.DATA.DATASET in ['RSNA']:
        acc = sum*1.0/(idx+1)
        print(f'acc: {acc}')
        print(f'acc: {acc}', file=log_writer)
    else:
        auc = metric_AUROC(target, output, nb_classes=config.MODEL.NUM_CLASSES)
        print(f'auc: {auc}')
        print(f'auc: {auc}', file=log_writer)
        
        if config.DATA.DATASET == 'CheXpert_robust':
            print((auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5)
            print((auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5, file=log_writer)
        elif config.DATA.DATASET in ['NIHchest_robust', 'CovidQuEx_robust', 'RSNA_robust']:
            print(np.array(auc).mean())
            print(np.array(auc).mean(), file=log_writer)
        
        
    sum = 0
    print(f'Testing augmentation of gaussiannoise:', file=log_writer)
    print(f'Testing augmentation of gaussiannoise:')
    for idx, (image, label) in enumerate(test_dataset_gaussiannoise):
            
        if len(image.shape)==3:
            image = torch.unsqueeze(image, dim=0)
        image = image.to(device) # [10,3,224,224]
        
        # print(image.shape)
        # image = image.unsqueeze(0) # (3,512,512) --> (1,3,512,512)
        logits = model(image).mean(0).sigmoid() # ncrop
        # print(logits.shape)
        # sys.exit(1)
        logits = logits.cpu().detach().numpy()
        if config.DATA.DATASET in ['RSNA']:
            pred = logits.argmax()
            if pred == label.argmax():
                sum+=1

        if idx==0:
            output = logits
            target = label
        else:
            output = np.vstack([output, logits])
            target = np.vstack([target, label])
        print(idx)
        
    if config.DATA.DATASET in ['RSNA']:
        acc = sum*1.0/(idx+1)
        print(f'acc: {acc}')
        print(f'acc: {acc}', file=log_writer)
    else:
        auc = metric_AUROC(target, output, nb_classes=config.MODEL.NUM_CLASSES)
        print(f'auc: {auc}')
        print(f'auc: {auc}', file=log_writer)
        
        if config.DATA.DATASET == 'CheXpert_robust':
            print((auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5)
            print((auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5, file=log_writer)
        elif config.DATA.DATASET in ['NIHchest_robust', 'CovidQuEx_robust', 'RSNA_robust']:
            print(np.array(auc).mean())
            print(np.array(auc).mean(), file=log_writer)



    sum = 0
    print('Testing no augmentation:', file=log_writer)
    print('Testing no augmentation:')
    for idx, (image, label) in enumerate(test_dataset_init):
    
        if len(image.shape)==3:
            image = torch.unsqueeze(image, dim=0)
        image = image.to(device) # [10,3,224,224]
        
        # print(image.shape)
        # image = image.unsqueeze(0) # (3,512,512) --> (1,3,512,512)
        logits = model(image).mean(0).sigmoid() # ncrop
        # print(logits.shape)
        # sys.exit(1)
        logits = logits.cpu().detach().numpy()
        if config.DATA.DATASET in ['RSNA']:
            pred = logits.argmax()
            if pred == label.argmax():
                sum+=1

        if idx==0:
            output = logits
            target = label
        else:
            output = np.vstack([output, logits])
            target = np.vstack([target, label])
        print(idx)

    if config.DATA.DATASET in ['RSNA']:
        acc = sum*1.0/(idx+1)
        print(f'acc: {acc}')
        print(f'acc: {acc}', file=log_writer)
    else:
        auc = metric_AUROC(target, output, nb_classes=config.MODEL.NUM_CLASSES)
        print(f'auc: {auc}')
        print(f'auc: {auc}', file=log_writer)
        
        if config.DATA.DATASET == 'CheXpert_robust':
            print((auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5)
            print((auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5, file=log_writer)
        elif config.DATA.DATASET in ['NIHchest_robust', 'CovidQuEx_robust', 'RSNA_robust']:
            print(np.array(auc).mean())
            print(np.array(auc).mean(), file=log_writer)




    


def metric_AUROC(target, output, nb_classes=14):
    """
    Return AUROC for each class.
    If a class has no positive or no negative samples,
    return NaN for that class.
    """

    outAUROC = []

    for i in range(nb_classes):
        try:
            auc = roc_auc_score(target[:, i], output[:, i])
        except ValueError:
            auc = np.nan   # ⚠️ 关键：占位而不是跳过

        outAUROC.append(auc)

    return outAUROC


if __name__=='__main__':
    args, config = parse_option()
    main(config)