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
    parser.add_argument('--backbone', type=str, default='vit_base_patchsize16', help='swin_base, swin_large, vit_base, vit_base_patchsize16, vit_huge_patchsize14,resnet50,convnext')
    parser.add_argument('--batch-size', type=int,default=128, help="batch size for single GPU") # default=128 for 224 img_size, 32 for 448 img_size
    parser.add_argument('--dataset', type=str,default='NIHchest_gender', help="NIHchest_gender, NIHchest_age, CheXpert_gender, CheXpert_age")
    parser.add_argument('--img_size', type=int, default=224, help='image size of downstream task')
    parser.add_argument('--num_classes', type=int, default=14, help='number of classes')
    parser.add_argument('--mode', type=str, default='test', help='mode: train, val or test')

    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/CheXWorldvit_base_patchsize16_linearprob_224_1/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/Lamps_large_swin_448_1/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/adam-v2convnext_224_1/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/RAD-DINOvit_base_patchsize16_518_1/best.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/eva-x/checkpoint-best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/FoundationXswin_base_224_1/best.pth", help='test from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/Ark_plusswin_large_linearprob_768_2/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/Ark_plusswin_large_linearprob_768_5/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/FoundationXswin_base_linearprob_224_5/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/Lamps_large_swinv1swin_448_1/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/CheXWorldvit_base_patchsize16_linearprob_224_1/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/RAD-DINOvit_base_patchsize16_linearprob_518_1/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/adam-v2convnext_224_1/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/EVA-X/checkpoint-best.pth", help='resume from checkpoint')

    parser.add_argument('--pretrain_mode', type=str, default='eva-x', help='FoundationX,Ark_plus,CheXWorld,Lamps,Adamv2,RAD-DINO,eva-x')
    parser.add_argument('--model_type', type=str,default='swin', help="swin,swinv2")

    parser.add_argument("--device", type=str, default='0')

    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')


    args, unparsed = parser.parse_known_args()

    if args.dataset in ['NIHchest', 'NIHchest_gender', 'NIHchest_age']:
        config = get_config_NIHchest(args)
    elif args.dataset == 'RSNA':
        config = get_config_RSNA(args)
    elif args.dataset == 'shenzhenCXR':
        config = get_config_shenzhenCXR(args)
    elif args.dataset in ['CheXpert', 'CheXpert_gender', 'CheXpert_age']:
        config = get_config_CheXpert(args)
    elif args.dataset == 'vindrcxr':
        config = get_config_vindrcxr(args)
    elif args.dataset == 'SIIM_cls':
        config = get_config_SIIM(args)

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

    # 计算除去最后分类层的参数量
    total_params = sum(p.numel() for p in model.parameters())
    # 假设最后分类层为 model.head 或 model.fc 或 model.classifier，根据实际模型结构修改
    classifier_layer = None
    for name in ['head', 'fc', 'classifier']:
        if hasattr(model, name):
            classifier_layer = getattr(model, name)
            break
    if classifier_layer is not None:
        classifier_params = sum(p.numel() for p in classifier_layer.parameters())
    else:
        classifier_params = 0
        print("未找到分类层，参数量未减去分类层。请检查模型结构。")
    params_without_classifier = total_params - classifier_params
    params_M = params_without_classifier / 1e6
    print(f"除去分类层的参数量: {params_M:.2f}M")





if __name__=='__main__':
    args, config = parse_option()
    main(config)