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
    parser.add_argument('--backbone', type=str, default='convnext', help='swin_base, swin_large, vit_base, vit_base_patchsize16, vit_huge_patchsize14,resnet50,convnext')
    parser.add_argument('--batch-size', type=int,default=128, help="batch size for single GPU") # default=128 for 224 img_size, 32 for 448 img_size
    parser.add_argument('--dataset', type=str,default='CheXpert_race', help="NIHchest_gender, NIHchest_age, CheXpert_gender, CheXpert_age, CheXpert_race")
    parser.add_argument('--img_size', type=int, default=224, help='image size of downstream task')
    parser.add_argument('--num_classes', type=int, default=14, help='number of classes')
    parser.add_argument('--mode', type=str, default='test', help='mode: train, val or test')

    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/CheXWorldvit_base_patchsize16_linearprob_224_1/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/Lamps_large_swin_448_1/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/adam-v2convnext_224_1/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/RAD-DINOvit_base_patchsize16_518_1/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/eva-xvit_base_patchsize16_linearprob_224_1/best.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/FoundationXswin_base_224_1/best.pth", help='test from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/Ark_plusswin_large_linearprob_768_2/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/Ark_plusswin_large_linearprob_768_5/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/FoundationXswin_base_linearprob_224_1/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/Lamps_large_swinv1swin_linearprob_448_5/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/CheXWorldvit_base_patchsize16_linearprob_224_5/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/RAD-DINOvit_base_patchsize16_linearprob_518_5/best.pth", help='resume from checkpoint')
    parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/adam-v2convnext_linearprob_224_5/best.pth", help='resume from checkpoint')
    # parser.add_argument('--resume', default="/sda/zhouziyu/ssl/downstream_checkpoints/CheXpert/eva-xvit_base_patchsize16_linearprob_224_1/best.pth", help='resume from checkpoint')

    parser.add_argument('--pretrain_mode', type=str, default='Adamv2', help='FoundationX,Ark_plus,CheXWorld,Lamps,Adamv2,RAD-DINO,eva-x')
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
    elif args.dataset in ['CheXpert', 'CheXpert_gender', 'CheXpert_age', 'CheXpert_race']:
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

    
    sum = 0
    results_path = os.path.dirname(args.resume)
    if os.path.exists(os.path.join(results_path, 'test_result.txt')):
        log_writer = open(os.path.join(results_path, 'test_result.txt'), 'a')
    else:
        log_writer = open(os.path.join(results_path, 'test_result.txt'), 'w')
    
    
    if config.DATA.DATASET == 'NIHchest_gender':
    
        test_dataset_M, test_loader_M, test_dataset_F, test_loader_F = build_loader(config, dataset = config.DATA.DATASET)
    
        print('Testing male:', file=log_writer)
        print('Testing male:')
        for idx, (image, label) in enumerate(test_dataset_M):
        
            if len(image.shape)==3:
                image = torch.unsqueeze(image, dim=0)
            image = image.to(device) # [10,3,224,224]
            
            # print(image.shape)
            # image = image.unsqueeze(0) # (3,512,512) --> (1,3,512,512)
            logits = model(image).mean(0).sigmoid() # ncrop
            # print(logits.shape)
            # sys.exit(1)
            logits = logits.cpu().detach().numpy()

            if idx==0:
                output_m = logits
                target_m = label
            else:
                output_m = np.vstack([output_m, logits])
                target_m = np.vstack([target_m, label])
            print(idx)




        auc_m = metric_AUROC(target_m, output_m, nb_classes=config.MODEL.NUM_CLASSES)
        print(f'auc (male): {auc_m}')
        print(f'auc (male): {auc_m}', file=log_writer)
        if config.DATA.DATASET == 'CheXpert':
            print((auc_m[2]+auc_m[5]+auc_m[6]+auc_m[8]+auc_m[10])/5)
            print((auc_m[2]+auc_m[5]+auc_m[6]+auc_m[8]+auc_m[10])/5, file=log_writer)
        print(np.array(auc_m).mean())
        print(np.array(auc_m).mean(), file=log_writer)



        print('Testing female:', file=log_writer)
        print('Testing female:')
        for idx, (image, label) in enumerate(test_dataset_F):
        
            if len(image.shape)==3:
                image = torch.unsqueeze(image, dim=0)
            image = image.to(device) # [10,3,224,224]
            
            # print(image.shape)
            # image = image.unsqueeze(0) # (3,512,512) --> (1,3,512,512)
            logits = model(image).mean(0).sigmoid() # ncrop
            # print(logits.shape)
            # sys.exit(1)
            logits = logits.cpu().detach().numpy()


            if idx==0:
                output_f = logits
                target_f = label
            else:
                output_f = np.vstack([output_f, logits])
                target_f = np.vstack([target_f, label])
            print(idx)


        auc_f = metric_AUROC(target_f, output_f, nb_classes=config.MODEL.NUM_CLASSES)
        print(f'auc (female): {auc_f}')
        print(f'auc (female): {auc_f}', file=log_writer)
        print(np.array(auc_f).mean())
        print(np.array(auc_f).mean(), file=log_writer)

        # compute DEOdds across male/female
        try:
            outputs_all = np.vstack([output_m, output_f])
            targets_all = np.vstack([target_m, target_f])
        except Exception:
            outputs_all = output_m if output_f.size==0 else output_f
            targets_all = target_m if target_f.size==0 else target_f
        demographics = [{'sex': 'M'} for _ in range(output_m.shape[0])] + [{'sex': 'F'} for _ in range(output_f.shape[0])]
        thresholds = [0.5] * config.MODEL.NUM_CLASSES
        group_values = ['M', 'F']
        deodds_per_class, deodds_mean = compute_deodds(targets_all, outputs_all, demographics, thresholds, config.MODEL.NUM_CLASSES, 'sex', group_values)
        print(f'DEOdds per class (sex): {deodds_per_class}')
        print(f'DEOdds mean (sex): {deodds_mean}')
        print(f'DEOdds per class (sex): {deodds_per_class}', file=log_writer)
        print(f'DEOdds mean (sex): {deodds_mean}', file=log_writer)
        
        
    elif config.DATA.DATASET == 'NIHchest_age':
        age_loaders = build_loader(config, dataset = config.DATA.DATASET)

        outputs_all_age = []
        targets_all_age = []
        demographics_all_age = []

        for age_group, (dataset_age, loader) in age_loaders.items():
            print(f"Infer on age group: {age_group}")
            print(f"Infer on age group: {age_group}", file=log_writer)
            
            # ipdb.set_trace()
            for idx, (image, label) in enumerate(dataset_age):
        
                if len(image.shape)==3:
                    image = torch.unsqueeze(image, dim=0)
                image = image.to(device) # [10,3,224,224]
                
                # print(image.shape)
                # image = image.unsqueeze(0) # (3,512,512) --> (1,3,512,512)
                logits = model(image).mean(0).sigmoid() # ncrop
                # print(logits.shape)
                # sys.exit(1)
                logits = logits.cpu().detach().numpy()
                

                if idx==0:
                    output = logits
                    target = label
                else:
                    output = np.vstack([output, logits])
                    target = np.vstack([target, label])
                print(idx)

                # append group results for DEOdds
            outputs_all_age.append(output)
            targets_all_age.append(target)
            demographics_all_age += [{ 'age': age_group } for _ in range(output.shape[0])]

            auc = metric_AUROC(target, output, nb_classes=config.MODEL.NUM_CLASSES)
            print(f'auc: {auc}')
            print(f'auc: {auc}', file=log_writer)
            print(np.array(auc).mean())
            print(np.array(auc).mean(), file=log_writer)

        # compute DEOdds across age groups
        if len(outputs_all_age) > 0:
            outputs_all_age = np.vstack(outputs_all_age)
            targets_all_age = np.vstack(targets_all_age)
            group_values_age = list(age_loaders.keys())
            thresholds = [0.5] * config.MODEL.NUM_CLASSES
            deodds_per_class_age, deodds_mean_age = compute_deodds(targets_all_age, outputs_all_age, demographics_all_age, thresholds, config.MODEL.NUM_CLASSES, 'age', group_values_age)
            print(f'DEOdds per class (age): {deodds_per_class_age}')
            print(f'DEOdds mean (age): {deodds_mean_age}')
            print(f'DEOdds per class (age): {deodds_per_class_age}', file=log_writer)
            print(f'DEOdds mean (age): {deodds_mean_age}', file=log_writer)
            
            
    elif config.DATA.DATASET == 'CheXpert_gender':
        test_dataset_M, test_loader_M, test_dataset_F, test_loader_F = build_loader(config, dataset = config.DATA.DATASET)
    
        print('Testing male:', file=log_writer)
        print('Testing male:')
        for idx, (image, label) in enumerate(test_dataset_M):
        
            if len(image.shape)==3:
                image = torch.unsqueeze(image, dim=0)
            image = image.to(device) # [10,3,224,224]
            
            # print(image.shape)
            # image = image.unsqueeze(0) # (3,512,512) --> (1,3,512,512)
            logits = model(image).mean(0).sigmoid() # ncrop
            # print(logits.shape)
            # sys.exit(1)
            logits = logits.cpu().detach().numpy()

            if idx==0:
                output_m = logits
                target_m = label
            else:
                output_m = np.vstack([output_m, logits])
                target_m = np.vstack([target_m, label])
            print(idx)
        auc_m = metric_AUROC(target_m, output_m, nb_classes=config.MODEL.NUM_CLASSES)
        
        if config.MODEL.NUM_CLASSES==14:
            print(f'auc across 14 diseases (male): {auc_m}')
            print(f'auc across 14 diseases (male): {auc_m}', file=log_writer)
            print(f'average auc (male): {np.array(auc_m).mean()}')
            print(f'average auc (male): {np.array(auc_m).mean()}', file=log_writer)
            print(f'auc across 5 diseases (male): {auc_m[2], auc_m[5], auc_m[6], auc_m[8], auc_m[10]}')
            print(f'auc across 5 diseases (male): {auc_m[2], auc_m[5], auc_m[6], auc_m[8], auc_m[10]}', file=log_writer)
            print(f'average auc across 5 diseases (male): {(auc_m[2]+auc_m[5]+auc_m[6]+auc_m[8]+auc_m[10])/5}')
            print(f'average auc across 5 diseases (male): {(auc_m[2]+auc_m[5]+auc_m[6]+auc_m[8]+auc_m[10])/5}', file=log_writer)
        else:
            print(f'auc (male): {auc_m}')
            print(f'auc (male): {auc_m}', file=log_writer)
            print(np.array(auc_m).mean())
            print(np.array(auc_m).mean(), file=log_writer)
        
        
        print('Testing female:', file=log_writer)
        print('Testing female:')
        for idx, (image, label) in enumerate(test_dataset_F):
        
            if len(image.shape)==3:
                image = torch.unsqueeze(image, dim=0)
            image = image.to(device) # [10,3,224,224]
            
            # print(image.shape)
            # image = image.unsqueeze(0) # (3,512,512) --> (1,3,512,512)
            logits = model(image).mean(0).sigmoid() # ncrop
            # print(logits.shape)
            # sys.exit(1)
            logits = logits.cpu().detach().numpy()


            if idx==0:
                output_f = logits
                target_f = label
            else:
                output_f = np.vstack([output_f, logits])
                target_f = np.vstack([target_f, label])
            print(idx)
        
        auc_f = metric_AUROC(target_f, output_f, nb_classes=config.MODEL.NUM_CLASSES)
        print(f'auc across 14 diseases (female): {auc_f}')
        print(f'auc across 14 diseases (female): {auc_f}', file=log_writer)
        print(f'average auc (female): {np.array(auc_f).mean()}')
        print(f'average auc (female): {np.array(auc_f).mean()}', file=log_writer)
        print(f'auc across 5 diseases (female): {auc_f[2], auc_f[5], auc_f[6], auc_f[8], auc_f[10]}')
        print(f'auc across 5 diseases (female): {auc_f[2], auc_f[5], auc_f[6], auc_f[8], auc_f[10]}', file=log_writer)
        print(f'average auc across 5 diseases (female): {(auc_f[2]+auc_f[5]+auc_f[6]+auc_f[8]+auc_f[10])/5}')
        print(f'average auc across 5 diseases (female): {(auc_f[2]+auc_f[5]+auc_f[6]+auc_f[8]+auc_f[10])/5}', file=log_writer)

        # compute DEOdds across male/female for CheXpert
        try:
            outputs_all = np.vstack([output_m, output_f])
            targets_all = np.vstack([target_m, target_f])
        except Exception:
            outputs_all = output_m if output_f.size==0 else output_f
            targets_all = target_m if target_f.size==0 else target_f
        demographics = [{'sex': 'M'} for _ in range(output_m.shape[0])] + [{'sex': 'F'} for _ in range(output_f.shape[0])]
        thresholds = [0.5] * config.MODEL.NUM_CLASSES
        group_values = ['M', 'F']
        deodds_per_class, deodds_mean = compute_deodds(targets_all, outputs_all, demographics, thresholds, config.MODEL.NUM_CLASSES, 'sex', group_values)
        print(f'DEOdds per class (sex): {deodds_per_class}')
        print(f'DEOdds mean (sex): {deodds_mean}')
        print(f'DEOdds per class (sex): {deodds_per_class}', file=log_writer)
        print(f'DEOdds mean (sex): {deodds_mean}', file=log_writer)

    elif config.DATA.DATASET == 'CheXpert_age':
        age_loaders = build_loader(config, dataset = config.DATA.DATASET)

        outputs_all_age = []
        targets_all_age = []
        demographics_all_age = []

        for age_group, (dataset_age, loader) in age_loaders.items():
            print(f"Infer on age group: {age_group}")
            print(f"Infer on age group: {age_group}", file=log_writer)
            
            for idx, (image, label) in enumerate(dataset_age):
        
                if len(image.shape)==3:
                    image = torch.unsqueeze(image, dim=0)
                image = image.to(device) # [10,3,224,224]
                
                # print(image.shape)
                # image = image.unsqueeze(0) # (3,512,512) --> (1,3,512,512)
                logits = model(image).mean(0).sigmoid() # ncrop
                # print(logits.shape)
                # sys.exit(1)
                logits = logits.cpu().detach().numpy()
                

                if idx==0:
                    output = logits
                    target = label
                else:
                    output = np.vstack([output, logits])
                    target = np.vstack([target, label])
                print(idx)

            # append group results for DEOdds
            outputs_all_age.append(output)
            targets_all_age.append(target)
            demographics_all_age += [{ 'age': age_group } for _ in range(output.shape[0])]

            auc = metric_AUROC(target, output, nb_classes=config.MODEL.NUM_CLASSES)
            print(f'auc across 14 diseases: {auc}')
            print(f'auc across 14 diseases: {auc}', file=log_writer)
            print(f'average auc: {np.array(auc).mean()}')
            print(f'average auc: {np.array(auc).mean()}', file=log_writer)
            print(f'auc across 5 diseases: {auc[2], auc[5], auc[6], auc[8], auc[10]}')
            print(f'auc across 5 diseases: {auc[2], auc[5], auc[6], auc[8], auc[10]}', file=log_writer)
            print(f'average auc across 5 diseases: {(auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5}')
            print(f'average auc across 5 diseases: {(auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5}', file=log_writer)

        # compute DEOdds across age groups
        if len(outputs_all_age) > 0:
            outputs_all_age = np.vstack(outputs_all_age)
            targets_all_age = np.vstack(targets_all_age)
            group_values_age = list(age_loaders.keys())
            thresholds = [0.5] * config.MODEL.NUM_CLASSES
            deodds_per_class_age, deodds_mean_age = compute_deodds(targets_all_age, outputs_all_age, demographics_all_age, thresholds, config.MODEL.NUM_CLASSES, 'age', group_values_age)
            print(f'DEOdds per class (age): {deodds_per_class_age}')
            print(f'DEOdds mean (age): {deodds_mean_age}')
            print(f'DEOdds per class (age): {deodds_per_class_age}', file=log_writer)
            print(f'DEOdds mean (age): {deodds_mean_age}', file=log_writer)
            
    elif config.DATA.DATASET == 'CheXpert_race':
        race_loaders = build_loader(config, dataset = config.DATA.DATASET)
        outputs_all_race = []
        targets_all_race = []
        demographics_all_race = []

        for race_group, (dataset_race, loader) in race_loaders.items():
            print(f"Infer on race group: {race_group}")
            print(f"Infer on race group: {race_group}", file=log_writer)
            
            # ipdb.set_trace()
            for idx, (image, label) in enumerate(dataset_race):
        
                if len(image.shape)==3:
                    image = torch.unsqueeze(image, dim=0)
                image = image.to(device) # [10,3,224,224]
                
                # print(image.shape)
                # image = image.unsqueeze(0) # (3,512,512) --> (1,3,512,512)
                logits = model(image).mean(0).sigmoid() # ncrop
                # print(logits.shape)
                # sys.exit(1)
                logits = logits.cpu().detach().numpy()
                

                if idx==0:
                    output = logits
                    target = label
                else:
                    output = np.vstack([output, logits])
                    target = np.vstack([target, label])
                print(idx)

            outputs_all_race.append(output)
            targets_all_race.append(target)
            demographics_all_race += [{ 'race': race_group } for _ in range(output.shape[0])]

            auc = metric_AUROC(target, output, nb_classes=config.MODEL.NUM_CLASSES)
            print(f'auc across 14 diseases: {auc}')
            print(f'auc across 14 diseases: {auc}', file=log_writer)
            print(f'average auc: {np.array(auc).mean()}')
            print(f'average auc: {np.array(auc).mean()}', file=log_writer)
            print(f'auc across 5 diseases: {auc[2], auc[5], auc[6], auc[8], auc[10]}')
            print(f'auc across 5 diseases: {auc[2], auc[5], auc[6], auc[8], auc[10]}', file=log_writer)
            print(f'average auc across 5 diseases: {(auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5}')
            print(f'average auc across 5 diseases: {(auc[2]+auc[5]+auc[6]+auc[8]+auc[10])/5}', file=log_writer)

        # compute DEOdds across race groups
        if len(outputs_all_race) > 0:
            outputs_all_race = np.vstack(outputs_all_race)
            targets_all_race = np.vstack(targets_all_race)
            group_values_race = list(race_loaders.keys())
            thresholds = [0.5] * config.MODEL.NUM_CLASSES
            deodds_per_class_race, deodds_mean_race = compute_deodds(targets_all_race, outputs_all_race, demographics_all_race, thresholds, config.MODEL.NUM_CLASSES, 'race', group_values_race)
            print(f'DEOdds per class (race): {deodds_per_class_race}')
            print(f'DEOdds mean (race): {deodds_mean_race}')
            print(f'DEOdds per class (race): {deodds_per_class_race}', file=log_writer)
            print(f'DEOdds mean (race): {deodds_mean_race}', file=log_writer)
            



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




def compute_deodds(gt, pred, demographics, threshold, n_class, group_attr, group_values):
    """
    Compute DEOdds (Difference in Equalized Odds) per class.
    
    DEOdds = max over all group pairs of max(|TPR_a - TPR_b|, |FPR_a - FPR_b|)
    
    Parameters:
        gt:           np.ndarray, shape (N, n_class), ground truth labels (0/1)
        pred:         np.ndarray, shape (N, n_class), predicted probabilities
        demographics: list of dicts, length N, each dict has group_attr as key
                      e.g. [{'sex': 'M', 'age': '40-60'}, ...]
        threshold:    list of float, length n_class, per-class binarization threshold
        n_class:      int, number of classes
        group_attr:   str, demographic attribute name, e.g. 'sex', 'age', 'race'
        group_values: list of str, valid group values, e.g. ['M', 'F']
    
    Returns:
        deodds_per_class: list of float, DEOdds for each class
        deodds_mean:      float, mean DEOdds across classes
    """
    # Step 1: Binarize predictions using per-class thresholds
    pred_binary = np.zeros_like(pred)
    for i in range(n_class):
        th = threshold[i] if i < len(threshold) else 0.5
        pred_binary[:, i] = (pred[:, i] >= th).astype(np.float64)
    
    deodds_per_class = []
    
    for c in range(n_class):
        y_c = gt[:, c]
        p_c = pred_binary[:, c]
        
        # Step 2: Compute TPR and FPR for each demographic group
        tpr_fpr = {}
        for g in group_values:
            mask = np.array([d[group_attr] == g for d in demographics])
            if np.sum(mask) == 0:
                continue
            y_g = y_c[mask]
            p_g = p_c[mask]
            
            pos = (y_g >= 0.5)
            neg = ~pos
            tp = np.sum(p_g[pos] >= 0.5) if np.sum(pos) > 0 else 0
            fn = np.sum(p_g[pos] < 0.5) if np.sum(pos) > 0 else 0
            fp = np.sum(p_g[neg] >= 0.5) if np.sum(neg) > 0 else 0
            tn = np.sum(p_g[neg] < 0.5) if np.sum(neg) > 0 else 0
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            tpr_fpr[g] = (tpr, fpr)
        
        # Step 3: DEOdds = max over all group pairs of max(|TPR_diff|, |FPR_diff|)
        valid_vals = [v for v in tpr_fpr.values() if not (np.isnan(v[0]) or np.isnan(v[1]))]
        if len(valid_vals) >= 2:
            deo = 0.0
            for i in range(len(valid_vals)):
                for j in range(i + 1, len(valid_vals)):
                    tpr_diff = abs(valid_vals[i][0] - valid_vals[j][0])
                    fpr_diff = abs(valid_vals[i][1] - valid_vals[j][1])
                    deo = max(deo, tpr_diff, fpr_diff)
            deodds_per_class.append(deo)
        else:
            deodds_per_class.append(np.nan)
    
    deodds_mean = np.nanmean(deodds_per_class) if deodds_per_class else np.nan
    return deodds_per_class, deodds_mean





if __name__=='__main__':
    args, config = parse_option()
    main(config)