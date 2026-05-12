import os

import albumentations
import cv2
import numpy as np
import torch
import torch.distributed as dist
from sklearn.model_selection import KFold, train_test_split
# from albumentations.pytorch.transforms import ToTensorV2
# import albumentations.augmentations.transforms as transforms
from timm.data import Mixup
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import random
import csv
import copy
import pandas as pd
from .Augmentation import CustomMetalArtifact, CustomBrightnessEnhance, CustomGaussianNoise
import albumentations as A
from albumentations.pytorch import ToTensorV2


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def build_loader_CheXpert(config, ddp=False, uncertain_label='LSR-Ones', unknown_label=0):
    dataset_root = config.DATA.DATA_PATH
    traincsv = config.DATA.TRAIN_LIST
    valcsv = config.DATA.VAL_LIST
    testcsv = config.DATA.TEST_LIST

    train_list = []
    train_label = []
    val_list = []
    val_label = []
    test_list = []
    test_label = []
    
    fold = config.DATA.FOLD
    BASE_SEED = 42
    fold_seed = BASE_SEED + int(fold)

    if config.MODE == 'train':
        with open(traincsv, 'r') as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            next(csvReader, None)
            for line in csvReader:
                train_list.append(line[0])
                label = line[5:]
                for i in range(config.MODEL.NUM_CLASSES):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == 0:
                            label[i] = 0
                        elif a == -1:
                            if uncertain_label == "Ones":
                                label[i] = 1
                            elif uncertain_label == "Zeros":
                                label[i] = 0
                            elif uncertain_label == "LSR-Ones":
                                label[i] = random.uniform(0.55, 0.85)
                            elif uncertain_label == "LSR-Zeros":
                                label[i] = random.uniform(0, 0.3)
                    else:
                        label[i] = unknown_label
                
                imageLabel = [float(i) for i in label]
                train_label.append(imageLabel)    

        with open(valcsv, 'r') as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            next(csvReader, None)
            for line in csvReader:
                val_list.append(line[0])
                label = line[5:]
                for i in range(config.MODEL.NUM_CLASSES):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == 0:
                            label[i] = 0
                        elif a == -1:
                            if uncertain_label == "Ones":
                                label[i] = 1
                            elif uncertain_label == "Zeros":
                                label[i] = 0
                            elif uncertain_label == "LSR-Ones":
                                label[i] = random.uniform(0.55, 0.85)
                            elif uncertain_label == "LSR-Zeros":
                                label[i] = random.uniform(0, 0.3)
                    else:
                        label[i] = unknown_label
                
                imageLabel = [float(i) for i in label]
                val_label.append(imageLabel)  

        # 10折交叉验证
        # rkf = KFold(n_splits=10, shuffle=False)
        # for fold, (train_index, val_index) in enumerate(rkf.split(train_list)): # rkf.split返回的是train和val的index
        #     locals()['train_list'+str(fold)] = []
        #     locals()['val_list'+str(fold)] = []
        #     locals()['train_label'+str(fold)] = []
        #     locals()['val_label'+str(fold)] = []
        #     for i in train_index:
        #         locals()['train_list'+str(fold)].append(train_list[i])
        #         locals()['train_label'+str(fold)].append(train_label[i])
        #     for i in val_index:
        #         locals()['val_list'+str(fold)].append(train_list[i])
        #         locals()['val_label'+str(fold)].append(train_label[i])

        # train_list = locals()['train_list'+config.DATA.FOLD] 
        # val_list = locals()['val_list'+config.DATA.FOLD] 
        # train_label = locals()['train_label'+config.DATA.FOLD] 
        # val_label = locals()['val_label'+config.DATA.FOLD] 

        # train_list, val_list, train_label, val_label = train_test_split(train_list, train_label, test_size=0.1, random_state=24)

        img_train_transforms = img_transforms(mode='train', config=config)
        train_dataset = CheXpert_dataset(dataset_root=dataset_root, datalist=train_list, labellist=train_label, img_transforms=img_train_transforms)


        # 此数据集不使用验证集，返回val_loader是为了和其他数据集的返回值对齐
        img_val_transforms = img_transforms(mode='val', config=config)
        val_dataset = CheXpert_dataset(dataset_root=dataset_root, datalist=val_list, labellist=val_label, img_transforms=img_val_transforms)

        if ddp:
            sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, seed=fold_seed)
            sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            train_loader = DataLoader(dataset=train_dataset, 
                                    sampler=sampler_train,
                                    batch_size=config.DATA.BATCH_SIZE, 
                                    # shuffle=True, 
                                    num_workers=config.DATA.NUM_WORKERS,
                                    drop_last=True,
                                    pin_memory=True,
                                    worker_init_fn=seed_worker)

            val_loader = DataLoader(dataset=val_dataset, 
                                sampler=sampler_val,
                                batch_size=config.DATA.BATCH_SIZE, 
                                num_workers=config.DATA.NUM_WORKERS,
                                worker_init_fn=seed_worker)
        else:
            train_loader = DataLoader(dataset=train_dataset, 
                                    # sampler=sampler_train,
                                    batch_size=config.DATA.BATCH_SIZE, 
                                    shuffle=True, 
                                    num_workers=config.DATA.NUM_WORKERS,
                                    drop_last=True)

            
            val_loader = DataLoader(dataset=val_dataset, 
                                # sampler=sampler_val,
                                batch_size=config.DATA.BATCH_SIZE, 
                                num_workers=config.DATA.NUM_WORKERS)

        # setup mixup / cutmix
        mixup_fn = None
        mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
        if mixup_active:
            mixup_fn = Mixup(
                        mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
                        prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
                        label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
        return train_dataset, val_dataset, train_loader, val_loader, mixup_fn
    
    elif config.MODE == 'test':
        with open(testcsv, 'r') as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            next(csvReader, None)
            for line in csvReader:
                test_list.append(line[0])
                label = line[1:]
                for i in range(config.MODEL.NUM_CLASSES):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == 0:
                            label[i] = 0
                        elif a == -1:
                            if uncertain_label == "Ones":
                                label[i] = 1
                            elif uncertain_label == "Zeros":
                                label[i] = 0
                            elif uncertain_label == "LSR-Ones":
                                label[i] = random.uniform(0.55, 0.85)
                            elif uncertain_label == "LSR-Zeros":
                                label[i] = random.uniform(0, 0.3)
                    else:
                        label[i] = unknown_label
                
                imageLabel = [float(i) for i in label]
                test_label.append(imageLabel)   
        img_test_transforms = img_transforms(mode='test', config=config)
        test_dataset = CheXpert_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, img_transforms=img_test_transforms)
        test_loader = DataLoader(dataset=test_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        num_workers=config.DATA.NUM_WORKERS)
        return test_dataset, test_loader

class CheXpert_dataset(Dataset):
    def __init__(self, dataset_root, datalist, labellist, img_transforms):
        # super(NIHchest_dataset, self).__init__()
        self.img_transforms = img_transforms
        self.dataset_root = dataset_root
        self.datalist = datalist
        self.labellist = labellist

    def __getitem__(self, index):

        # print(os.path.join(self.dataset_root, self.datalist[index]))
        image = cv2.imread(os.path.join(self.dataset_root, self.datalist[index]))
        # image = torch.tensor(image)
        # image = image.cuda()
        image = self.img_transforms(image)

        label = torch.FloatTensor(self.labellist[index]) 

        return image.float(), label
    # return image 
    def __len__(self):
        return len(self.datalist)
    
    
    
class CheXpert_robust_dataset(Dataset):
    def __init__(self, dataset_root, datalist, labellist, albu_transform, ten_crop_transform):
        # super(NIHchest_dataset, self).__init__()
        self.albu_transform = albu_transform
        self.dataset_root = dataset_root
        self.datalist = datalist
        self.labellist = labellist
        
        self.ten_crop_transform = ten_crop_transform


    def __getitem__(self, index):

        image = cv2.imread(os.path.join(self.dataset_root, self.datalist[index]))
        label = torch.FloatTensor(self.labellist[index]) 
        # Albumentations：robust artifact / noise
        image = self.albu_transform(image=image)["image"]  # (1, H, W)

        # torchvision：TenCrop
        image = self.ten_crop_transform(image)             # (10, 1, Hc, Wc)
        return image.float(), label
    # return image 
    def __len__(self):
        return len(self.datalist)    




def img_transforms(mode, config):
    if mode == 'train':
        data_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.RandomResizedCrop(config.DATA.IMG_SIZE),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomRotation(degrees=7),
                                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
                                            ])
    elif mode == 'val':
        data_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Resize(config.DATA.CROP_SIZE),
                                            transforms.CenterCrop(config.DATA.IMG_SIZE),
                                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
                                            ])
    elif mode == 'test':
        data_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Resize(config.DATA.CROP_SIZE),
                                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252]),
                                            transforms.TenCrop(config.DATA.IMG_SIZE),
                                            transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
                                            ])

    return data_transforms



def img_transform_robust(mode, config):
    if mode == 'CustomMetalArtifact':
        aug = CustomMetalArtifact(p=1.0)
    elif mode == 'CustomBrightnessEnhance':
        aug = CustomBrightnessEnhance(p=1.0)
    elif mode == 'CustomGaussianNoise':
        aug = CustomGaussianNoise(p=1.0)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    return A.Compose([
        A.Resize(config.DATA.CROP_SIZE, config.DATA.CROP_SIZE),
        aug,
        A.Normalize(mean=[0.5056], std=[0.252]),
        ToTensorV2(),
    ])


def img_transform_tencrop(config):
    ten_crop_transform = transforms.Compose([
    transforms.TenCrop(config.DATA.IMG_SIZE),
    transforms.Lambda(lambda crops: torch.stack(crops))
])
    return ten_crop_transform



def build_loader_CheXpert_gender(config, ddp=False, uncertain_label='LSR-Ones', unknown_label=0):
    dataset_root = config.DATA.DATA_PATH
    testcsv = config.DATA.VAL_LIST

    test_list_M, test_label_M = [], []
    test_list_F, test_label_F = [], []

    with open(testcsv, 'r') as fileDescriptor:
        csvReader = csv.reader(fileDescriptor)
        next(csvReader, None)  # skip header

        for line in csvReader:
            image_path = line[0]
            sex = line[1]  # 第二列是 Sex
            label = line[5:]  # 第六列开始是 label

            # label 处理逻辑保持不变
            for i in range(config.MODEL.NUM_CLASSES):
                if label[i]:
                    a = float(label[i])
                    if a == 1:
                        label[i] = 1
                    elif a == 0:
                        label[i] = 0
                    elif a == -1:
                        if uncertain_label == "Ones":
                            label[i] = 1
                        elif uncertain_label == "Zeros":
                            label[i] = 0
                        elif uncertain_label == "LSR-Ones":
                            label[i] = random.uniform(0.55, 0.85)
                        elif uncertain_label == "LSR-Zeros":
                            label[i] = random.uniform(0, 0.3)
                else:
                    label[i] = unknown_label

            imageLabel = [int(i) for i in label]

            # === 按性别分流 ===
            if sex == "Male":
                test_list_M.append(image_path)
                test_label_M.append(imageLabel)
            elif sex == "Female":
                test_list_F.append(image_path)
                test_label_F.append(imageLabel)

    img_test_transforms = img_transforms(mode='test', config=config)

    test_dataset_M = CheXpert_dataset(
        dataset_root=dataset_root,
        datalist=test_list_M,
        labellist=test_label_M,
        img_transforms=img_test_transforms
    )

    test_dataset_F = CheXpert_dataset(
        dataset_root=dataset_root,
        datalist=test_list_F,
        labellist=test_label_F,
        img_transforms=img_test_transforms
    )

    test_loader_M = DataLoader(
        dataset=test_dataset_M,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        shuffle=False
    )

    test_loader_F = DataLoader(
        dataset=test_dataset_F,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        shuffle=False
    )

    return test_dataset_M, test_loader_M, test_dataset_F, test_loader_F




def build_loader_CheXpert_age(
    config,
    ddp=False,
    uncertain_label='LSR-Ones',
    unknown_label=0
):
    dataset_root = config.DATA.DATA_PATH
    testcsv = config.DATA.VAL_LIST

    # 五个年龄段
    age_bins = {
        # "0_20":   (0, 20),
        "20_40":  (20, 40),
        "40_60":  (40, 60),
        "60_80":  (60, 80),
        "80_plus": (80, float("inf")),
    }

    test_lists = {k: [] for k in age_bins}
    test_labels = {k: [] for k in age_bins}

    with open(testcsv, 'r') as fileDescriptor:
        csvReader = csv.reader(fileDescriptor)
        next(csvReader, None)  # skip header

        for line in csvReader:
            image_path = line[0]
            age = line[2]              # 第三列是 Age
            label = line[5:]           # 后面才是 label

            # -------- label 处理逻辑（完全保留） --------
            for i in range(config.MODEL.NUM_CLASSES):
                if label[i]:
                    a = float(label[i])
                    if a == 1:
                        label[i] = 1
                    elif a == 0:
                        label[i] = 0
                    elif a == -1:
                        if uncertain_label == "Ones":
                            label[i] = 1
                        elif uncertain_label == "Zeros":
                            label[i] = 0
                        elif uncertain_label == "LSR-Ones":
                            label[i] = random.uniform(0.55, 0.85)
                        elif uncertain_label == "LSR-Zeros":
                            label[i] = random.uniform(0, 0.3)
                else:
                    label[i] = unknown_label

            imageLabel = [int(i) for i in label]

            # -------- 年龄分组 --------
            try:
                age = float(age)
            except:
                continue  # 跳过无效年龄

            for k, (low, high) in age_bins.items():
                if low <= age < high:
                    test_lists[k].append(image_path)
                    test_labels[k].append(imageLabel)
                    break

    img_test_transforms = img_transforms(mode='test', config=config)

    age_loaders = {}

    for age_group in age_bins:
        test_dataset = CheXpert_dataset(
            dataset_root=dataset_root,
            datalist=test_lists[age_group],
            labellist=test_labels[age_group],
            img_transforms=img_test_transforms
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            shuffle=False
        )

        age_loaders[age_group] = (test_dataset, test_loader)

    return age_loaders




def build_loader_CheXpert_robustness(config, uncertain_label='LSR-Ones', unknown_label=0):
    dataset_root = config.DATA.DATA_PATH
    testcsv = config.DATA.TEST_LIST
    test_list = []
    test_label = []

    with open(testcsv, 'r') as fileDescriptor:
        csvReader = csv.reader(fileDescriptor)
        next(csvReader, None)  # skip header

        with open(testcsv, 'r') as fileDescriptor:
            csvReader = csv.reader(fileDescriptor)
            next(csvReader, None)
            for line in csvReader:
                test_list.append(line[0])
                label = line[1:]
                for i in range(config.MODEL.NUM_CLASSES):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == 0:
                            label[i] = 0
                        elif a == -1:
                            if uncertain_label == "Ones":
                                label[i] = 1
                            elif uncertain_label == "Zeros":
                                label[i] = 0
                            elif uncertain_label == "LSR-Ones":
                                label[i] = random.uniform(0.55, 0.85)
                            elif uncertain_label == "LSR-Zeros":
                                label[i] = random.uniform(0, 0.3)
                    else:
                        label[i] = unknown_label
                
                imageLabel = [float(i) for i in label]
                test_label.append(imageLabel)   

            
    # no augmentation
    img_transforms_init = img_transforms(mode='test', config=config)
    test_dataset_init = CheXpert_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, img_transforms=img_transforms_init)
    test_loader_init = DataLoader(dataset=test_dataset_init, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS)
    
    # add augmentation to test robustness
    img_transforms_robustness = img_transform_robust(mode='CustomMetalArtifact', config=config)
    ten_crop_transform = img_transform_tencrop(config=config)
    test_dataset_metalartifact = CheXpert_robust_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, albu_transform=img_transforms_robustness, ten_crop_transform=ten_crop_transform)
    test_loader_robustness = DataLoader(dataset=test_dataset_metalartifact, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS)
    
    img_transforms_robustness = img_transform_robust(mode='CustomBrightnessEnhance', config=config)
    ten_crop_transform = img_transform_tencrop(config=config)
    test_dataset_brightnessenhance = CheXpert_robust_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, albu_transform=img_transforms_robustness, ten_crop_transform=ten_crop_transform)
    test_loader_brightnessenhance = DataLoader(dataset=test_dataset_brightnessenhance, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS)
    
    img_transforms_robustness = img_transform_robust(mode='CustomGaussianNoise', config=config)
    ten_crop_transform = img_transform_tencrop(config=config)
    test_dataset_gaussiannoise = CheXpert_robust_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, albu_transform=img_transforms_robustness, ten_crop_transform=ten_crop_transform)
    test_loader_gaussiannoise = DataLoader(dataset=test_dataset_gaussiannoise, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS)
    
    
    return test_dataset_init, test_dataset_metalartifact, test_dataset_brightnessenhance, test_dataset_gaussiannoise





def build_loader_CheXpert_race(
    config,
    uncertain_label='LSR-Ones',
    unknown_label=0
):
    dataset_root = config.DATA.DATA_PATH
    testcsv = config.DATA.VAL_LIST
    metadata_path = config.DATA.METADATA


    # -------- 定义需要的 race 分组 --------
    race_groups = ["White", "Black", "Asian", "Other"]

    test_lists = {k: [] for k in race_groups}
    test_labels = {k: [] for k in race_groups}

    # -------- 读取 metadata，建立 Path -> race 映射（仅 split=val） --------
    path_to_race = {}

    with open(metadata_path, 'r') as metaFile:
        metaReader = csv.reader(metaFile)
        header = next(metaReader)

        path_index = header.index("path_to_image")
        race_index = header.index("race")
        split_index = header.index("split")

        for row in metaReader:
            split = row[split_index].strip().lower()

            # 只保留 split 为 val 的样本
            if split != "valid":
                continue

            img_path = row[path_index]
            race = row[race_index].strip()

            if race in ["White", "Black", "Asian"]:
                path_to_race[img_path] = race
            else:
                path_to_race[img_path] = "Other"

    # -------- 读取 testcsv --------
    with open(testcsv, 'r') as fileDescriptor:
        csvReader = csv.reader(fileDescriptor)
        next(csvReader, None)

        for line in csvReader:
            image_path = line[0]
            label = line[5:]

            # -------- label 处理逻辑（保持不变） --------
            for i in range(config.MODEL.NUM_CLASSES):
                if label[i]:
                    a = float(label[i])
                    if a == 1:
                        label[i] = 1
                    elif a == 0:
                        label[i] = 0
                    elif a == -1:
                        if uncertain_label == "Ones":
                            label[i] = 1
                        elif uncertain_label == "Zeros":
                            label[i] = 0
                        elif uncertain_label == "LSR-Ones":
                            label[i] = random.uniform(0.55, 0.85)
                        elif uncertain_label == "LSR-Zeros":
                            label[i] = random.uniform(0, 0.3)
                else:
                    label[i] = unknown_label

            imageLabel = [int(i) for i in label]

            # -------- 根据 metadata 找 race --------
            align_path = image_path.removeprefix("CheXpert-v1.0/")
            # import ipdb; ipdb.set_trace()
            if align_path not in path_to_race:
                continue  # metadata 里找不到的跳过

            
            race = path_to_race[align_path]

            test_lists[race].append(image_path)
            test_labels[race].append(imageLabel)

    img_test_transforms = img_transforms(mode='test', config=config)

    race_loaders = {}

    for race in race_groups:
        test_dataset = CheXpert_dataset(
            dataset_root=dataset_root,
            datalist=test_lists[race],
            labellist=test_labels[race],
            img_transforms=img_test_transforms
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            shuffle=False
        )

        race_loaders[race] = (test_dataset, test_loader)

    return race_loaders