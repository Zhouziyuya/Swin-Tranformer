import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
import csv
import pandas as pd
from .Augmentation import CustomMetalArtifact, CustomBrightnessEnhance, CustomGaussianNoise
import random



def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader_NIHchest(config, ddp):
    dataset_root = config.DATA.DATA_PATH
    traintxt = config.DATA.TRAIN_LIST
    valtxt = config.DATA.VAL_LIST
    testtxt = config.DATA.TEST_LIST

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
        with open(traintxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                train_list.append(i.split(' ')[0])
                # label = i.split(' ')[1:15]
                label = [int(x) for x in i.split(' ')[1:15]]
                train_label.append(label)
        with open(valtxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                val_list.append(i.split(' ')[0])
                label = [int(x) for x in i.split(' ')[1:15]]
                val_label.append(label)

        if config.POPAR_FORM:
            train_list = np.hstack((train_list, val_list))
            train_label = np.vstack((train_label, val_label))

        # 10折交叉验证
        rkf = KFold(n_splits=10, shuffle=False)
        for fold, (train_index, val_index) in enumerate(rkf.split(train_list)): # rkf.split返回的是train和val的index
            locals()['train_list'+str(fold)] = []
            locals()['val_list'+str(fold)] = []
            locals()['train_label'+str(fold)] = []
            locals()['val_label'+str(fold)] = []
            for i in train_index:
                locals()['train_list'+str(fold)].append(train_list[i])
                locals()['train_label'+str(fold)].append(train_label[i])
            for i in val_index:
                locals()['val_list'+str(fold)].append(train_list[i])
                locals()['val_label'+str(fold)].append(train_label[i])

        train_list = locals()['train_list'+config.DATA.FOLD] 
        val_list = locals()['val_list'+config.DATA.FOLD] 
        train_label = locals()['train_label'+config.DATA.FOLD] 
        val_label = locals()['val_label'+config.DATA.FOLD] 

        # train_list, val_list, train_label, val_label = train_test_split(train_list, train_label, test_size=0.1, random_state=24)

        img_train_transforms = img_transforms(mode='train', config=config)
        train_dataset = NIHchest_dataset(dataset_root=dataset_root, datalist=train_list, labellist=train_label, img_transforms=img_train_transforms)
        
        img_val_transforms = img_transforms(mode='val', config=config)
        val_dataset = NIHchest_dataset(dataset_root=dataset_root, datalist=val_list, labellist=val_label, img_transforms=img_val_transforms)
        

        if ddp:
            sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, seed=fold_seed)
            sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

            train_loader = DataLoader(dataset=train_dataset, 
                                    # sampler=sampler_train,
                                    batch_size=config.DATA.BATCH_SIZE, 
                                    # shuffle=True, 
                                    num_workers=config.DATA.NUM_WORKERS,
                                    drop_last=True,
                                    sampler=sampler_train,
                                    worker_init_fn=seed_worker)

            
            val_loader = DataLoader(dataset=val_dataset, 
                                # sampler=sampler_val,
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
        with open(testtxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                test_list.append(i.split(' ')[0])
                label = [int(x) for x in i.split(' ')[1:15]]
                test_label.append(label)
        img_test_transforms = img_transforms(mode='test', config=config)
        test_dataset = NIHchest_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, img_transforms=img_test_transforms)
        test_loader = DataLoader(dataset=test_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        num_workers=config.DATA.NUM_WORKERS)
        return test_dataset, test_loader

class NIHchest_dataset(Dataset):
    def __init__(self, dataset_root, datalist, labellist, img_transforms):
        # super(NIHchest_dataset, self).__init__()
        self.img_transforms = img_transforms
        self.dataset_root = dataset_root
        self.datalist = datalist
        self.labellist = labellist

    def __getitem__(self, index):

        image = cv2.imread(os.path.join(self.dataset_root, self.datalist[index]))
        label = torch.FloatTensor(self.labellist[index]) 
        image = self.img_transforms(image)

        # return image.float(), label, self.datalist[index]
        return image.float(), label
    # return image 
    def __len__(self):
        return len(self.datalist)
    
    
class NIHchest_robust_dataset(Dataset):
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




def get_patient_sex(df, image_index):
    matches = df[df.iloc[:, 0].str.strip() == str(image_index).strip()]
    return matches.iloc[0, 5]


def save_NIHchest_gender_txt(config):
    testtxt = config.DATA.TEST_LIST
    metacsv = config.METADATA
    save_dir = os.path.dirname(testtxt)  # 和原 TEST_LIST 放一起

    male_txt = os.path.join(save_dir, "test_male.txt")
    female_txt = os.path.join(save_dir, "test_female.txt")

    # 如果已经存在，直接返回路径（防止重复生成）
    if os.path.exists(male_txt) and os.path.exists(female_txt):
        print("Gender txt files already exist.")
        return male_txt, female_txt

    df = pd.read_csv(metacsv)

    male_lines = []
    female_lines = []

    with open(testtxt, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split(' ')
            image_index = items[0]
            # labels = items[1:15]

            sex = get_patient_sex(df, image_index)
            print(f"Image: {image_index}, Sex: {sex}")

            if sex == 'M':
                male_lines.append(line)
            elif sex == 'F':
                female_lines.append(line)

    # 写入 txt
    with open(male_txt, "w", encoding="utf-8") as f:
        f.writelines(male_lines)

    with open(female_txt, "w", encoding="utf-8") as f:
        f.writelines(female_lines)

    print(f"Saved male test list to: {male_txt}")
    print(f"Saved female test list to: {female_txt}")

    return male_txt, female_txt


def build_loader_NIHchest_gender(config, ddp): # split the man and women on test set for fairness test
    male_txt, female_txt = save_NIHchest_gender_txt(config)
    dataset_root = config.DATA.DATA_PATH
    
    test_man_list = []
    test_man_label = []
    test_woman_list = []
    test_woman_label = []
    

    with open(male_txt, encoding='utf-8') as e: # load train list and train label
        list = e.readlines()
        for i in list:
            image_index = i.split(' ')[0]
            label = [int(x) for x in i.split(' ')[1:15]]

            test_man_list.append(image_index)
            test_man_label.append(label)
    
    with open(female_txt, encoding='utf-8') as e: # load train list and train label
        list = e.readlines()
        for i in list:
            image_index = i.split(' ')[0]
            label = [int(x) for x in i.split(' ')[1:15]]
            test_woman_list.append(image_index)
            test_woman_label.append(label)

    img_transforms_M = img_transforms(mode='test', config=config)
    img_transforms_F = img_transforms(mode='test', config=config)
    
    test_dataset_M = NIHchest_dataset(dataset_root=dataset_root, datalist=test_man_list, labellist=test_man_label, img_transforms=img_transforms_M)
    test_dataset_F = NIHchest_dataset(dataset_root=dataset_root, datalist=test_woman_list, labellist=test_woman_label, img_transforms=img_transforms_F)
    test_loader_M = DataLoader(dataset=test_dataset_M, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS, shuffle=False)
    test_loader_F = DataLoader(dataset=test_dataset_F, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS, shuffle=False)
    return test_dataset_M, test_loader_M, test_dataset_F, test_loader_F



def get_patient_age(df, image_index):
    matches = df[df.iloc[:, 0].str.strip() == str(image_index).strip()]
    return float(matches.iloc[0, 4])  # 第五列：Patient Age


def save_NIHchest_age_txt(config):
    testtxt = config.DATA.TEST_LIST
    metacsv = config.METADATA
    save_dir = os.path.dirname(testtxt)

    age_bins = {
        "0_20":   (0, 20),
        "20_40":  (20, 40),
        "40_60":  (40, 60),
        "60_80":  (60, 80),
        "80_plus": (80, float("inf")),
    }

    age_txt_paths = {
        k: os.path.join(save_dir, f"test_age_{k}.txt")
        for k in age_bins
    }

    # 如果全部已存在，直接返回
    if all(os.path.exists(p) for p in age_txt_paths.values()):
        print("Age-based test txt files already exist.")
        return age_txt_paths

    df = pd.read_csv(metacsv)

    age_lines = {k: [] for k in age_bins}

    with open(testtxt, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split(" ")
            image_index = items[0]

            age = get_patient_age(df, image_index)
            print(f"Image: {image_index}, Age: {age}")

            for k, (low, high) in age_bins.items():
                if low <= age < high:
                    age_lines[k].append(line)
                    break

    # 写入 txt
    for k, path in age_txt_paths.items():
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(age_lines[k])
        print(f"Saved {k} age test list to: {path} ({len(age_lines[k])} samples)")

    return age_txt_paths





def build_loader_NIHchest_age(config, ddp):
    """
    Build test dataloaders for different age groups:
    [0,20), [20,40), [40,60), [60,80), >=80
    """

    # 先生成（或直接读取）age-based txt
    age_txts = save_NIHchest_age_txt(config)

    dataset_root = config.DATA.DATA_PATH

    age_loaders = {}

    for age_group, txt_path in age_txts.items():
        test_list = []
        test_label = []

        with open(txt_path, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                image_index = line.split(' ')[0]
                label = [int(x) for x in line.split(' ')[1:15]]

                test_list.append(image_index)
                test_label.append(label)

        img_tf = img_transforms(mode='test', config=config)

        test_dataset = NIHchest_dataset(
            dataset_root=dataset_root,
            datalist=test_list,
            labellist=test_label,
            img_transforms=img_tf
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            shuffle=False
        )

        age_loaders[age_group] = (test_dataset, test_loader)

    return age_loaders




def build_loader_NIHchest_robustness(config):
    dataset_root = config.DATA.DATA_PATH
    testtxt = config.DATA.TEST_LIST
    augmode = config.AUGMODE

    test_list = []
    test_label = []

    with open(testtxt, encoding='utf-8') as e: # load train list and train label
        list = e.readlines()
        for i in list:
            test_list.append(i.split(' ')[0])
            label = [int(x) for x in i.split(' ')[1:15]]
            test_label.append(label)
            
    # no augmentation
    img_transforms_init = img_transforms(mode='test', config=config)
    test_dataset_init = NIHchest_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, img_transforms=img_transforms_init)
    test_loader_init = DataLoader(dataset=test_dataset_init, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS)
    
    # add augmentation to test robustness
    img_transforms_robustness = img_transform_robust(mode='CustomMetalArtifact', config=config)
    ten_crop_transform = img_transform_tencrop(config=config)
    test_dataset_metalartifact = NIHchest_robust_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, albu_transform=img_transforms_robustness, ten_crop_transform=ten_crop_transform)
    test_loader_robustness = DataLoader(dataset=test_dataset_metalartifact, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS)
    
    img_transforms_robustness = img_transform_robust(mode='CustomBrightnessEnhance', config=config)
    ten_crop_transform = img_transform_tencrop(config=config)
    test_dataset_brightnessenhance = NIHchest_robust_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, albu_transform=img_transforms_robustness, ten_crop_transform=ten_crop_transform)
    test_loader_brightnessenhance = DataLoader(dataset=test_dataset_brightnessenhance, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS)
    
    img_transforms_robustness = img_transform_robust(mode='CustomGaussianNoise', config=config)
    ten_crop_transform = img_transform_tencrop(config=config)
    test_dataset_gaussiannoise = NIHchest_robust_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, albu_transform=img_transforms_robustness, ten_crop_transform=ten_crop_transform)
    test_loader_gaussiannoise = DataLoader(dataset=test_dataset_gaussiannoise, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS)
    
    
    return test_dataset_init, test_dataset_metalartifact, test_dataset_brightnessenhance, test_dataset_gaussiannoise