import os
import albumentations
import albumentations.augmentations.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
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
import pydicom as dicom
from .Augmentation import CustomMetalArtifact, CustomBrightnessEnhance, CustomGaussianNoise
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_loader_CovidQuEx(config, ddp):
    data_root = config.DATA.DATA_PATH
    traintxt = config.DATA.TRAIN_LIST
    valtxt = config.DATA.VAL_LIST
    testtxt = config.DATA.TEST_LIST
    

    train_list = []
    train_mask = []
    val_list = []
    val_mask = []
    test_list = []
    test_mask = []

    if config.MODE == 'train':
        with open(traintxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                i = i.strip()
                train_list.append(i.split(',')[0])
                train_mask.append(i.split(',')[1])
        # train_num = len(train_list)
        # if config.RATIO == '50':
        #     train_list = train_list[:int(train_num/2)]
        # elif config.RATIO == '25':
        #     train_list = train_list[:int(train_num/4)]
        # elif config.RATIO == '1shot':
        #     train_list = [train_list[0]]
        # elif config.RATIO == '5shot':
        #     train_list = train_list[:5]
        # elif config.RATIO == '10shot':
        #     train_list = train_list[:10]
                
        with open(valtxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                i = i.strip()
                val_list.append(i.split(',')[0])
                val_mask.append(i.split(',')[1])


        img_train_transforms = img_transforms(mode='train', config=config)
        train_dataset = CovidQuEx_dataset(data_root=data_root, img_list=train_list, mask_list=train_mask, img_transforms=img_train_transforms, config=config)
        
        img_val_transforms = img_transforms(mode='val', config=config)
        val_dataset = CovidQuEx_dataset(data_root=data_root, img_list=val_list, mask_list=val_mask, img_transforms=img_val_transforms, config=config)
        

        if ddp:
            sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
            sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset)

            train_loader = DataLoader(dataset=train_dataset, 
                                    # sampler=sampler_train,
                                    batch_size=config.DATA.BATCH_SIZE, 
                                    # shuffle=True, 
                                    num_workers=config.DATA.NUM_WORKERS,
                                    drop_last=True,
                                    sampler=sampler_train)

            
            val_loader = DataLoader(dataset=val_dataset, 
                                # sampler=sampler_val,
                                batch_size=config.DATA.BATCH_SIZE, 
                                num_workers=config.DATA.NUM_WORKERS,
                                sampler=sampler_val)
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
                i = i.strip()
                test_list.append(i.split(',')[0])
                test_mask.append(i.split(',')[1])
        img_test_transforms = img_transforms(mode='test', config=config)
        test_dataset = CovidQuEx_dataset(data_root=data_root, img_list=test_list, mask_list=test_mask, img_transforms=img_test_transforms, config=config)
        test_loader = DataLoader(dataset=test_dataset, 
                                batch_size=config.DATA.BATCH_SIZE, 
                                num_workers=config.DATA.NUM_WORKERS)
        return test_dataset, test_loader

class CovidQuEx_dataset(Dataset):
    def __init__(self, data_root, img_list, mask_list, img_transforms, config):
        # super(NIHchest_dataset, self).__init__()
        self.img_transforms = img_transforms
        self.data_root = data_root
        self.img_list = img_list 
        self.mask_list = mask_list 
        self.config = config

    def __getitem__(self, index):
        # print(os.path.join(self.data_root, self.img_list[index]))
        image = cv2.imread(os.path.join(self.data_root, self.img_list[index]))
        
        if len(image.shape) == 2:
            image = np.repeat(image[..., None], 3, 2)

        mask = cv2.imread(os.path.join(self.data_root, self.mask_list[index]))
        mask[mask==255]=1

        transformed = self.img_transforms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'] 
        mask = mask.permute([2,0,1])
        mask = mask[0].unsqueeze(0)

        return image.float(), mask.float()
    # return image 
    def __len__(self):
        return len(self.img_list)

def img_transforms(mode, config):
    if mode == 'train':
        data_transforms = albumentations.Compose([
                                            albumentations.Resize(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                            albumentations.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252]),
                                            albumentations.ShiftScaleRotate(rotate_limit=10),
                                            albumentations.RandomBrightnessContrast(),
                                            ToTensorV2()
                                            ])
        
    elif mode == 'val' or mode == 'test':
        data_transforms = albumentations.Compose([
                                            albumentations.Resize(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                            albumentations.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252]),
                                            ToTensorV2()
                                            ])


    return data_transforms






def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def build_loader_CovidQuEx_cls(config, ddp=False):
    dataset_root = config.DATA.DATA_PATH
    traintxt = config.DATA.TRAIN_LIST_CLS
    valtxt = config.DATA.VAL_LIST_CLS
    testtxt = config.DATA.TEST_LIST_CLS

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
                train_list.append(i.split(',')[0])
                # label = [int(i.split(' ')[1])]
                if int(i.split(',')[1])==0:
                    label=[1,0,0]
                elif int(i.split(',')[1])==1:
                    label=[0,1,0]
                elif int(i.split(',')[1])==2:
                    label=[0,0,1]
                train_label.append(label)

        
        with open(valtxt, encoding='utf-8') as e: # load train list and train label
            list = e.readlines()
            for i in list:
                val_list.append(i.split(',')[0])
                # label = [int(i.split(' ')[1])]
                if int(i.split(',')[1])==0:
                    label=[1,0,0]
                elif int(i.split(',')[1])==1:
                    label=[0,1,0]
                elif int(i.split(',')[1])==2:
                    label=[0,0,1]
                val_label.append(label)

        img_train_transforms = img_transforms_cls(mode='train', config=config)
        train_dataset = CovidQuEx_cls_dataset(dataset_root=dataset_root, datalist=train_list, labellist=train_label, img_transforms=img_train_transforms)
        # print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
        # sampler_train = torch.utils.data.DistributedSampler(
        # train_dataset, num_replicas=1, rank=0, shuffle=True
        # )

        

        img_val_transforms = img_transforms_cls(mode='val', config=config)
        val_dataset = CovidQuEx_cls_dataset(dataset_root=dataset_root, datalist=val_list, labellist=val_label, img_transforms=img_val_transforms)
        # print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")
        # sampler_val = torch.utils.data.distributed.DistributedSampler(
        # val_dataset, shuffle=config.TEST.SHUFFLE
        # )

        if ddp:
            sampler_train = torch.utils.data.DistributedSampler(train_dataset, shuffle=True, seed=fold_seed)
            train_loader = DataLoader(dataset=train_dataset, 
                                sampler=sampler_train,
                                batch_size=config.DATA.BATCH_SIZE, 
                                # shuffle=True, 
                                num_workers=config.DATA.NUM_WORKERS,
                                drop_last=True,
                                worker_init_fn=seed_worker)
            
            sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            val_loader = DataLoader(dataset=val_dataset, 
                            sampler=sampler_val,
                            batch_size=config.DATA.BATCH_SIZE, 
                            num_workers=config.DATA.NUM_WORKERS)
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
                test_list.append(i.split(',')[0])
                # label = [int(i.split(' ')[1])]
                if int(i.split(',')[1])==0:
                    label=[1,0,0]
                elif int(i.split(',')[1])==1:
                    label=[0,1,0]
                elif int(i.split(',')[1])==2:
                    label=[0,0,1]
                test_label.append(label)
        img_test_transforms = img_transforms_cls(mode='test', config=config)
        test_dataset = CovidQuEx_cls_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, img_transforms=img_test_transforms)
        test_loader = DataLoader(dataset=test_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        num_workers=config.DATA.NUM_WORKERS)
        return test_dataset, test_loader

class CovidQuEx_cls_dataset(Dataset):
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

        return image.float(), label
    # return image 
    def __len__(self):
        return len(self.datalist)
    
    
class CovidQuEx_robust_dataset(Dataset):
    def __init__(self, dataset_root, datalist, labellist, albu_transform, ten_crop_transform):
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



def img_transforms_cls(mode, config):
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




def build_loader_CovidQuEx_robustness(config):
    dataset_root = config.DATA.DATA_PATH
    testtxt = config.DATA.TEST_LIST_CLS
    # augmode = config.AUGMODE

    test_list = []
    test_label = []

    with open(testtxt, encoding='utf-8') as e: # load train list and train label
        list = e.readlines()
        for i in list:
            test_list.append(i.split(',')[0])
            # label = [int(i.split(' ')[1])]
            if int(i.split(',')[1])==0:
                label=[1,0,0]
            elif int(i.split(',')[1])==1:
                label=[0,1,0]
            elif int(i.split(',')[1])==2:
                label=[0,0,1]
            test_label.append(label)
            
    # no augmentation
    img_transforms_init = img_transforms_cls(mode='test', config=config)
    test_dataset_init = CovidQuEx_cls_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, img_transforms=img_transforms_init)
    test_loader_init = DataLoader(dataset=test_dataset_init, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS)
    
    # add augmentation to test robustness
    img_transforms_robustness = img_transform_robust(mode='CustomMetalArtifact', config=config)
    ten_crop_transform = img_transform_tencrop(config=config)
    test_dataset_metalartifact = CovidQuEx_robust_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, albu_transform=img_transforms_robustness, ten_crop_transform=ten_crop_transform)
    test_loader_robustness = DataLoader(dataset=test_dataset_metalartifact, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS)
    
    img_transforms_robustness = img_transform_robust(mode='CustomBrightnessEnhance', config=config)
    ten_crop_transform = img_transform_tencrop(config=config)
    test_dataset_brightnessenhance = CovidQuEx_robust_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, albu_transform=img_transforms_robustness, ten_crop_transform=ten_crop_transform)
    test_loader_brightnessenhance = DataLoader(dataset=test_dataset_brightnessenhance, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS)
    
    img_transforms_robustness = img_transform_robust(mode='CustomGaussianNoise', config=config)
    ten_crop_transform = img_transform_tencrop(config=config)
    test_dataset_gaussiannoise = CovidQuEx_robust_dataset(dataset_root=dataset_root, datalist=test_list, labellist=test_label, albu_transform=img_transforms_robustness, ten_crop_transform=ten_crop_transform)
    test_loader_gaussiannoise = DataLoader(dataset=test_dataset_gaussiannoise, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS)
    
    
    return test_dataset_init, test_dataset_metalartifact, test_dataset_brightnessenhance, test_dataset_gaussiannoise






