
import csv
import sys
import cv2
import numpy as np
import torch
import argparse
import json
import os
import csv
from tqdm import tqdm
from models import build_model
from torchvision.transforms import Compose, Normalize, ToTensor
from configs.config_ChestXdet import get_config_ChestXdet
from configs.config_NIHchest import get_config_NIHchest
from models.upernet import UperNet_swin

from pytorch_grad_cam import GradCAM, \
                            ScoreCAM, \
                            GradCAMPlusPlus, \
                            AblationCAM, \
                            XGradCAM, \
                            EigenCAM, \
                            EigenGradCAM, \
                            LayerCAM, \
                            FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default='configs/swin/swin_base_patch4_window7_224.yaml', help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')
    # easy config modification
    parser.add_argument('--backbone', type=str, default='swin_base', help='swin_base, vit_base, resnet50')
    parser.add_argument('--model_type', type=str, default='swinv2', help='swin, swinv2')
    parser.add_argument('--batch-size', type=int,default=30, help="batch size for single GPU") # default=128 for 224 img_size, 32 for 448 img_size
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=500)
    # parser.add_argument('--dataset', type=str,default='ChestXdet', help="JSRT, ChestXdet")
    parser.add_argument('--img_size', type=int, default=512, help='image size of downstream task')
    # parser.add_argument('--pretrain_mode', type=str, default='compose_12N', help='popar_pec_seg, seg_simmim,seg_simmim_global,simmim_global_infonce,simmim_global_barlow')
    # compose_12N
    parser.add_argument('--fold', type=str,default='1', help="10 split of NIHchest dataset")
    # parser.add_argument('--pretrain_weight', type=str, default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/swinv2_fromIN_unique_multiscale_consis_compdecomp_25epcswinv2_512_1/best.pth')
    parser.add_argument('--pretrain_weight', type=str, default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/large_swinv2_fromIN_unique_multiscale_consis_compdecomp_20epcswinv2_512_1/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/fromIN_unique_multiscale_consis_compdecomp_100epcswin_448_1/best.pth')
    # parser.add_argument('--pretrain_weight', type=str, default='/sda/zhouziyu/ssl/downstream_checkpoints/NIHchest/onebranch_3component_24epc_swin_base_448_1/best.pth')

    parser.add_argument('--ratio', type=str, default='100', help='100, 50, 25, 5shot, 10shot')
    parser.add_argument('--seg_part', type=str, default='all', help='all, lung, heart, clavicle')
    parser.add_argument('--mode', type=str, default='train', help='mode: train, val or test')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    # parser.add_argument('--output', default='/sda1/zhouziyu/ssl/downstream_checkpoints/ChestXdet', type=str, metavar='PATH')
    # parser.add_argument('--bbox_file', default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/BBox_List_2017.csv', type=str, metavar='PATH')
    parser.add_argument('--bbox_file', default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/BBox_List_2017_onebox.csv', type=str, metavar='PATH')
    parser.add_argument('--test_path', default='/sda/zhouziyu/ssl/datasets/ChestXray/NIHChestX-ray14/images_withBBox/images_all/', type=str, metavar='PATH')
    parser.add_argument('--out_path', default='/nvme1n1/zhouziyu/Swin-Transformer/figures/gradcam/ACEv2_swinv2_large_init/', type=str, metavar='PATH')


    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    parser.add_argument('--master_port', type=str, default='12345')

    # args, unparsed = parser.parse_known_args()
    # args = parser.parse_args([])
    args = parser.parse_args()

    config = get_config_NIHchest(args)
    
    # config = get_config(args)

    return args, config


def bbox_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 转换为(x_min, y_min, x_max, y_max)
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # 计算并集
    union_area = w1 * h1 + w2 * h2 - inter_area

    # 计算IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou


def read_csv(filename, train_size = 448, init_shape = False):
    data = {}
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            # image_name, x, y, w, h = row
            image_name = row[0]
            if len(row) < 5:
                print(f"警告: {image_name}数据不足，跳过")
                data[image_name] = []
                continue
            x, y, w, h = row[-4:]
            if init_shape:
                x = float(x)*train_size/1024
                y = float(y)*train_size/1024
                w = float(w)*train_size/1024
                h = float(h)*train_size/1024
            
            if image_name not in data:
                data[image_name] = []
            data[image_name].append((int(float(x)), int(float(y)), int(float(w)), int(float(h))))
    return data


def reshape_transform(tensor, height=14, width=14): # swin
        # 去掉cls token
        # result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # 将通道维度放到第一个位置
        result = result.transpose(2, 3).transpose(1, 2) # [1,1024,14,14]
        # result = result.transpose(2, 3).transpose(1, 2) # [1,1024,14,14]
        return result

def reshape_transform_v2(tensor, height=16, width=16):
        # 去掉cls token
        # result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # 将通道维度放到第一个位置
        result = result.transpose(2, 3).transpose(1, 2) # [1,1024,14,14]
        # result = result.transpose(2, 3).transpose(1, 2) # [1,1024,14,14]
        return result

def reshape_transform_vit(tensor, height=14, width=14):
        # 去掉cls token
        # print(tensor.shape) # [1, 197, 768]
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        # result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

        # 将通道维度放到第一个位置
        result = result.transpose(2, 3).transpose(1, 2) # [1,1024,14,14]
        # result = result.transpose(2, 3).transpose(1, 2) # [1,1024,14,14]
        return result

def draw_bbox(img_path, bbox_file):
    with open(bbox_file, 'r') as f:
        reader = csv.reader(f)
    # print(reader)
        for row in reader:
            if row[0] == 'Image Index':
                continue
            img_name = row[0]
            # print(img_name)
            x1,y1 = int(float(row[-4])*448/1024), int(float(row[-3])*448/1024)
            x2,y2 = x1+int(float(row[-2])*448/1024), y1+int(float(row[-1])*448/1024)
            image = cv2.imread(os.path.join(img_path, img_name))
            # print(os.path.join(img_path, img_name))
            # cv2.rectangle(image, (int(float(row[2])), int(float(row[3]))), (int(float(row[2]))+int(float(row[4])), int(float(row[3]))+int(float(row[5]))), (255, 0, 255), 3)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 3) # white
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.imwrite(os.path.join(img_path,img_name), image)


def draw_bbox2(img_path, save_path, bbox_file):
    with open(bbox_file, 'r') as f:
        reader = csv.reader(f)
    # print(reader)
        for row in reader:
            img_name = row[0]
            if row[0] == 'Image Index':
                continue
            if len(row) < 5:
                print(f"警告: 行数据不足，跳过 {img_name}")
                continue
            
            # print(img_name)
            x1,y1 = int(float(row[-4])), int(float(row[-3]))
            x2,y2 = x1+int(float(row[-2])), y1+int(float(row[-1]))
            image = cv2.imread(os.path.join(img_path, img_name))
            # print(os.path.join(img_path, img_name))
            # cv2.rectangle(image, (int(float(row[2])), int(float(row[3]))), (int(float(row[2]))+int(float(row[4])), int(float(row[3]))+int(float(row[5]))), (255, 0, 255), 3)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.imwrite(os.path.join(save_path,img_name), image)


def get_box(grad_cam):
    # 生成gradcam的bounding box
    # 假设grad_cam是已经生成的Grad-CAM热图，且是灰度图像
    # 标准化热图

    # grad_cam = cv2.normalize(grad_cam, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # _, thresholded = cv2.threshold(grad_cam, int(255 * 0.9), 255, cv2.THRESH_BINARY)
    
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
    thresholded = np.uint8(grad_cam * 255)
    thresholded[thresholded<80]=0
    thresholded[thresholded>=80]=255
    
    # 找到轮廓
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # 设置面积阈值
    # area_threshold = 100  # 举例，可以根据需要调整
    # # 过滤掉小于阈值的轮廓
    # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= area_threshold]
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
    else:
        max_contour = None  # 或者适当的默认处理

    # 计算每个轮廓的边界框
    if max_contour is not None:
        # bounding_boxes = [cv2.boundingRect(c) for c in max_contour]
        bounding_boxes = cv2.boundingRect(max_contour)
    else:
        bounding_boxes=None
    # print(bounding_boxes)
    # sys.exit(1)
    return bounding_boxes


def generate_bboxes(heatmap, low_thresh=60, high_thresh=180, min_area=10):
    """
    使用轮廓检测的双阈值方法
    """
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 创建二值掩码
    _, low_mask = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    _, high_mask = cv2.threshold(heatmap, high_thresh, 255, cv2.THRESH_BINARY)
    
    # 在低阈值区域内寻找高阈值区域
    final_mask = cv2.bitwise_and(high_mask, low_mask)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 如果没有找到轮廓，返回空列表
    if not contours:
        return []
    
    # 找到面积最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 计算最大轮廓的面积
    max_area = cv2.contourArea(max_contour)
    
    # 如果最大轮廓面积小于最小面积阈值，返回空列表
    if max_area < min_area:
        return []
    
    # 获取最大轮廓的边界框
    bbox = cv2.boundingRect(max_contour)
    
    return bbox



def main():
    device = torch.device('cuda', 5)
    args, config = parse_option()
    # 加载预训练的 ViT 模型
    model = build_model(config)
    # model = UperNet_swin(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES)
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

    state_dict = checkpoint['model']

    # with open('/data/zhouziyu/home3/zhouziyu/warmup/sslpretrain/Swin-Transformer/model_keys/resnet50.txt', 'w') as f:
    #     for i in range(len(list(state_dict.keys()))):
    #         f.writelines(list(state_dict.keys())[i]+'\n')
    # # sys.exit(1)

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # model.eval()

    # 判断是否使用 GPU 加速
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.to(device)


    
    if config.BACKBONE == 'swin_base':
        if config.MODEL.TYPE == 'swin':
            target_layer = [model.layers[-1].blocks[-1].norm2]
            cam = GradCAM(model=model,
                    target_layers=target_layer,
                    # use_cuda=device,
                    reshape_transform=reshape_transform)
        elif config.MODEL.TYPE == 'swinv2':
            target_layer = [model.layers[-1].blocks[-1].norm2]
            cam = GradCAM(model=model,
                target_layers=target_layer,
                # use_cuda=device,
                reshape_transform=reshape_transform_v2)
    elif config.BACKBONE == 'resnet50':
        target_layer = [model.layer4[-1]]
        cam = GradCAM(model=model,
                target_layers=target_layer,
                use_cuda=device)
    elif config.BACKBONE == 'vit_base':
        target_layer = [model.blocks[-1].norm1]
        cam = GradCAM(model=model,
                target_layers=target_layer,
                use_cuda=device,
                    reshape_transform=reshape_transform_vit)
        
    file =  open(os.path.join(args.out_path, 'gradcam_box.csv'), mode='w', newline='')
    writer = csv.writer(file)
    # 写入标题行
    writer.writerow(['Image Index', 'X', 'Y', 'W', 'H'])
    
    save_path_red_heatmap_with_gtbox = os.path.join(args.out_path, 'pred_heatmap_with_gtbox/')
    if not os.path.exists(save_path_red_heatmap_with_gtbox):
        os.makedirs(save_path_red_heatmap_with_gtbox)

    # 读取输入图像
    testlist = os.listdir(args.test_path)

    for i in testlist:
        rgb_img = cv2.imread(args.test_path+i)
        # rgb_img = rgb_img[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (args.img_size, args.img_size))
        rgb_img = np.float32(rgb_img)/255

        # 预处理图像
        input_tensor = preprocess_image(rgb_img, mean=[0.5056, 0.5056, 0.5056], std=[0.252, 0.252, 0.252]), 
        # input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 

        if use_cuda:
            input_tensor = input_tensor[0].to(device)


        # 计算 grad-cam
        
        target_category = None # 可以指定一个类别，或者使用 None 表示最高概率的类别
        grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
        grayscale_cam = grayscale_cam[0, :]

        # gradcam_savedir = args.out_path
        # cv2.imwrite(os.path.join(save_path_red_heatmap_with_gtbox,i), grayscale_cam*255)

        # gradcam_box = get_box(grayscale_cam)
        gradcam_box = generate_bboxes(grayscale_cam, low_thresh=60, high_thresh=80, min_area=10)
        # if i=='00000032_037.png':
            # print(gradcam_box)
        if gradcam_box is not None:
            writer.writerow([i,*gradcam_box])
            # for box in gradcam_box:
            #     writer.writerow([i, *box])  # 展开box元组

        # 将 grad-cam 的输出叠加到原始图像上
        visualization = show_cam_on_image(rgb_img, grayscale_cam)

        # 保存可视化结果
        cv2.imwrite(save_path_red_heatmap_with_gtbox+i, visualization)
    
    draw_bbox(save_path_red_heatmap_with_gtbox, args.bbox_file)
    
    save_path_red_heatmap_with_predbox = os.path.join(args.out_path, 'pred_heatmap_with_predbox/')
    if not os.path.exists(save_path_red_heatmap_with_predbox):
        os.makedirs(save_path_red_heatmap_with_predbox)
    draw_bbox2(save_path_red_heatmap_with_gtbox, save_path_red_heatmap_with_predbox, os.path.join(args.out_path, 'gradcam_box.csv'))


    


if __name__ == "__main__":
    main()
    
    args, config = parse_option()
     # 读取CSV文件
    pred_boxes = read_csv(os.path.join(args.out_path, 'gradcam_box.csv'), args.img_size)
    gt_boxes = read_csv(args.bbox_file, args.img_size, init_shape=True)

    # 计算准确度
    iou_threshold = 0.1
    correct_predictions = 0
    total_predictions = 0

    for image_name, boxes in pred_boxes.items():
        if len(boxes) == 0:
            total_predictions += 1
            continue
        for box in boxes:
            total_predictions += 1
            if image_name in gt_boxes:
                for gt_box in gt_boxes[image_name]:
                    # print(gt_box)
                    if bbox_iou(box, gt_box) >= iou_threshold:
                        # print(image_name)
                        correct_predictions += 1
                        break

    # 计算并输出准确度
    print(correct_predictions, total_predictions)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")
    

