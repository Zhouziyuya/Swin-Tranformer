# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swin_transformer_moe import SwinTransformerMoE
from .swin_mlp import SwinMLP
from .simmim import build_simmim
import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.swin_transformer import SwinTransformer as TimmSwinTransformer
from functools import partial
import torchvision.models as models
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification, AutoModel
import models.convnext as convnext
# from models import models_eva # infer EVA时打开
from timm.models import create_model
# from .jepa_vit import vit_predictor
from . import jepa_vit
import ipdb
from . import vision_transformer as vits



def build_model(config, is_pretrain=False):
    backbone = config.BACKBONE
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if is_pretrain:
        model = build_simmim(config)
        return model

    if backbone == 'swin_base':
        if model_type == 'swin':
            model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                    depths=config.MODEL.SWIN.DEPTHS,
                                    num_heads=config.MODEL.SWIN.NUM_HEADS,
                                    window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN.APE,
                                    norm_layer=layernorm,
                                    patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    fused_window_process=config.FUSED_WINDOW_PROCESS)
        elif model_type == 'swinv2':
            model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                    in_chans=config.MODEL.SWINV2.IN_CHANS,
                                    num_classes=config.MODEL.NUM_CLASSES,
                                    embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                    depths=config.MODEL.SWINV2.DEPTHS,
                                    num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                    window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWINV2.APE,
                                    patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                    pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
        
        
        else:
            raise NotImplementedError(f"Unkown model: {model_type}")
        
    elif backbone == 'swin_large': # for Ark_plus, imgae size 768
        # ipdb.set_trace()
        
        model = SwinTransformer(num_classes=config.MODEL.NUM_CLASSES, img_size = config.DATA.IMG_SIZE,
                patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))
        
    elif backbone=='vit_base':
        model = VisionTransformer(num_classes=config.MODEL.NUM_CLASSES, img_size=config.DATA.IMG_SIZE,
                        patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6))
        model.default_cfg = _cfg()

    elif backbone=="vit_base_patchsize16":
        if config.PRETRAIN_MODE in ['ce_clip_itm_vitbps16','ce_clip_vitbps16']:
            model = ViTForImageClassification.from_pretrained(
                '/mnt/sda/zhouziyu/ssl/pretrained_model/huggingface/vit-base-patch16-224-in21k',
                num_labels=config.MODEL.NUM_CLASSES)
        elif config.PRETRAIN_MODE in ['RAD-DINO']:
            base_model = AutoModel.from_pretrained('/sda/zhouziyu/ssl/pretrained_model/huggingface/rad-dino')
            model = ClassificationModel(base_model, num_labels=config.MODEL.NUM_CLASSES)
        elif 'DINOv2' in config.PRETRAIN_MODE:
            base_model = AutoModel.from_pretrained('/sda/zhouziyu/ssl/pretrained_model/huggingface/dinov2-base')
            model = ClassificationModel(base_model, num_labels=config.MODEL.NUM_CLASSES)
        elif 'eva-x' in config.PRETRAIN_MODE:
            from models import models_eva
            model = create_model(
            'eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE',
            pretrained=False,
            img_size=224,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=0,
            drop_path_rate=0.2,
            attn_drop_rate=0,
            drop_block_rate=None,
            use_mean_pooling=True,
            use_checkpoint=False,
            stop_grad_conv1=False,
        )
        elif 'CheXWorld' in config.PRETRAIN_MODE:
            encoder = jepa_vit.__dict__['vit_base'](
            img_size=224,
            patch_size=16,
            drop_path_rate=0.0)
            feature_dim = jepa_vit.VIT_EMBED_DIMS.get('vit_base', 768)

            model = FineTuner(encoder, feature_dim=feature_dim, num_classes=config.MODEL.NUM_CLASSES)
        
        else:
            model = VisionTransformer(num_classes=config.MODEL.NUM_CLASSES, img_size=config.DATA.IMG_SIZE,
                            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            model.default_cfg = _cfg()

    elif backbone=="vit_huge_patchsize14":
        model = VisionTransformer(num_classes=config.MODEL.NUM_CLASSES, img_size=config.DATA.IMG_SIZE,
                                  patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
                                  qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
    elif backbone=='vit_large':
        if 'CheXFound' in config.PRETRAIN_MODE:
            vit_kwargs = dict(
            img_size=512,
            num_classes=config.MODEL.NUM_CLASSES,
            patch_size=16,
            init_values=1.0e-05,
            ffn_layer="mlp",
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            num_register_tokens=0,
            interpolate_offset=0.1,
            interpolate_antialias=False,
        )
            model = vits.__dict__['vit_large'](**vit_kwargs)
    
    elif backbone=='resnet50':
        model = models.__dict__['resnet50'](pretrained=False)
        kernelCount = model.fc.in_features
        model.fc = nn.Linear(kernelCount, config.MODEL.NUM_CLASSES)
        
    elif backbone=='convnext':
        model = convnext.__dict__['convnext_base'](num_classes = config.MODEL.NUM_CLASSES)

    return model




class ClassificationModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(ClassificationModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, x):
        outputs = self.base_model(x)
        pooled_output = outputs.pooler_output  # 获取池化后的输出
        logits = self.classifier(pooled_output)
        return logits


class FineTuner(nn.Module):
    def __init__(self, feature_model, feature_dim, num_classes, with_cls_token=False):
        super().__init__()
        self.feature_model = feature_model
        self.num_classes = num_classes
        self.with_cls_token = with_cls_token
        self.head = nn.Linear(feature_dim, num_classes)
        
    
    def forward(self, x):
        features = self.feature_model(x)
        
        if self.with_cls_token:
            features = features[:, 1:].mean(dim=1)
        else:
            features = features.mean(dim=1)
        return self.head(features)
    

