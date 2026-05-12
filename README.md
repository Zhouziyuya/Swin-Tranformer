# Benchmark for VLM, SSL, SL models based on Swin-Transformer codebase

## Training 

* For classification task, ddp is added in this code to accelerate the training processing via multiple gpus.
You can use several gpus and set `--nproc_per_node=gpu number`:
```
CUDA_VISIBLE_DEVICES="5,6" python -m torch.distributed.launch --nproc_per_node 2 --master_port=25641 poparval_main_ddp.py --img_size 448 --fold 1 --dataset NIHchest
```

* For segmentation task, ddp is not added. `--local_rank` is used to set device number
```
python main_seg.py --backbone vit_base --pretrain_mode vit_seg_selfpatch --pretrain_weight /sda1/zhouziyu/ssl/NIHChestX-ray14_pretrain/checkpoints/SelfPatch_vit-b32_448/checkpoint0300.pth --local_rank 6 --dataset SIIM


## Testing

* For classification testing, Use `--resume` to set your testing checkpoint.
```
python test.py --dataset NIHchest --resume '/sda1/zhouziyu/ssl/downstream_checkpoints/NIHChestX-ray14/popar_adodocar_448_1/best.pth' --device 1
```

* For segmentation testing:
```
python test_seg.py --resume
```
