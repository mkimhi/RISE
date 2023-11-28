## RISE 



### Model zoo

**1. COCO Pretraining**

Download pre-trained **instance segmentation backbone** on COCO

| Backbone   | Weights                                                                                     |
| ---------- |-------------------------------------------------------------------------------------------- |
| ResNet-50  | [click](https://drive.google.com/file/d/1ip-pxavMcyWOxfBcl_4cUIBKXRTA4wrO/view?usp=sharing) |
| ResNet-101 | [click](https://drive.google.com/file/d/1Gm162LthxorsS6pMX_XoVTn5iDAAMWbU/view?usp=sharing) |
| Swin-Large | [click](https://drive.google.com/file/d/1o-q4WIcMn_D5p1tSubJBWlPAnJLQ5Cbb/view?usp=sharing) |


**2. Armbench trained models**

You can download the trained models reported in the paper from here

| Backbone                                                     | AP   | AP50 | AP75 | AR1  | AR10 |
| ------------------------------------------------------------ | ---- | ---- | ---- | ---- | ---- | 
| R50 (10%) - TBD                                              | ---- | ---- | ---- | ---- | ---- | 
| ------------------------------------------------------------ | ---- | ---- | ---- | ---- | ---- |





 **Custom training  and validation split** 

We provide the split partitining files: **TBD**


### Training

To train RISE, run:

```
python3 projects/RISE/train_net.py --config-file projects/RISE/configs/XXX.yaml --num-gpus 8 
```



### Inference & Evaluation

Evaluating on Armbench:

```
python3 projects/RISE/train_net.py --config-file projects/RISE/configs/XXX.yaml --num-gpus 8 --eval-only
```



## Citation

AFTER ARXIVE VERSION

## Acknowledgement

This repo is based on [detectron2](https://github.com/facebookresearch/detectron2), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [VisTR](https://github.com/Epiphqny/VisTR), and [IFC](https://github.com/sukjunhwang/IFC)  Thanks for their wonderful works.
