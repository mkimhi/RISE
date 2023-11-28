# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_rise_config(cfg):
    """
    Add config for RISE.
    """
    cfg.MODEL.RISE = CN()
    cfg.MODEL.RISE.NUM_CLASSES = 80

    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 1
    cfg.INPUT.SAMPLING_FRAME_RANGE = 10
    cfg.INPUT.SAMPLING_INTERVAL = 1
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"
    cfg.INPUT.COCO_PRETRAIN = False
    cfg.INPUT.PRETRAIN_SAME_CROP = False

    # LOSS
    cfg.MODEL.RISE.MASK_WEIGHT = 2.0
    cfg.MODEL.RISE.DICE_WEIGHT = 5.0
    cfg.MODEL.RISE.GIOU_WEIGHT = 2.0
    cfg.MODEL.RISE.L1_WEIGHT = 5.0
    cfg.MODEL.RISE.CLASS_WEIGHT = 2.0
    cfg.MODEL.RISE.REID_WEIGHT = 2.0
    cfg.MODEL.RISE.DEEP_SUPERVISION = True
    cfg.MODEL.RISE.MASK_STRIDE = 4
    cfg.MODEL.RISE.MATCH_STRIDE = 4
    cfg.MODEL.RISE.FOCAL_ALPHA = 0.25

    cfg.MODEL.RISE.SET_COST_CLASS = 2
    cfg.MODEL.RISE.SET_COST_BOX = 5
    cfg.MODEL.RISE.SET_COST_GIOU = 2

    # TRANSFORMER
    cfg.MODEL.RISE.NHEADS = 8
    cfg.MODEL.RISE.DROPOUT = 0.1
    cfg.MODEL.RISE.DIM_FEEDFORWARD = 1024
    cfg.MODEL.RISE.ENC_LAYERS = 6
    cfg.MODEL.RISE.DEC_LAYERS = 6

    cfg.MODEL.RISE.HIDDEN_DIM = 256
    cfg.MODEL.RISE.NUM_OBJECT_QUERIES = 300
    cfg.MODEL.RISE.DEC_N_POINTS = 4
    cfg.MODEL.RISE.ENC_N_POINTS = 4
    cfg.MODEL.RISE.NUM_FEATURE_LEVELS = 4


    # Evaluation
    cfg.MODEL.RISE.CLIP_STRIDE = 1
    cfg.MODEL.RISE.MERGE_ON_CPU = True
    cfg.MODEL.RISE.MULTI_CLS_ON = True
    cfg.MODEL.RISE.APPLY_CLS_THRES = 0.05

    cfg.MODEL.RISE.TEMPORAL_SCORE_TYPE = 'mean' # mean or max score for sequence masks during inference,
    cfg.MODEL.RISE.INFERENCE_SELECT_THRES = 0.1  # 0.05 for ytvis
    cfg.MODEL.RISE.NMS_PRE =  0.5
    cfg.MODEL.RISE.ADD_NEW_SCORE = 0.2
    cfg.MODEL.RISE.INFERENCE_FW = True #frame weight
    cfg.MODEL.RISE.INFERENCE_TW = True  #temporal weight
    cfg.MODEL.RISE.MEMORY_LEN = 3
    cfg.MODEL.RISE.BATCH_INFER_LEN = 10

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    ## support Swin backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # find_unused_parameters
    cfg.FIND_UNUSED_PARAMETERS = True