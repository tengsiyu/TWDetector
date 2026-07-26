# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN
import torch.nn as nn
from functools import partial

def add_diffusiondet_config(cfg):
    """
    Add config for DiffusionDet
    """
    cfg.MODEL.DiffusionDet = CN()
    cfg.MODEL.DiffusionDet.NUM_CLASSES = 80
    cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.DiffusionDet.NHEADS = 8
    cfg.MODEL.DiffusionDet.DROPOUT = 0.0
    cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionDet.ACTIVATION = 'relu'
    cfg.MODEL.DiffusionDet.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionDet.NUM_CLS = 1
    cfg.MODEL.DiffusionDet.NUM_REG = 3
    cfg.MODEL.DiffusionDet.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.DiffusionDet.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionDet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionDet.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionDet.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DiffusionDet.USE_FOCAL = True
    cfg.MODEL.DiffusionDet.USE_FED_LOSS = False
    cfg.MODEL.DiffusionDet.ALPHA = 0.25
    cfg.MODEL.DiffusionDet.GAMMA = 2.0
    cfg.MODEL.DiffusionDet.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionDet.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffusionDet.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.DiffusionDet.USE_NMS = True

    # # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify
    
    #efficientnet
    # _C = cfg
    # _C.MODEL.EFFICIENTNET = CN()
    # _C.MODEL.EFFICIENTNET.NAME = "efficientnet_b0"
    # _C.MODEL.EFFICIENTNET.FEATURE_INDICES = [1, 4, 10, 15]
    # _C.MODEL.EFFICIENTNET.OUT_FEATURES = ["stride4", "stride8", "stride16", "stride32"]

    # #lsknet
    # _C = cfg
    # _C.MODEL.LSKNET = CN()
    # _C.MODEL.LSKNET.NAME = "LSKNet"  # 可以根据需要添加不同的变体名称
    # _C.MODEL.LSKNET.FEATURE_INDICES = [1, 2, 3, 4]  # 类似于EfficientNet的特征提取层索引
    # _C.MODEL.LSKNET.OUT_FEATURES = ["stride4", "stride8", "stride16", "stride32"]  # 输出特征图的步幅
    # _C.MODEL.LSKNET.IMG_SIZE = 224  # 输入图像大小
    # _C.MODEL.LSKNET.IN_CHANS = 3  # 输入通道数
    # _C.MODEL.LSKNET.EMBED_DIMS = [64, 128, 256, 512]  # 每个阶段的嵌入维度
    # _C.MODEL.LSKNET.MLP_RATIOS = [8, 8, 4, 4]  # MLP的扩展比例
    # _C.MODEL.LSKNET.DROP_RATE = 0.0  # Dropout率
    # _C.MODEL.LSKNET.DROP_PATH_RATE = 0.0  # DropPath率
    # _C.MODEL.LSKNET.NORM_LAYER = "LN"  # 归一化层
    # _C.MODEL.LSKNET.DEPTHS = [3, 4, 6, 3]  # 每个阶段的块数
    # _C.MODEL.LSKNET.NUM_STAGES = 4  # 阶段数
    # _C.MODEL.LSKNET.PRETRAINED = None  # 预训练模型路径

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
