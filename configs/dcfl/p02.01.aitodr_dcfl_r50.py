'''
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.123
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.367
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.045
Average Precision  (AP) @[ IoU=0.50:0.95 | area=very tiny | maxDets=1500 ] = 0.043
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.107
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.172
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.222
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.239
Average Recall     (AR) @[ IoU=0.50:0.95 | area=very tiny | maxDets=1500 ] = 0.072
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.237
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.293
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.312
2024-06-23 06:17:55,238 - mmrotate - INFO - Exp name: p02.01.aitodr_dcfl_r50.py
2024-06-23 06:17:55,248 - mmrotate - INFO - Epoch(val) [12][13940]      mAP_AP: 0.1230, mAP_AP_50: 0.3670, mAP_AP_75: 0.0450, mAP_AP_vt: 0.0430, mAP_AP_t: 0.1070, mAP_AP_s: 0.1720, mAP_AP_m: 0.2220, mAP_mAP_copypaste: 0.123 0.367 0.045 0.043 0.107 0.172 0.222
'''
_base_ = [
    '../_base_/datasets/aitodr.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le135'
model = dict(
    type='RotatedRetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RDCFLHead',
        num_classes=8,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        dcn_assign = True,
        dilation_rate = 3,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=1, 
            ratios=[1.0], 
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        reg_decoded_bbox=True,
        loss_bbox=dict(
            type='RotatedIoULoss',
            loss_weight=1.0)
        ), 
    train_cfg=dict(
        assigner=dict(
            type='C2FAssigner',
            ignore_iof_thr=-1,
            gpu_assign_thr= 1024,
            iou_calculator=dict(type='RBboxMetrics2D'),
            assign_metric='gjsd',
            topk=16,
            topq=12,
            constraint='dgmm',
            gauss_thr=0.8),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.05, 
        nms=dict(iou_thr=0.4), 
        max_per_img=3000))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(interval=4)
evaluation = dict(interval=12, metric='mAP')