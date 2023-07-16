#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.    def __init__(self, reduction="none", loss_type="alphaaiou"):

import torch
import torch.nn as nn
import math


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="alphaaiou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        # 求两框左上角较小值和左下角较大值
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == "aiou":
            # 求出预测框左上角右下角
            b1_xy = pred[..., :2]
            b1_wh = pred[..., 2:4]
            b1_wh_half = b1_wh / 2.
            b1_mins = b1_xy - b1_wh_half
            b1_maxes = b1_xy + b1_wh_half
            # 求出真实框左上角右下角
            b2_xy = target[..., :2]
            b2_wh = target[..., 2:4]
            b2_wh_half = b2_wh / 2.
            b2_mins = b2_xy - b2_wh_half
            b2_maxes = b2_xy + b2_wh_half

            # 避免两框不存在，给下面提供0矩阵用
            intersect_maxes = torch.min(b1_maxes, b2_maxes)

            # 计算中心的差距
            center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

            # 找到包裹两个框的最小框的左上角和右下角
            enclose_mins = torch.min(b1_mins, b2_mins)
            enclose_maxes = torch.max(b1_maxes, b2_maxes)
            enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
            # 计算对角线距离
            enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
            ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
                b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
                b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
            alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
            ciou = ciou - alpha * v

            # 计算aiou添加项
            # 预测框与真实框左上角距离，左上角横坐标差值为zuox, 左上角纵坐标差值为zuoy, aiou1为左上角距离的平方
            zuox = (b2_mins[..., 0] - b1_mins[..., 0]) * (b2_mins[..., 0] - b1_mins[..., 0])
            zuoy = (b2_mins[..., 1] - b1_mins[..., 1]) * (b2_mins[..., 1] - b1_mins[..., 1])
            aiou1 = zuox + zuoy
            # 预测框与真实框右下角距离，右下角横坐标差值为youx, 右下角纵坐标差值为youy, aiou2为右下角距离的平方
            youx = (b2_maxes[..., 0] - b1_maxes[..., 0]) * (b2_maxes[..., 0] - b1_maxes[..., 0])
            youy = (b2_maxes[..., 1] - b1_maxes[..., 1]) * (b2_maxes[..., 1] - b1_maxes[..., 1])
            aiou2 = youx + youy
            # aiou为两框两点绝对距离差值的平方, 0.05为平衡系数
            aiou = (aiou1 + aiou2) * 1
            aiou = aiou / torch.clamp(enclose_diagonal, min=1e-6)

            # 添加aiou项
            ciou = ciou - aiou

            loss = 1 - ciou

        elif self.loss_type == "ciou":
            # 求出预测框左上角右下角
            b1_xy = pred[..., :2]
            b1_wh = pred[..., 2:4]
            b1_wh_half = b1_wh / 2.
            b1_mins = b1_xy - b1_wh_half
            b1_maxes = b1_xy + b1_wh_half
            # 求出真实框左上角右下角
            b2_xy = target[..., :2]
            b2_wh = target[..., 2:4]
            b2_wh_half = b2_wh / 2.
            b2_mins = b2_xy - b2_wh_half
            b2_maxes = b2_xy + b2_wh_half

            # 避免两框不存在，给下面提供0矩阵用
            intersect_maxes = torch.min(b1_maxes, b2_maxes)

            # 计算中心的差距
            center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

            # 找到包裹两个框的最小框的左上角和右下角
            enclose_mins = torch.min(b1_mins, b2_mins)
            enclose_maxes = torch.max(b1_maxes, b2_maxes)
            enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
            # 计算对角线距离
            enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
            ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
                b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
                b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
            alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
            ciou = ciou - alpha * v

            loss = 1 - ciou

        elif self.loss_type == "alphaiou":
            # 求出预测框左上角右下角
            b1_xy = pred[..., :2]
            b1_wh = pred[..., 2:4]
            b1_wh_half = b1_wh / 2.
            b1_mins = b1_xy - b1_wh_half
            b1_maxes = b1_xy + b1_wh_half
            # 求出真实框左上角右下角
            b2_xy = target[..., :2]
            b2_wh = target[..., 2:4]
            b2_wh_half = b2_wh / 2.
            b2_mins = b2_xy - b2_wh_half
            b2_maxes = b2_xy + b2_wh_half

            # 避免两框不存在，给下面提供0矩阵用
            intersect_maxes = torch.min(b1_maxes, b2_maxes)

            # 计算中心的差距
            center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

            # 找到包裹两个框的最小框的左上角和右下角
            enclose_mins = torch.min(b1_mins, b2_mins)
            enclose_maxes = torch.max(b1_maxes, b2_maxes)
            enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
            # 计算对角线距离
            enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
            ciou = iou**3 - 1.0 * (center_distance**3) / torch.clamp(enclose_diagonal**3, min=1e-6)

            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
                b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
                b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
            alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
            ciou = ciou - (alpha * v)**3

            loss = 1 - ciou

        elif self.loss_type == "alphaaiou":
            # 求出预测框左上角右下角
            b1_xy = pred[..., :2]
            b1_wh = pred[..., 2:4]
            b1_wh_half = b1_wh / 2.
            b1_mins = b1_xy - b1_wh_half
            b1_maxes = b1_xy + b1_wh_half
            # 求出真实框左上角右下角
            b2_xy = target[..., :2]
            b2_wh = target[..., 2:4]
            b2_wh_half = b2_wh / 2.
            b2_mins = b2_xy - b2_wh_half
            b2_maxes = b2_xy + b2_wh_half

            # 避免两框不存在，给下面提供0矩阵用
            intersect_maxes = torch.min(b1_maxes, b2_maxes)

            # 计算中心的差距
            center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

            # 找到包裹两个框的最小框的左上角和右下角
            enclose_mins = torch.min(b1_mins, b2_mins)
            enclose_maxes = torch.max(b1_maxes, b2_maxes)
            enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
            # 计算对角线距离
            enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
            ciou = iou**1.5 - 1.0 * (center_distance**1.5) / torch.clamp(enclose_diagonal**1.5, min=1e-6)

            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
                b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
                b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
            alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
            ciou = ciou - (alpha * v)**1.5

            # 计算aiou添加项
            # 预测框与真实框左上角距离，左上角横坐标差值为zuox, 左上角纵坐标差值为zuoy, aiou1为左上角距离的平方
            zuox = (b2_mins[..., 0] - b1_mins[..., 0]) * (b2_mins[..., 0] - b1_mins[..., 0])
            zuoy = (b2_mins[..., 1] - b1_mins[..., 1]) * (b2_mins[..., 1] - b1_mins[..., 1])
            aiou1 = zuox + zuoy
            # 预测框与真实框右下角距离，右下角横坐标差值为youx, 右下角纵坐标差值为youy, aiou2为右下角距离的平方
            youx = (b2_maxes[..., 0] - b1_maxes[..., 0]) * (b2_maxes[..., 0] - b1_maxes[..., 0])
            youy = (b2_maxes[..., 1] - b1_maxes[..., 1]) * (b2_maxes[..., 1] - b1_maxes[..., 1])
            aiou2 = youx + youy
            # aiou为两框两点绝对距离差值的平方, 0.05为平衡系数
            aiou = (aiou1**1.5 + aiou2**1.5) * 1
            aiou = aiou / torch.clamp(enclose_diagonal**1.5, min=1e-6)

            # 添加aiou项
            ciou = ciou - aiou

            loss = 1 - ciou


        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss