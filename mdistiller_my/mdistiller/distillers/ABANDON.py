
#底层特征用logits
"""     
#first block logits loss
        f_s1 = feature_student["feats"][1]
        f_t1 = feature_teacher["feats"][1]
        f_s1 = self.conv1_1(f_s1)
        f_t1 = self.conv1_1(f_t1)
        f_s1 = self.conv1_2(f_s1)
        f_t1 = self.conv1_2(f_t1)
        f_s1 = self.avgpool(f_s1)
        f_t1 = self.avgpool(f_t1)
        avg_fs1 = f_s1.reshape(f_s1.size(0), -1)
        avg_ft1 = f_t1.reshape(f_t1.size(0), -1)
        logits_s1 = fc_student(avg_fs1)
        logits_t1 = fc_teacher(avg_ft1)
        loss_kd1 = self.kd1_loss_weight * kd_loss(
            logits_s1, logits_t1, self.temperature
        )

        #second block logits loss
        f_s2 = feature_student["feats"][2]
        f_t2 = feature_teacher["feats"][2]
        f_s2 = self.conv1_2(f_s2)
        f_t2 = self.conv1_2(f_t2)
        f_s2 = self.avgpool(f_s2)
        f_t2 = self.avgpool(f_t2)
        avg_fs2 = f_s2.reshape(f_s2.size(0), -1)
        avg_ft2 = f_t2.reshape(f_t2.size(0), -1)
        logits_s2 = fc_student(avg_fs2)
        logits_t2 = fc_teacher(avg_ft2)
        loss_kd2 = self.kd2_loss_weight * kd_loss(
            logits_s2, logits_t2, self.temperature
        )
"""


#FSP
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def fsp_loss(gs, gt):
    return (gs - gt).pow(2).mean()

def single_stage_fsp_loss(pre_feat, feat):
    b_H, t_H = pre_feat.shape[2], feat.shape[2]
    if b_H > t_H:
        pre_feat = F.adaptive_avg_pool2d(pre_feat, (t_H, t_H))
    elif b_H < t_H:
        feat = F.adaptive_avg_pool2d(feat, (b_H, b_H))
    else:
        pass
    bot = pre_feat.unsqueeze(1)
    top = feat.unsqueeze(2)
    bot = bot.view(bot.shape[0], bot.shape[1], bot.shape[2], -1)
    top = top.view(top.shape[0], top.shape[1], top.shape[2], -1)
    fsp = (bot * top).mean(-1)
    return fsp



class TEST(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(TEST, self).__init__(student, teacher)
        self.temperature = cfg.TEST.TEMPERATURE
        self.ce_loss_weight = cfg.TEST.LOSS.CE_WEIGHT
        self.fsp_loss_weight = cfg.TEST.LOSS.FSP_WEIGHT


    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher, _ = self.teacher(image)

        f_s0 = feature_student["feats"][0]
        f_s1_pre = feature_student["preact_feats"][1]
        f_t0 = feature_teacher["feats"][0]
        f_t1_pre = feature_teacher["preact_feats"][1]
        g_s1 = single_stage_fsp_loss(f_s0, f_s1_pre)
        g_t1 = single_stage_fsp_loss(f_t0, f_t1_pre)
        loss_1 = self.fsp_loss_weight[0] * fsp_loss(g_s1, g_t1)

        f_s1 = feature_student["feats"][1]
        f_s2_pre = feature_student["preact_feats"][2]
        f_t1 = feature_teacher["feats"][1]
        f_t2_pre = feature_teacher["preact_feats"][2]
        g_s2 = single_stage_fsp_loss(f_s1, f_s2_pre)
        g_t2 = single_stage_fsp_loss(f_t1, f_t2_pre)
        loss_2 = self.fsp_loss_weight[1] * fsp_loss(g_s2, g_t2)

        f_s2 = feature_student["feats"][2]
        f_s3_pre = feature_student["preact_feats"][3]
        f_t2 = feature_teacher["feats"][2]
        f_t3_pre = feature_teacher["preact_feats"][3]
        g_s3 = single_stage_fsp_loss(f_s2, f_s3_pre)
        g_t3 = single_stage_fsp_loss(f_t2, f_t3_pre)
        loss_3 = self.fsp_loss_weight[2] * fsp_loss(g_s3, g_t3)   
        
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_fsp = loss_1 + loss_2 + loss_3
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_fsp,
        }
        return logits_student, losses_dict