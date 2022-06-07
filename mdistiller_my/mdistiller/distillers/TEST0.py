import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


def single_stage_tat_loss(f_s, f_t, f_s_o):
    #同论文假设s和t hwc 都相等
    bsz = f_s.size(0)
    c = f_s.size(1)
    f_s = f_s.reshape(bsz, c, -1).transpose(1, 2)
    f_t = f_t.reshape(bsz, c, -1).transpose(1, 2)
    f_s_o = f_s_o.reshape(bsz, c, -1).transpose(1, 2)
    f_t_t = f_t.transpose(1, 2)
    tem = F.softmax(torch.bmm(f_s, f_t_t).transpose(1, 2), dim=2)
    f_ns = torch.bmm(tem, f_s_o)
    return (f_ns - f_t).pow(2).mean()


def tat_loss(g_s, g_t, g_s_o):
    return sum([single_stage_tat_loss(f_s, f_t, f_s_o) for f_s, f_t, f_s_o in zip(g_s, g_t, g_s_o)])


def do_patches(f):
    f1, f2 = f.split([4, 4], dim=2)
    f3, f4 = f1.split([4, 4], dim=3)
    f5, f6 = f2.split([4, 4], dim=3)
    f_g1 = torch.cat([f3, f4], dim=1)
    f_g2 = torch.cat([f5, f6], dim=1)
    f_r = [f_g1, f_g2]
    return f_r


def do_patches2(f):
    avg = nn.AvgPool2d(2, 2)
    f1, f2 = f.split([4, 4], dim=2)
    f3, f4 = f1.split([4, 4], dim=3)
    f5, f6 = f2.split([4, 4], dim=3)
    f3 = avg(f3)
    f4 = avg(f4)
    f5 = avg(f5)
    f6 = avg(f6)
    f7 = torch.cat([f3, f4], dim=3)
    f8 = torch.cat([f5, f6], dim=3)
    f_r = torch.cat([f7, f8], dim=2)
    f_r = [f_r]
    return f_r


    """
    h=w=2,g=p=4

    f1, f2 = f.split([4, 4], dim=2)
    f3, f4 = f1.split([2, 2], dim=2)
    f5, f6 = f3.split([4, 4], dim=3)
    f7, f8 = f5.split([2, 2], dim=3)
    f9, f10 = f6.split([2, 2], dim=3)
    f11, f12 = f4.split([4, 4], dim=3)
    f13, f14 = f11.split([2, 2], dim=3)
    f15, f16 = f12.split([2, 2], dim=3)
    f17, f18 = f2.split([2, 2], dim=2)
    f19, f20 = f17.split([4, 4], dim=3)
    f21, f22 = f19.split([2, 2], dim=3)
    f23, f24 = f20.split([2, 2], dim=3)
    f25, f26 = f18.split([4, 4], dim=3)
    f27, f28 = f25.split([2, 2], dim=3)
    f29, f30 = f26.split([2, 2], dim=3)
    f_g1 = torch.cat([f7, f8, f9, f10], dim=1)
    f_g2 = torch.cat([f13, f14, f15, f16], dim=1)
    f_g3 = torch.cat([f21, f22, f23, f24], dim=1)
    f_g4 = torch.cat([f27, f28, f29, f30], dim=1)
    f_r = [f_g1, f_g2, f_g3, f_g4]
    return f_r
    """





class TEST0(Distiller):
    """fu'xian TaT"""

    def __init__(self, student, teacher, cfg):
        super(TEST0, self).__init__(student, teacher)
        self.temperature = cfg.TEST.TEMPERATURE
        self.task_loss_weight = cfg.TEST.LOSS.TASK_WEIGHT
        self.kl_loss_weight = cfg.TEST.LOSS.KL_WEIGHT
        self.tat_loss_weight = cfg.TEST.LOSS.TAT_WEIGHT
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)


    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher, _ = self.teacher(image)


        # losses
        f_s = feature_student["feats"][1:]
        f_t = feature_teacher["feats"][1:]
        l_fs = []
        l_fso = []
        f_s_l0 = self.conv4(f_s[0])
        f_s_l0 = self.bn3(f_s_l0)
        l_fs.append(f_s_l0)
        f_s_l1 = self.conv5(f_s[1])
        f_s_l1 = self.bn2(f_s_l1)
        l_fs.append(f_s_l1)
        f_s_l2 = self.conv1(f_s[2])
        f_s_l2 = self.bn1(f_s_l2)
        l_fs.append(f_s_l2)
        f_s_o0 = self.conv6(f_s[0])
        f_s_o0 = self.bn3(f_s_o0)
        l_fso.append(f_s_o0)
        f_s_o1 = self.conv7(f_s[1])
        f_s_o1 = self.bn2(f_s_o1)
        l_fso.append(f_s_o1)
        f_s_o2 = self.conv2(f_s[2])
        f_s_o2 = self.bn1(f_s_o2)
        l_fso.append(f_s_o2)
        #f_t_l = self.conv2(f_t)
        #f_t_l = self.bn2(f_t_l)
        #f_s_o1 = do_patches(f_s_o)
        #f_s_l1= do_patches(f_s_l)
        #f_t_l1 = do_patches(f_t)
        #f_s_o2 = do_patches2(f_s_o)
        #f_s_l2 = do_patches2(f_s_l)
        #f_t_l2 = do_patches2(f_t_l)
        

        loss_ce = self.task_loss_weight * F.cross_entropy(logits_student, target)

        loss_feat = self.tat_loss_weight * tat_loss(
            l_fs, f_t, l_fso
            )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict
