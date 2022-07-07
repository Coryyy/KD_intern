import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller
import matplotlib.pyplot as plt
import numpy as np

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


def alignment(l_s, l_t):
        bsz = l_s.shape[0]
        v_t, _ = torch.max(l_t, 1)
        v_s, _ = torch.max(l_s, 1)
        times = torch.div(v_t, v_s).reshape(bsz, 1)
        #s = tuple([i for i in range(bsz)])
        #plt.plot(s, l_s.cpu().detach().numpy(), l_t.cpu().detach().numpy())
        #plt.savefig('/data/mdistiller/visualization/unprocessed.jpg')   
        #print(l_s)
        l_s = l_s * times
        return l_s, l_t

def tat_alignment(f_s, f_t, f_s_o):
    #同论文假设s和t hwc 都相等
    bsz = f_s.size(0)
    c = f_s.size(1)
    f_s = f_s.reshape(bsz, c, -1).transpose(1, 2)
    f_t = f_t.reshape(bsz, c, -1).transpose(1, 2)
    f_s_o = f_s_o.reshape(bsz, c, -1).transpose(1, 2)
    f_t_t = f_t.transpose(1, 2)
    tem = F.softmax(torch.bmm(f_s, f_t_t), dim=2)
    f_ns = torch.bmm(tem, f_s_o)
    return f_ns, f_t


class TEST0(Distiller):
    """fu'xian idea2"""

    def __init__(self, student, teacher, cfg):
        super(TEST0, self).__init__(student, teacher)
        self.temperature = cfg.TEST.TEMPERATURE
        self.ce_loss_weight = cfg.TEST.LOSS.TASK_WEIGHT
        self.kd_loss_weight = cfg.TEST.LOSS.KL_WEIGHT
        self.criterion = nn.L1Loss()

    def forward_train(self, image, target, **kwargs):
        logits_student, _, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _, _ = self.teacher(image)

        l_s, l_t = tat_alignment(logits_student, logits_teacher, logits_student)

        # losses
        loss_ce = 1 * F.cross_entropy(logits_student, target)
        loss_kd = 1 * self.criterion(l_s, l_t)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
