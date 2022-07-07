import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller
import numpy as np

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


def split_featmap(f):
    y = torch.split(f, 1, dim=3)
    lst = []
    for i in range(8):
        z = torch.split(y[i], 1, dim=2)
        for j in z:
            lst.append(j)
    return lst


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


def gt_and_non_loss(aligned_feature, lst_gt, lst_non_gt):
    bsz = aligned_feature.size(0)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #nup_gt = np.array(lst_gt).astype(float)
    #nup_non_gt = np.array(lst_non_gt).astype(float)
    #lst_gt = torch.from_numpy(nup_gt).to(device).reshape(bsz, 64, 1)
    #lst_non_gt = torch.from_numpy(nup_non_gt).to(device).reshape(bsz, 64, 1)
    gt = aligned_feature * lst_gt.reshape(bsz, 64, 1)
    non_gt = aligned_feature * lst_non_gt.reshape(bsz, 64, 1)
    return gt, non_gt
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bsz = aligned_feature.size(0)
    c = aligned_feature.size(2)
    res_gt = None
    res_non_gt = None
    t_l = torch.split(aligned_feature, 1, dim=1)
    for i in range(len(lst)):
        tem_gt = None
        tem_non_gt = None
        for j in range(len(lst)):
            if lst[i][j] == 1:
                tem_gt = t_l[j][i].reshape(1, 1, -1) if tem_gt==None else torch.cat((tem_gt, t_l[j][i].reshape(1, 1, -1)), dim=1)
                tem_non_gt = torch.zeros(1, 1, c).to(device) if tem_non_gt==None else torch.cat((tem_non_gt, torch.zeros(1, 1, c).to(device)), dim=1)
            else:
                tem_non_gt = t_l[j][i].reshape(1, 1, -1) if tem_non_gt==None else torch.cat((tem_non_gt, t_l[j][i].reshape(1, 1, -1)), dim=1)
                tem_gt = torch.zeros(1, 1, c).to(device) if tem_gt==None else torch.cat((tem_gt, torch.zeros(1, 1, c).to(device)), dim=1)
        
        res_gt = tem_gt if res_gt==None else torch.cat((res_gt, tem_gt), dim=0)
        res_non_gt = tem_non_gt if res_non_gt==None else torch.cat((res_non_gt, tem_non_gt), dim=0)
    return res_gt, res_non_gt
    '''


def avg_pool(gt):
    res = gt.mean(dim=1)
    return res

class TEST2(Distiller):
    """fu'xian idea4"""

    def __init__(self, student, teacher, cfg):
        super(TEST2, self).__init__(student, teacher)
        self.temperature = cfg.TEST2.TEMPERATURE
        self.ce_loss_weight = cfg.TEST2.LOSS.TASK_WEIGHT
        self.gt_loss_weight = cfg.TEST2.LOSS.GT_WEIGHT
        self.non_gt_loss_weight = cfg.TEST2.LOSS.NON_GT_WEIGHT
        self.sftmx = nn.Softmax(dim=1)
        self.criterion = nn.L1Loss()

    def forward_train(self, image, target, **kwargs):
        logits_student, feats_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, feats_teacher, fc_teacher = self.teacher(image)

        f_t = feats_teacher["feats"][3]
        f_s = feats_student["feats"][3]
        lst = split_featmap(f_t)                            #len 64 item[64,256,1,1]
        bsz = f_t.size(0)
        lst_comp_gt = None    
        lst_comp_non_gt = None                  
        for i in range(len(lst)):                           #loop = 64
            spc = lst[i].reshape(lst[i].size(0), -1)        #64*256
            out = fc_teacher(spc)                           #64*100
            res = self.sftmx(out)                           #64*100
            idx = torch.argmax(res, dim=1)                  #64
            lst_comp_gt = idx.eq_(target).reshape(1, bsz) if lst_comp_gt==None else torch.cat([lst_comp_gt, idx.eq_(target).reshape(1, bsz)], dim=0)
            lst_comp_non_gt = idx.ne_(target).reshape(1, bsz) if lst_comp_non_gt==None else torch.cat([lst_comp_non_gt, idx.ne_(target).reshape(1, bsz)], dim=0)
            '''
            for j in range(bsz):
                if idx[j] == target[j]:           
                    lst_comp_gt[j].append(1)                   #[[which spc works(one-hot)]*bsz]
                    lst_comp_non_gt[j].append(0)
                else:
                    lst_comp_gt[j].append(0)
                    lst_comp_non_gt[j].append(1)
            '''
        nf_s, nf_t = tat_alignment(f_s, f_t, f_s)
        gt_s, non_gt_s = gt_and_non_loss(nf_s, lst_comp_gt, lst_comp_non_gt)
        gt_t, non_gt_t = gt_and_non_loss(nf_t, lst_comp_gt, lst_comp_non_gt)
        res_s = avg_pool(gt_s)
        res_non_s = avg_pool(non_gt_s)
        res_t = avg_pool(gt_t)
        res_non_t = avg_pool(non_gt_t)  


        # avg+L1
        #gt_loss = self.criterion(res_s, res_t)
        #non_gt_loss = self.criterion(res_non_s, res_non_t)  


        # avg+KL    
        #log_gt_student = F.log_softmax(res_s, dim=1)
        #gt_teacher = F.softmax(res_t, dim=1)
        #gt_loss = F.kl_div(log_gt_student, gt_teacher, reduction="none").sum(1).mean()
        #log_non_gt_student = F.log_softmax(res_non_s, dim=1)
        #non_gt_teacher = F.softmax(res_non_t, dim=1)
        #non_gt_loss = F.kl_div(log_non_gt_student, non_gt_teacher, reduction="none").sum(1).mean()


        # element+KL
        log_gt_student = F.log_softmax(gt_s, dim=1)
        gt_teacher = F.softmax(gt_t, dim=1)
        gt_loss = F.kl_div(log_gt_student, gt_teacher, reduction="none").sum(1).mean()
        log_non_gt_student = F.log_softmax(non_gt_s, dim=1)
        non_gt_teacher = F.softmax(non_gt_t, dim=1)
        non_gt_loss = F.kl_div(log_non_gt_student, non_gt_teacher, reduction="none").sum(1).mean()       


        # element+L1
        #gt_loss = self.criterion(gt_s, gt_t)
        #non_gt_loss = self.criterion(non_gt_s, non_gt_t)

        
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.gt_loss_weight * gt_loss + self.non_gt_loss_weight * non_gt_loss

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
