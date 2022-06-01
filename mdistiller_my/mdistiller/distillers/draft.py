from imghdr import tests
from sklearn.datasets import fetch_lfw_people #导入人脸识别数据集库
from sklearn.decomposition import PCA         #导入降维算法PCA模块
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""faces=fetch_lfw_people(min_faces_per_person=60)  #实例化
faces.data.shape                                 #查看数据结构
faces.images.shape
x=faces.data

pca=PCA(150).fit(x)   #实例化 拟合
x_dr=pca.transform(x) #训练转换
x_dr.shape
"""

"""f_s2 = torch.rand(2, 64, 32, 32)
conv1_2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
conv1_1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2, bias=False)  
f_s2 = conv1_1(f_s2)      
f_s2 = conv1_2(f_s2)

avgpool = nn.AvgPool2d(8)
f_s2 = avgpool(f_s2)
avg_fs2 = f_s2.reshape(f_s2.size(0), -1)
print(avg_fs2.shape)"""
def single_stage_tat_loss(f_s, f_t, f_s_o):
    #同论文假设s和t hwc 都相等
    bsz = f_s.size(0)
    c = f_s.size(1)
    h = f_s.size(2)
    w = f_s.size(3)
    f_s = f_s.view(bsz, c, -1).transpose(1, 2)
    f_t = f_t.view(bsz, c, -1).transpose(1, 2)
    f_s_o = f_s_o.view(bsz, c, -1).transpose(1, 2)
    f_t_t = f_t.transpose(1, 2)
    sig = nn.Sigmoid()
    print(torch.matmul(f_s, f_t_t))
    tem = sig(torch.matmul(f_s, f_t_t))
    print(tem, f_s_o.shape)
    f_ns = torch.matmul(tem, f_s_o)
    print(f_ns.shape)
    criterion = nn.MSELoss(reduction="mean")
    print((f_ns - f_t).pow(2).mean(), criterion(f_ns, f_t))
    return (f_ns - f_t).pow(2).mean()

def tat_loss(g_s, g_t, g_s_o):
    return sum([single_stage_tat_loss(f_s, f_t, f_s_o) for f_s, f_t, f_s_o in zip(g_s, g_t, g_s_o)])

def do_patches(f):
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

test_s = torch.rand([2, 256, 8, 8])
test_t = torch.rand([2, 256, 8, 8])
conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
bn1 = nn.BatchNorm2d(256)
bn2 = nn.BatchNorm2d(256)
f_s_l = conv1(test_s)
f_s_l = bn1(f_s_l)
f_t_l = conv2(test_t)
f_t_l = bn2(f_t_l)



loss_feat = single_stage_tat_loss(
            f_s_l, f_t_l, test_s
        )
print(loss_feat)
"""x = torch.tensor([[[[1., 2.],
                    [3., 4.]], 
                    
                    [[5., 6.], 
                    [7., 8.]]],
                    
                    
                    [[[9., 10.],
                    [11., 12.]],
                    
                    [[13., 14.],
                    [15., 16.]]]])
print(x.shape)
y = torch.rand(2, 256, 8, 8)
z = torch.rand(2, 256, 8, 8)
print(single_stage_tat_loss(y, z))"""