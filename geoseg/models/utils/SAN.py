from math import ceil
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import torch.utils.model_zoo as model_zoo
import kmeans1d
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten


class SAN(nn.Module):

    def __init__(self, inplanes, selected_classes=None):
        super(SAN, self).__init__()
        self.margin = 0
        affine_par = True
        self.IN = nn.InstanceNorm2d(inplanes, affine=affine_par)
        self.selected_classes = selected_classes
        self.CFR_branches = nn.ModuleList()
        self.CFR_branches.append(
                nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3, bias=False))

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.mask_matrix = None

    def cosine_distance(self, obs, centers):# 计算观察值obs和中心点centers之间的余弦距离。
        obs_norm = obs / obs.norm(dim=1, keepdim=True)
        centers_norm = centers / centers.norm(dim=1, keepdim=True)
        cos = torch.matmul(obs_norm, centers_norm.transpose(1, 0))
        return 1 - cos

    def l2_distance(self, obs, centers):#  计算观察值obs和中心点centers之间的L2距离（欧几里得距离）。
        dis = ((obs.unsqueeze(dim=1) - centers.unsqueeze(dim=0)) ** 2.0).sum(dim=-1).squeeze()
        return dis

    def _kmeans_batch(self, obs: torch.Tensor, k: int, distance_function,batch_size=0, thresh=1e-5, norm_center=False):# 实现K-Means聚类的批处理版本

        # k x D
        centers = obs[torch.randperm(obs.size(0))[:k]].clone()
        history_distances = [float('inf')]
        if batch_size == 0:
            batch_size = obs.shape[0]
        while True:
            # (N x D, k x D) -> N x k
            segs = torch.split(obs, batch_size)
            seg_center_dis = []
            seg_center_ids = []
            for seg in segs:
                distances = distance_function(seg, centers)
                center_dis, center_ids = distances.min(dim=1)
                seg_center_ids.append(center_ids)
                seg_center_dis.append(center_dis)

            obs_center_dis_mean = torch.cat(seg_center_dis).mean()
            obs_center_ids = torch.cat(seg_center_ids)
            history_distances.append(obs_center_dis_mean.item())
            diff = history_distances[-2] - history_distances[-1]
            if diff < thresh:
                if diff < 0:
                    warnings.warn("Distance diff < 0, distances: " + ", ".join(map(str, history_distances)))
                break
            for i in range(k):
                obs_id_in_cluster_i = obs_center_ids == i
                if obs_id_in_cluster_i.sum() == 0:
                    continue
                obs_in_cluster = obs.index_select(0, obs_id_in_cluster_i.nonzero().squeeze())
                c = obs_in_cluster.mean(dim=0)
                if norm_center:
                    c /= c.norm()
                centers[i] = c
        return centers, history_distances[-1]

    def kmeans(self, obs: torch.Tensor, k: int, distance_function=l2_distance, iter=20, batch_size=0, thresh=1e-5, norm_center=False):# 实现K-Means聚类的完整版本。

        best_distance = float("inf")
        best_centers = None
        for i in range(iter):
            if batch_size == 0:
                batch_size == obs.shape[0]
            centers, distance = self._kmeans_batch(obs, k,
                                              norm_center=norm_center,
                                              distance_function=distance_function,
                                              batch_size=batch_size,
                                              thresh=thresh)
            if distance < best_distance:
                best_centers = centers
                best_distance = distance
        return best_centers, best_distance

    def product_quantization(self, data, sub_vector_size, k, **kwargs):# 实现产品量化技术，将数据向量分块并对每块进行聚类。
        centers = []
        for i in range(0, data.shape[1], sub_vector_size):
            sub_data = data[:, i:i + sub_vector_size]
            sub_centers, _ = self.kmeans(sub_data, k=k, **kwargs)
            centers.append(sub_centers)
        return centers

    def data_to_pq(self, data, centers):# 将数据向量映射到量化后的表示形式。
        assert (len(centers) > 0)
        assert (data.shape[1] == sum([cb.shape[1] for cb in centers]))

        m = len(centers)
        sub_size = centers[0].shape[1]
        ret = torch.zeros(data.shape[0], m,
                          dtype=torch.uint8,
                          device=data.device)
        for idx, sub_vec in enumerate(torch.split(data, sub_size, dim=1)):
            dis = self.l2_distance(sub_vec, centers[idx])
            ret[:, idx] = dis.argmin(dim=1).to(dtype=torch.uint8)
        return ret

    def train_product_quantization(self, data, sub_vector_size, k, **kwargs):# 训练产品量化模型并生成量化后的数据。
        center_list = self.product_quantization(data, sub_vector_size, k, **kwargs)
        pq_data = self.data_to_pq(data, center_list)
        return pq_data, center_list

    def _gram(self, x):# 计算输入特征x的Gram矩阵，用于捕捉特征的相关性。
        (bs, ch, h, w) = x.size()
        f = x.view(bs, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G

    def pq_distance_book(self, pq_centers):# 计算量化中心点之间的距离矩阵。
        assert (len(pq_centers) > 0)

        pq = torch.zeros(len(pq_centers),
                         len(pq_centers[0]),
                         len(pq_centers[0]),
                         device=pq_centers[0].device)
        for ci, center in enumerate(pq_centers):
            for i in range(len(center)):
                dis = self.l2_distance(center[i:i + 1, :], center)
                pq[ci, i] = dis
        return pq

    def Regional_Normalization(self, region_mask, x):# 对特定区域的特征进行归一化。
        masked = x*region_mask
        RN_feature_map = self.IN(masked)
        return RN_feature_map

    def asymmetric_table(self, query, centers):# 计算查询向量和中心向量之间的非对称距离表。
        m = len(centers)
        sub_size = centers[0].shape[1]
        ret = torch.zeros(
            query.shape[0], m, centers[0].shape[0],
            device=query.device)
        assert (query.shape[1] == sum([cb.shape[1] for cb in centers]))
        for i, offset in enumerate(range(0, query.shape[1], sub_size)):
            sub_query = query[:, offset: offset + sub_size]
            ret[:, i, :] = self.l2_distance(sub_query, centers[i])
        return ret

    def asymmetric_distance_slow(self, asymmetric_tab, pq_data):# 根据非对称距离表计算查询向量与量化数据之间的距离。
        ret = torch.zeros(asymmetric_tab.shape[0], pq_data.shape[0])
        for i in range(asymmetric_tab.shape[0]):
            for j in range(pq_data.shape[0]):
                dis = 0
                for k in range(pq_data.shape[1]):
                    sub_dis = asymmetric_tab[i, k, pq_data[j, k].item()]
                    dis += sub_dis
                ret[i, j] = dis
        return ret

    def asymmetric_distance(self, asymmetric_tab, pq_data):# 计算非对称距离（相较于 asymmetric_distance_slow 更高效）。
        pq_db = pq_data.long()
        dd = [torch.index_select(asymmetric_tab[:, i, :], 1, pq_db[:, i]) for i in range(pq_data.shape[1])]
        return sum(dd)

    def pq_distance(self, obj, centers, pq_disbook):# 计算物体与中心之间的量化距离。
        ret = torch.zeros(obj.shape[0], centers.shape[0])
        for obj_idx, o in enumerate(obj):
            for ct_idx, c in enumerate(centers):
                for i, (oi, ci) in enumerate(zip(o, c)):
                    ret[obj_idx, ct_idx] += pq_disbook[i, oi.item(), ci.item()]
        return ret

    def set_class_mask_matrix(self, normalized_map):# 根据归一化的热图设置类别掩码矩阵。

        b,c,h,w = normalized_map.size()
        var_flatten = torch.flatten(normalized_map)

        try:  # kmeans1d clustering setting for RN block
            clusters, centroids = kmeans1d.cluster(var_flatten,5, 3)
            num_category = var_flatten.size()[0] - clusters.count(0)  # 1: class-region, 2~5: background
            _, indices = torch.topk(var_flatten, k=int(num_category))
            mask_matrix = torch.flatten(torch.zeros(b, c, h, w).cuda())
            mask_matrix[indices] = 1
        except:
            mask_matrix = torch.ones(var_flatten.size()[0]).cuda()

        mask_matrix = mask_matrix.view(b, c, h, w)

        return mask_matrix

    def forward(self, x, masks):
        outs=[]
        idx = 0
        masks = masks.float()
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        inchannel = x.size(2)
        masks = F.interpolate(masks, size=(inchannel, inchannel), mode='bilinear', align_corners=False)
        # masks = F.softmax(masks, dim=1)
        for i in range(self.selected_classes):

            # masks = torch.where(masks == i, torch.ones_like(masks), torch.tensor(0))

            # mask = torch.unsqueeze(masks[:,i,:,:], 1)
            mid = x * masks

            ########公式11
            avg_out = torch.mean(mid, dim=1, keepdim=True)
            max_out,_ = torch.max(mid,dim=1, keepdim=True)
            atten = torch.cat([avg_out,max_out,masks],dim=1)
            atten = self.sigmoid(self.CFR_branches[idx](atten))
            out = mid*atten
            heatmap = torch.mean(out, dim=1, keepdim=True)

            class_region = self.set_class_mask_matrix(heatmap)
            out = self.Regional_Normalization(class_region,out)
            outs.append(out)

        out_ = sum(outs)
        out_ = self.relu(out_)

        return out_





