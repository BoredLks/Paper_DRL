"""Cache preference calculation translated from MATLAB cal_cache_preference.m."""

from __future__ import annotations

import numpy as np


def cal_cache_preference(h, total_users, F, user_file_req_prob, joint_edge_weights):
    """根据邻居的请求概率与边权，估算每个节点对文件的缓存偏好"""
    cache_preference = np.zeros((total_users + h, F))

    for i in range(total_users + h):
        out_neighbors = np.where(joint_edge_weights[i, :] > 0)[0]
        out_neighbors = out_neighbors[out_neighbors != i]  # 排除自身

        if out_neighbors.size > 0:
            for k in range(F):
                preference_sum = 0.0
                for j in out_neighbors:
                    if j < total_users:
                        # 只有真实用户（非 SBS）才有请求概率
                        preference_sum += user_file_req_prob[j, k] * joint_edge_weights[
                            i, j
                        ]
                cache_preference[i, k] = preference_sum / len(out_neighbors)

    return cache_preference
