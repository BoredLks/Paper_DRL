"""Zipf-based user file request probabilities translated from MATLAB cal_user_file_req_prob.m.

作用：为每个用户生成对 F 个视频的请求概率，满足 Zipf 分布，γ 值越大表示热门内容集中。
"""

from __future__ import annotations

import numpy as np


def cal_user_file_req_prob(total_users, F, gamma_m):
    user_file_req_prob = np.zeros((total_users, F))
    zipf_denominator = np.sum((np.arange(1, F + 1)) ** (-gamma_m))  # Zipf 归一化因子

    for u in range(total_users):
        phi_k_ui = np.random.permutation(F)  # 随机打乱文件顺序，模拟每个用户的偏好
        rank_positions = np.empty(F, dtype=int)
        rank_positions[phi_k_ui] = np.arange(1, F + 1)  # 排名越靠前越热门
        user_file_req_prob[u, :] = (rank_positions ** (-gamma_m)) / zipf_denominator

    return user_file_req_prob
