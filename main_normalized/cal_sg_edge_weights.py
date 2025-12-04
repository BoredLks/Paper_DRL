"""Social graph edge weights translated from MATLAB cal_sg_edge_weights.m."""

from __future__ import annotations

import numpy as np


def cal_sg_edge_weights(
    h,
    total_users,
    sbs_coverage,
    alpha,
    beta,
    gamma,
    sbs_positions,
    user_positions,
    user_community,
    sim_matrix,
):
    sg_edge_weights = np.zeros((total_users + h, total_users + h))
    user_avg_positions = np.mean(user_positions, axis=2)  # 取用户轨迹的平均位置，用于和 SBS 判断联系

    user_friends = [None] * total_users
    for u in range(total_users):
        community_u = user_community[u]
        community_users = np.where(user_community == community_u)[0]
        num_friends = min(round(len(community_users) * 0.5), len(community_users))
        # 随机挑选社区内朋友集合，用来计算亲密度
        user_friends[u] = np.random.permutation(community_users)[:num_friends]

    for i in range(total_users):
        sg_edge_weights[i, i] = 1
        community_i = user_community[i]

        for j in range(i + 1, total_users):
            community_j = user_community[j]
            if community_i == community_j:
                # 同社区：按亲密度、偏好相似度、重要性加权构成边
                J_vi = user_friends[i]
                J_vj = user_friends[j]
                intersection_size = len(np.intersect1d(J_vi, J_vj))
                union_size = len(np.union1d(J_vi, J_vj))
                IN_vi_vj = intersection_size / union_size if union_size > 0 else 0
                PS_vi_vj = sim_matrix[i, j]
                IM_vi_vj = np.random.rand()
                edge_weight = alpha * IN_vi_vj + beta * PS_vi_vj + gamma * IM_vi_vj
                sg_edge_weights[i, j] = edge_weight
                sg_edge_weights[j, i] = edge_weight
            else:
                # 不同社区则无社交连边
                sg_edge_weights[i, j] = 0
                sg_edge_weights[j, i] = 0

        for m in range(h):
            sbs_idx = total_users + m
            avg_dist = np.linalg.norm(user_avg_positions[i] - sbs_positions[m])
            if avg_dist <= sbs_coverage:
                sg_edge_weights[sbs_idx, i] = 1  # 用户与覆盖内的 SBS 建立社交（合作）边
                sg_edge_weights[i, sbs_idx] = 1
            else:
                sg_edge_weights[sbs_idx, i] = 0
                sg_edge_weights[i, sbs_idx] = 0

    return sg_edge_weights
