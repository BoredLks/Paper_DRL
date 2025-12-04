"""Algorithm 1 cache decision (Local Most Popular Content) translated from MATLAB.

思路：统计每个节点覆盖到的邻居用户，请求概率之和越大、说明越热门，按照热门度排序填充缓存。
"""

from __future__ import annotations

import numpy as np


def cal_cache_decision_1(
    h,
    total_users,
    F,
    iu_count_per_community,
    sbs_coverage,
    iu_coverage,
    nf,
    v_chi_star,
    D_eg,
    D_iu,
    sbs_positions,
    user_positions,
    iu_indices,
    iu_flags,
    user_file_req_prob,
):
    cache_decision = np.zeros((total_users + h, F))
    user_avg_positions = np.array(user_positions)[:, :, 0]  # 使用第一个时间片坐标
    cache_popular = np.zeros((total_users + h, F))  # 记录每个节点视角下的热门度
    sbs_cache_space = np.zeros(h)
    iu_cache_space = np.zeros(total_users)

    # 计算每个节点的热门度：SBS 看覆盖范围内用户，IU 看通信范围内的邻居
    for i in range(total_users + h):
        if i < total_users:
            if iu_flags[i] == 1:
                out_neighbors = []
                for j in range(total_users):
                    dist = np.linalg.norm(user_avg_positions[i] - user_avg_positions[j])
                    if dist <= iu_coverage:
                        out_neighbors.append(j)
            else:
                out_neighbors = []
        else:
            m = i - total_users
            out_neighbors = []
            for j in range(total_users):
                dist = np.linalg.norm(user_avg_positions[j] - sbs_positions[m])
                if dist <= sbs_coverage:
                    out_neighbors.append(j)

        if out_neighbors:
            for k in range(F):
                popular_sum = 0.0
                for j in out_neighbors:
                    if j < total_users:
                        popular_sum += user_file_req_prob[j, k]
                cache_popular[i, k] = popular_sum / len(out_neighbors)

    # SBS 按热门度缓存
    for m in range(h):
        sbs_idx = total_users + m
        sbs_popular = np.array(cache_popular[sbs_idx], dtype=float)
        file_indices = np.argsort(sbs_popular)[::-1]

        for f in file_indices:
            pop = sbs_popular[f]
            if pop > 0:
                if sbs_cache_space[m] + nf * v_chi_star <= D_eg:
                    cache_decision[sbs_idx, f] = 1
                    sbs_cache_space[m] += nf * v_chi_star
                else:
                    break
            else:
                break

    # IU 按热门度缓存
    for m in range(h):
        for i in range(iu_count_per_community):
            iu_idx = int(iu_indices[m, i])
            iu_popular = np.array(cache_popular[iu_idx], dtype=float)
            file_indices = np.argsort(iu_popular)[::-1]

            for f in file_indices:
                pop = iu_popular[f]
                if pop > 0:
                    if iu_cache_space[iu_idx] + nf * v_chi_star <= D_iu:
                        cache_decision[iu_idx, f] = 1
                        iu_cache_space[iu_idx] += nf * v_chi_star
                    else:
                        break
                else:
                    break

    return cache_decision
