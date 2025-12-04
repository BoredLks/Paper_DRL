"""Algorithm 0 cache decision converted from MATLAB cal_cache_decision_0.m.

按照 joint 权重得到的缓存偏好，先给每个 SBS 分配不同的热门文件，再让 IU 根据各自偏好填满缓存。
"""

from __future__ import annotations

import numpy as np


def cal_cache_decision_0(
    h,
    total_users,
    F,
    iu_count_per_community,
    nf,
    v_chi_star,
    D_eg,
    D_iu,
    iu_indices,
    cache_preference,
):
    cache_decision = np.zeros((total_users + h, F))

    # SBS 缓存容量跟踪
    sbs_cache_space = np.zeros(h)
    cached_files_by_sbs = np.zeros(F, dtype=bool)

    # IU 缓存容量跟踪
    iu_cache_space = np.zeros(total_users)

    # SBS 缓存阶段（避免多个 SBS 缓存同一文件）
    for m in range(h):
        sbs_idx = total_users + m
        sbs_preferences = np.array(cache_preference[sbs_idx], dtype=float)
        sbs_preferences[cached_files_by_sbs] = -np.inf
        file_indices = np.argsort(sbs_preferences)[::-1]  # 按偏好降序排列

        for f in file_indices:
            pref = sbs_preferences[f]
            if pref > 0:
                if sbs_cache_space[m] + nf * v_chi_star <= D_eg:
                    cache_decision[sbs_idx, f] = 1
                    sbs_cache_space[m] += nf * v_chi_star
                    cached_files_by_sbs[f] = True
                else:
                    break
            else:
                break

    # IU 缓存阶段
    for m in range(h):
        for i in range(iu_count_per_community):
            iu_idx = int(iu_indices[m, i])
            iu_preferences = np.array(cache_preference[iu_idx], dtype=float)
            file_indices = np.argsort(iu_preferences)[::-1]

            for f in file_indices:
                pref = iu_preferences[f]
                if pref > 0:
                    if iu_cache_space[iu_idx] + nf * v_chi_star <= D_iu:
                        cache_decision[iu_idx, f] = 1
                        iu_cache_space[iu_idx] += nf * v_chi_star
                    else:
                        break
                else:
                    break

    return cache_decision
