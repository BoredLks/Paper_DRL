"""Algorithm 5: Non-collaborative caching (NCC) translated from MATLAB.

解释：SBS 仍按 joint 权重决定缓存文件，但 IU 完全不参与缓存，
用于模拟“无协作缓存”场景。
"""

from __future__ import annotations

import numpy as np


def cal_cache_decision_5(
    h,
    total_users,
    F,
    nf,
    v_chi_star,
    D_eg,
    cache_preference,
):
    cache_decision = np.zeros((total_users + h, F))

    sbs_cache_space = np.zeros(h)
    cached_files_by_sbs = np.zeros(F, dtype=bool)  # 记录哪些文件已被某个 SBS 缓存

    for m in range(h):
        sbs_idx = total_users + m

        sbs_preferences = np.array(cache_preference[sbs_idx], dtype=float)
        sbs_preferences[cached_files_by_sbs] = -np.inf  # 避免多个 SBS 缓存同一文件
        file_indices = np.argsort(sbs_preferences)[::-1]  # 按偏好降序排序

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

    # NCC 情形下 IU 不缓存任何内容
    return cache_decision
