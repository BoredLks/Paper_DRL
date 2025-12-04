"""Algorithm 2: Random caching (RA) translated from MATLAB cal_cache_decision_2.m."""

from __future__ import annotations

import numpy as np


def cal_cache_decision_2(
    h,
    total_users,
    F,
    iu_count_per_community,
    nf,
    v_chi_star,
    D_eg,
    D_iu,
    iu_indices,
):
    cache_decision = np.zeros((total_users + h, F))

    eg_count = int(D_eg / (nf * v_chi_star))
    iu_count = int(D_iu / (nf * v_chi_star))

    # SBS 随机挑选文件缓存
    for m in range(h):
        sbs_idx = total_users + m
        idx_ra_eg = np.random.choice(F, eg_count, replace=False)
        cache_decision[sbs_idx, idx_ra_eg] = 1

    # IU 随机挑选文件缓存
    for m in range(h):
        for i in range(iu_count_per_community):
            iu_idx = int(iu_indices[m, i])
            idx_ra_iu = np.random.choice(F, iu_count, replace=False)
            cache_decision[iu_idx, idx_ra_iu] = 1

    return cache_decision
