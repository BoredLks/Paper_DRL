"""Initialize optimization variables translated from MATLAB cal_Initialize_optimization_variables.m."""

from __future__ import annotations

import numpy as np


def cal_Initialize_optimization_variables(
    chi,
    iu_count_per_community,
    nf,
    D_bf,
    iu_flags,
    cache_decision,
    community_users,
    requested_videos,
):
    user_count = len(community_users)
    r_decision = np.zeros((user_count, nf))
    r_previous = np.zeros((user_count, nf))

    r_decision[:, :] = chi[0]  # 初始码率默认设为最低档

    for i in range(user_count):
        user_idx = community_users[i]
        if iu_flags[user_idx] == 1 and cache_decision[user_idx, requested_videos[i]] == 1:
            r_decision[i, :] = chi[-1]  # 若命中缓存则直接给最高档

    buffer_state = np.zeros((user_count, nf))
    buffer_state[:, 0] = D_bf  # 初始缓冲填满

    lambda_val = 1e0 * np.random.rand(user_count, nf)  # 用户速率约束的拉格朗日乘子
    mu_sbs = 1e0 * np.random.rand(1, nf)  # SBS 容量乘子
    mu_iu = 1e0 * np.random.rand(iu_count_per_community, nf)  # IU 容量乘子

    return r_decision, r_previous, lambda_val, mu_sbs, mu_iu, buffer_state
