"""IU selection translated from MATLAB cal_Select_iu.m (random selection version)."""

from __future__ import annotations

import numpy as np


def cal_Select_iu(
    h,
    user_per_community,
    total_users,
    iu_count_per_community,
    iu_coverage,
    user_positions,
):
    # iu_indices 记录每个社区挑选出的 IU 编号；iu_flags 表示用户是否为 IU
    iu_indices = np.zeros((h, iu_count_per_community), dtype=int)
    iu_flags = np.zeros(total_users, dtype=int)

    for m in range(h):
        community_user_indices = np.arange(m * user_per_community, (m + 1) * user_per_community)
        # 每个社区随机挑选固定数量的 IU，代表具有缓存/算力的终端
        selected_iu = np.random.choice(community_user_indices, iu_count_per_community, replace=False)
        iu_indices[m, :] = selected_iu
        iu_flags[selected_iu] = 1

    return iu_indices, iu_flags
