"""Algorithm 3: Hybrid caching (HC) translated from MATLAB cal_cache_decision_3.m.

思想：SBS 先缓存社区内最热门的文件，IU 再缓存剩余热门内容，实现分散化。
"""

from __future__ import annotations

import numpy as np


def cal_cache_decision_3(
    h,
    user_per_community,
    total_users,
    F,
    iu_count_per_community,
    nf,
    v_chi_star,
    D_eg,
    D_iu,
    iu_indices,
    user_file_req_prob,
):
    cache_decision = np.zeros((total_users + h, F))

    files_per_sbs = int(D_eg / (nf * v_chi_star))
    files_per_iu = int(D_iu / (nf * v_chi_star))

    for m in range(h):
        sbs_idx = total_users + m
        community_slice_start = m * user_per_community
        community_slice_end = community_slice_start + user_per_community
        global_popularity = np.mean(
            user_file_req_prob[community_slice_start:community_slice_end, :], axis=0
        )
        popularity_ranking = np.argsort(global_popularity)[::-1]

        top_sbs = min(files_per_sbs, F)
        sbs_cached_files = np.zeros(top_sbs, dtype=int)
        for rank in range(top_sbs):
            f = int(popularity_ranking[rank])
            cache_decision[sbs_idx, f] = 1
            sbs_cached_files[rank] = f

        # IU 只能缓存 SBS 未覆盖的剩余文件
        remaining_files = popularity_ranking[
            ~np.isin(popularity_ranking, sbs_cached_files)
        ]

        for i in range(iu_count_per_community):
            iu_idx = int(iu_indices[m, i])
            top_iu = min(files_per_iu, len(remaining_files))
            for rank in range(top_iu):
                f = int(remaining_files[rank])
                cache_decision[iu_idx, f] = 1

    return cache_decision
