"""Algorithm 4: Global Most Popular Content translated from MATLAB.

GMPC：SBS 与 IU 都按照社区内的平均热门度排序缓存内容，强调全局统一策略。
"""

from __future__ import annotations

import numpy as np


def cal_cache_decision_4(
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
        community_start = m * user_per_community
        community_end = community_start + user_per_community
        global_popularity = np.mean(
            user_file_req_prob[community_start:community_end, :], axis=0
        )
        popularity_ranking = np.argsort(global_popularity)[::-1]

        top_sbs = min(files_per_sbs, F)
        for rank in range(top_sbs):
            f = int(popularity_ranking[rank])
            cache_decision[sbs_idx, f] = 1

    for m in range(h):
        community_start = m * user_per_community
        community_end = community_start + user_per_community
        global_popularity = np.mean(
            user_file_req_prob[community_start:community_end, :], axis=0
        )
        popularity_ranking = np.argsort(global_popularity)[::-1]

        for i in range(iu_count_per_community):
            iu_idx = int(iu_indices[m, i])
            top_iu = min(files_per_iu, F)
            for rank in range(top_iu):
                f = int(popularity_ranking[rank])
                cache_decision[iu_idx, f] = 1

    return cache_decision
