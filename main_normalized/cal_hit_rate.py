"""Cache hit rate calculation translated from MATLAB cal_hit_rate.m."""

from __future__ import annotations

import numpy as np


def cal_hit_rate(
    total_users,
    nf,
    target_community,
    cache_decision,
    community_users,
    requested_videos,
    task_assignment,
):
    cache_hit_allcount = np.zeros(nf)
    total_requests = len(requested_videos)
    target_idx = target_community - 1

    for t in range(nf):
        for i in range(len(community_users)):
            requested_video = requested_videos[i]

            if task_assignment[i, t] != 0:
                # 分配到 IU（task>0）说明命中，直接统计
                cache_hit_allcount[t] += 1
                continue

            if cache_decision[total_users + target_idx, requested_video] == 1:
                cache_hit_allcount[t] += 1

    cache_hit_rate = (np.sum(cache_hit_allcount) / total_requests) / nf
    return cache_hit_rate
