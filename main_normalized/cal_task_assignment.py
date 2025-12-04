"""Task assignment translated from MATLAB cal_task_assignment.m.

根据缓存命中和距离判断每个用户在各时间片由哪个 IU/SBS 提供服务，
返回的 task_assignment[i, t] 表示用户 i 在时隙 t 分配到的 IU 索引（0 表示由 SBS 服务）。
"""

from __future__ import annotations

import numpy as np


def cal_task_assignment(
    iu_count_per_community,
    nf,
    iu_coverage,
    target_community,
    user_positions,
    iu_indices,
    cache_decision,
    community_users,
    requested_videos,
):
    task_assignment = np.zeros((len(community_users), nf))
    target_idx = target_community - 1

    for t in range(nf):
        for i, user_idx in enumerate(community_users):
            requested_video = requested_videos[i]
            assigned = False
            # 依次检查目标社区内的 IU，看是否命中缓存且距离在覆盖范围内
            for j in range(iu_count_per_community):
                iu_idx = int(iu_indices[target_idx, j])
                if cache_decision[iu_idx, requested_video] == 1:
                    dist = np.linalg.norm(user_positions[user_idx, :, t] - user_positions[iu_idx, :, t])
                    if dist <= iu_coverage:
                        task_assignment[i, t] = j
                        assigned = True
                        break
            if not assigned:
                task_assignment[i, t] = 0  # 没有合适的 IU，则默认由目标社区 SBS 服务

    return task_assignment
