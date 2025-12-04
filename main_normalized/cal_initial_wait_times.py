"""Initial wait time calculation translated from MATLAB cal_initial_wait_times.m."""

from __future__ import annotations

import numpy as np


def cal_initial_wait_times(
    h,
    total_users,
    iu_count_per_community,
    v_chi_star,
    iu_coverage,
    p_sbs,
    p_iu,
    D_bf,
    t_cloud,
    t_propagation,
    target_community,
    user_positions,
    iu_indices,
    iu_flags,
    cache_decision,
    community_users,
    requested_videos,
    download_rates,
):
    initial_wait_times = np.zeros(len(community_users))
    target_idx = target_community - 1

    for i, user_idx in enumerate(community_users):
        requested_video = requested_videos[i]

        if iu_flags[user_idx] == 1 and cache_decision[user_idx, requested_video] == 1:
            initial_wait_times[i] = 0  # 自己缓存命中，直接播放
        elif cache_decision[total_users + target_idx, requested_video] == 1:
            transcode_time = v_chi_star / p_sbs
            buffer_fill_time = D_bf / download_rates[i, 0]
            initial_wait_times[i] = transcode_time + buffer_fill_time
        else:
            found_in_iu = False
            for j in range(iu_count_per_community):
                iu_idx = int(iu_indices[target_idx, j])
                dist = np.linalg.norm(user_positions[user_idx, :, 0] - user_positions[iu_idx, :, 0])
                if cache_decision[iu_idx, requested_video] == 1 and dist <= iu_coverage:
                    transcode_time = v_chi_star / p_iu
                    buffer_fill_time = D_bf / download_rates[i, 0]
                    initial_wait_times[i] = transcode_time + buffer_fill_time
                    found_in_iu = True
                    break

            if not found_in_iu:
                found_in_other_sbs = False
                for m in range(h):
                    if m != target_idx:
                        sbs_idx = total_users + m
                        if cache_decision[sbs_idx, requested_video] == 1:
                            transcode_time = v_chi_star / p_sbs
                            buffer_fill_time = D_bf / download_rates[i, 0]
                            initial_wait_times[i] = t_propagation + transcode_time + buffer_fill_time
                            found_in_other_sbs = True
                            break

                if not found_in_other_sbs:
                    transcode_time = v_chi_star / p_sbs
                    buffer_fill_time = D_bf / download_rates[i, 0]
                    initial_wait_times[i] = t_cloud + transcode_time + buffer_fill_time

    return initial_wait_times
