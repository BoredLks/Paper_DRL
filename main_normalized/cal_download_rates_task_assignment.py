"""Download rates and task assignment translated from MATLAB cal_download_rates_task_assignment.m.

负责在每个时间片内，给目标社区用户分配 IU/SBS 传输，并计算对应的下载速率矩阵。
"""

from __future__ import annotations

import numpy as np


def cal_download_rates_task_assignment(
    h,
    iu_count_per_community,
    nf,
    sbs_coverage,
    iu_coverage,
    B,
    P_sbs,
    P_iu,
    N0,
    K,
    epsilon,
    slowfading_dB,
    target_community,
    sbs_positions,
    user_positions,
    iu_indices,
    iu_flags,
    cache_decision,
    community_users,
    requested_videos,
):
    download_rates = np.zeros((len(community_users), nf))
    task_assignment = np.zeros((len(community_users), nf))
    R_max = B * np.log2(1 + (P_sbs * K * 1 * 1) / N0)

    slowfading_sd = slowfading_dB / (10 * np.log10(np.e))
    slowfading_avg = -slowfading_sd**2 / 2
    target_idx = target_community - 1

    for t in range(nf):
        iu_busy_other = np.zeros(iu_count_per_community, dtype=bool)  # 防止同一时间一个 IU 服务多人

        for i, user_idx in enumerate(community_users):
            requested_video = requested_videos[i]

            if iu_flags[user_idx] == 1 and cache_decision[user_idx, requested_video] == 1:
                # 用户自身就是 IU 且命中缓存，直接以最大速率传输
                download_rates[i, t] = R_max
                positions = np.where(iu_indices[target_idx, :] == user_idx)[0]
                task_assignment[i, t] = positions[0] if positions.size > 0 else 0
                continue

            best_rate = 0.0
            best_assignment = 0

            for j in range(iu_count_per_community):
                iu_idx = int(iu_indices[target_idx, j])
                if (
                    cache_decision[iu_idx, requested_video] == 1
                    and not iu_busy_other[j]
                ):
                    dist = np.linalg.norm(
                        user_positions[user_idx, :, t] - user_positions[iu_idx, :, t]
                    )
                    if dist <= iu_coverage:
                        vartheta = np.random.lognormal(
                            mean=slowfading_avg, sigma=slowfading_sd
                        )
                        re_real = np.random.randn()
                        re_imag = np.random.randn()
                        xi = np.sqrt(re_real**2 + re_imag**2)
                        channel_gain = K * vartheta * xi * (dist ** (-epsilon))

                        interference = 0.0
                        for k in range(iu_count_per_community):
                            other_iu_idx = int(iu_indices[target_idx, k])
                            if other_iu_idx != iu_idx and other_iu_idx != user_idx:
                                dist_interferer = np.linalg.norm(
                                    user_positions[other_iu_idx, :, t]
                                    - user_positions[user_idx, :, t]
                                )
                                if dist_interferer <= iu_coverage:
                                    vartheta_int = np.random.lognormal(
                                        mean=slowfading_avg, sigma=slowfading_sd
                                    )
                                    re_real_int = np.random.randn()
                                    re_imag_int = np.random.randn()
                                    xi_int = np.sqrt(re_real_int**2 + re_imag_int**2)
                                    interference_gain = (
                                        K
                                        * vartheta_int
                                        * xi_int
                                        * (dist_interferer ** (-epsilon))
                                    )
                                    interference += P_iu * interference_gain

                        rate = B * np.log2(1 + (P_iu * channel_gain) / (N0 + interference))

                        if rate > best_rate:
                            best_rate = rate
                            best_assignment = j

            if best_rate <= 0.5:
                best_rate = 0.0
                best_assignment = 0

            if best_rate == 0:
                dist_to_sbs = np.linalg.norm(
                    user_positions[user_idx, :, t] - sbs_positions[target_idx]
                )
                if dist_to_sbs <= sbs_coverage:
                    vartheta = np.random.lognormal(
                        mean=slowfading_avg, sigma=slowfading_sd
                    )
                    re_real = np.random.randn()
                    re_imag = np.random.randn()
                    xi = np.sqrt(re_real**2 + re_imag**2)
                    channel_gain = K * vartheta * xi * (dist_to_sbs ** (-epsilon))

                    interference = 0.0
                    for m in range(h):
                        if m != target_idx:
                            dist_interferer_sbs = np.linalg.norm(
                                user_positions[user_idx, :, t] - sbs_positions[m]
                            )
                            vartheta_int = np.random.lognormal(
                                mean=slowfading_avg, sigma=slowfading_sd
                            )
                            re_real_int = np.random.randn()
                            re_imag_int = np.random.randn()
                            xi_int = np.sqrt(re_real_int**2 + re_imag_int**2)
                            interference_gain = (
                                K
                                * vartheta_int
                                * xi_int
                                * (dist_interferer_sbs ** (-epsilon))
                            )
                            interference += P_sbs * interference_gain

                    best_rate = B * np.log2(
                        1 + (P_sbs * channel_gain) / (N0 + interference)
                    )
                    if best_rate < 0.5:
                        best_rate = 0.5

            download_rates[i, t] = best_rate
            task_assignment[i, t] = best_assignment
            if best_assignment != 0:
                iu_busy_other[best_assignment] = True

    return download_rates, task_assignment
