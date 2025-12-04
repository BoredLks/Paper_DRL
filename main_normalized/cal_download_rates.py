"""Download rate computation translated from MATLAB cal_download_rates.m."""

from __future__ import annotations

import numpy as np


def cal_download_rates(
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
    task_assignment,
):
    download_rates = np.zeros((len(community_users), nf))
    R_max = B * np.log2(1 + (P_sbs * K * 1 * 1) / N0)

    target_idx = target_community - 1  # MATLAB 使用 1 基社区索引
    slowfading_sd = slowfading_dB / (10 * np.log10(np.e))
    slowfading_avg = -slowfading_sd**2 / 2

    for t in range(nf):
        for i, user_idx in enumerate(community_users):
            requested_video = requested_videos[i]

            if iu_flags[user_idx] == 1 and cache_decision[user_idx, requested_video] == 1:
                download_rates[i, t] = R_max
                continue

            best_rate = 0.0

            # 情形 2：IU 邻居传输
            for j in range(iu_count_per_community):
                iu_idx = int(iu_indices[target_idx, j])
                if cache_decision[iu_idx, requested_video] == 1:
                    dist = np.linalg.norm(user_positions[user_idx, :, t] - user_positions[iu_idx, :, t])
                    if dist <= iu_coverage:
                        vartheta = np.random.lognormal(mean=slowfading_avg, sigma=slowfading_sd)
                        re_real = np.random.randn()
                        re_imag = np.random.randn()
                        xi = np.sqrt(re_real**2 + re_imag**2)
                        channel_gain = K * vartheta * xi * (dist ** (-epsilon))

                        interference = 0.0
                        for k in range(iu_count_per_community):
                            other_iu_idx = int(iu_indices[target_idx, k])
                            if other_iu_idx != iu_idx and other_iu_idx != user_idx:
                                dist_interferer = np.linalg.norm(
                                    user_positions[other_iu_idx, :, t] - user_positions[user_idx, :, t]
                                )
                                if dist_interferer <= iu_coverage:
                                    positions = np.where(iu_indices[target_idx, :] == other_iu_idx)[0]
                                    if positions.size > 0 and np.any(task_assignment[:, t] == positions[0]):
                                        vartheta_int = np.random.lognormal(
                                            mean=slowfading_avg, sigma=slowfading_sd
                                        )
                                        re_real_int = np.random.randn()
                                        re_imag_int = np.random.randn()
                                        xi_int = np.sqrt(re_real_int**2 + re_imag_int**2)
                                        interference_gain = K * vartheta_int * xi_int * (
                                            dist_interferer ** (-epsilon)
                                        )
                                        interference += P_iu * interference_gain

                        rate = B * np.log2(1 + (P_iu * channel_gain) / (N0 + interference))
                        best_rate = max(best_rate, rate)

            if best_rate <= 0.5:
                best_rate = 0.0

            # 情形 3：由 SBS 传输
            if best_rate == 0:
                dist_to_sbs = np.linalg.norm(user_positions[user_idx, :, t] - sbs_positions[target_idx])
                if dist_to_sbs <= sbs_coverage:
                    vartheta = np.random.lognormal(mean=slowfading_avg, sigma=slowfading_sd)
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
                            interference_gain = K * vartheta_int * xi_int * (
                                dist_interferer_sbs ** (-epsilon)
                            )
                            interference += P_sbs * interference_gain

                    best_rate = B * np.log2(1 + (P_sbs * channel_gain) / (N0 + interference))

            download_rates[i, t] = best_rate

    return download_rates
