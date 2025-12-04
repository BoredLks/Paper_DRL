"""Physical link edge weights translated from MATLAB cal_pl_edge_weights.m."""

from __future__ import annotations

import numpy as np


def cal_pl_edge_weights(
    h,
    total_users,
    iu_count_per_community,
    sbs_coverage,
    iu_coverage,
    T_small,
    B,
    P_sbs,
    P_iu,
    N0,
    K,
    epsilon,
    slowfading_dB,
    sbs_positions,
    user_positions,
    iu_indices,
    iu_flags,
):
    pl_edge_weights = np.zeros((total_users + h, total_users + h))
    # 计算理想情况下的最大传输速率，用于归一化
    R_max_sbs = B * np.log2(1 + (P_sbs * K * 1 * 1) / N0)
    R_max_iu = B * np.log2(1 + (P_iu * K * 1 * 1) / N0)

    users_per_comm = int(total_users / h)
    slowfading_sd = slowfading_dB / (10 * np.log10(np.e))
    slowfading_avg = -slowfading_sd**2 / 2

    for m in range(h):
        sbs_idx = total_users + m

        for i in range(m * users_per_comm, (m + 1) * users_per_comm):
            pl_edge_weights[i, i] = 1 if iu_flags[i] == 1 else 0  # IU 节点和自身有边

            for j in range(i + 1, (m + 1) * users_per_comm):
                if iu_flags[i] == 1 or iu_flags[j] == 1:
                    total_rate = 0.0
                    valid_slots = 0

                    for t in range(T_small):
                        dist_t = np.linalg.norm(user_positions[i, :, t] - user_positions[j, :, t])
                        # 只在 IU 覆盖范围内才考虑直连
                        if dist_t <= iu_coverage:
                            vartheta = np.random.lognormal(mean=slowfading_avg, sigma=slowfading_sd)
                            re = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                            xi = np.abs(re) ** 2
                            channel_gain = K * vartheta * xi * (dist_t ** (-epsilon))

                            interference = 0.0
                            for k in range(iu_count_per_community):
                                other_iu_idx = int(iu_indices[m, k])
                                if other_iu_idx != i and other_iu_idx != j:
                                    dist_interferer = np.linalg.norm(
                                        user_positions[other_iu_idx, :, t] - user_positions[j, :, t]
                                    )
                                    if dist_interferer <= iu_coverage:
                                        vartheta_int = np.random.lognormal(
                                            mean=slowfading_avg, sigma=slowfading_sd
                                        )
                                        re_int = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                                        xi_int = np.abs(re_int) ** 2
                                        interference_gain = (
                                            K * vartheta_int * xi_int * (dist_interferer ** (-epsilon))
                                        )
                                        interference += P_iu * interference_gain

                            R_instantaneous = B * np.log2(1 + (P_iu * channel_gain) / (N0 + interference))
                            total_rate += R_instantaneous
                            valid_slots += 1

                    if valid_slots > 0:
                        avg_rate = total_rate / valid_slots
                        val = min(1, avg_rate / R_max_iu)
                        pl_edge_weights[i, j] = val
                        pl_edge_weights[j, i] = val
                    else:
                        pl_edge_weights[i, j] = 0
                        pl_edge_weights[j, i] = 0
                else:
                    pl_edge_weights[i, j] = 0
                    pl_edge_weights[j, i] = 0

            total_rate = 0.0
            valid_slots = 0

            for t in range(T_small):
                dist_t = np.linalg.norm(user_positions[i, :, t] - sbs_positions[m])
                if dist_t <= sbs_coverage:
                    vartheta = np.random.lognormal(mean=slowfading_avg, sigma=slowfading_sd)
                    re = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                    xi = np.abs(re) ** 2
                    channel_gain = K * vartheta * xi * (dist_t ** (-epsilon))

                    interference = 0.0
                    for n in range(h):
                        if n != m:
                            dist_interferer_sbs = np.linalg.norm(
                                user_positions[i, :, t] - sbs_positions[n]
                            )
                            vartheta_int = np.random.lognormal(mean=slowfading_avg, sigma=slowfading_sd)
                            re_int = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                            xi_int = np.abs(re_int) ** 2
                            interference_gain = (
                                K * vartheta_int * xi_int * (dist_interferer_sbs ** (-epsilon))
                            )
                            interference += P_sbs * interference_gain

                    R_instantaneous = B * np.log2(
                        1 + (P_sbs * channel_gain) / (N0 + interference)
                    )
                    total_rate += R_instantaneous
                    valid_slots += 1

            if valid_slots > 0:
                avg_rate = total_rate / valid_slots
                val = min(1, avg_rate / R_max_sbs)
                pl_edge_weights[i, sbs_idx] = val
                pl_edge_weights[sbs_idx, i] = val
            else:
                pl_edge_weights[i, sbs_idx] = 0
                pl_edge_weights[sbs_idx, i] = 0

        pl_edge_weights[sbs_idx, sbs_idx] = 1
        for n in range(m + 1, h):
            other_sbs_idx = total_users + n
            pl_edge_weights[sbs_idx, other_sbs_idx] = 0
            pl_edge_weights[other_sbs_idx, sbs_idx] = 0

    return pl_edge_weights
