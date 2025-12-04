"""QoE calculation without adaptation translated from MATLAB cal_final_alltime_qoe_without_adaption.m.

此函数模拟“固定码率”播放：每个时间片直接使用预设的码率档（这里取 chi[1]），
不迭代优化，也不考虑容量约束，只用于和算法 0 的自适应策略做对比。
"""

from __future__ import annotations

import numpy as np

from .calculate_qoe import calculate_qoe


def cal_final_alltime_qoe_without_adaption(
    nf,
    chi,
    delta_t,
    alpha_qoe,
    beta_qoe,
    gamma_qoe,
    delta_qoe,
    D_bf,
    community_users,
    download_rates,
    initial_wait_times,
    r_decision,
    r_previous,
    buffer_state,
):
    final_alltime_qoe = 0.0
    r_final = np.zeros(len(community_users))

    for t in range(nf):
        if t == 0:
            for i in range(len(community_users)):
                r_previous[i, t] = chi[np.random.randint(len(chi))]
        else:
            r_previous[:, t] = r_decision[:, t - 1]
            initial_wait_times[:] = 0
            for i in range(len(community_users)):
                buffer_state[i, t] = max(
                    0,
                    min(
                        D_bf,
                        buffer_state[i, t - 1]
                        + (download_rates[i, t - 1] - r_decision[i, t - 1]) * delta_t,
                    ),
                )

        r_final.fill(chi[1])  # 直接把码率固定在第二档
        r_decision[:, t] = r_final

        final_total_qoe = 0.0
        for i in range(len(community_users)):
            bu_current = buffer_state[i, t]
            qoe_i = calculate_qoe(
                bu_current,
                r_final[i],
                r_previous[i, t],
                initial_wait_times[i],
                download_rates[i, t],
                alpha_qoe,
                beta_qoe,
                gamma_qoe,
                delta_qoe,
                nf,
                delta_t,
                t + 1,
                chi,
                initial_wait_times,
            )
            final_total_qoe += qoe_i

        final_alltime_qoe += final_total_qoe

    return final_alltime_qoe
