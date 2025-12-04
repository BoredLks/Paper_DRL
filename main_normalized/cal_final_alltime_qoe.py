"""Iterative QoE optimization translated from MATLAB cal_final_alltime_qoe.m.

该函数实现论文中的“多约束 QoE 优化”迭代：对每个时间片循环更新码率决策、拉格朗日乘子，
直至收敛或达到最大迭代次数，并输出所有用户的累计 QoE。
"""

from __future__ import annotations

import numpy as np

from .calculate_qoe import calculate_qoe


def cal_final_alltime_qoe(
    iu_count_per_community,
    nf,
    chi,
    delta_t,
    alpha_qoe,
    beta_qoe,
    gamma_qoe,
    delta_qoe,
    p_sbs,
    p_iu,
    D_bf,
    eta,
    epsilon_conv,
    max_iterations,
    iu_flags,
    user_file_req_prob,
    cache_decision,
    community_users,
    requested_videos,
    download_rates,
    task_assignment,
    initial_wait_times,
    r_decision,
    r_previous,
    lambda_val,
    mu_sbs,
    mu_iu,
    buffer_state,
):
    final_alltime_qoe = 0.0
    chi = np.array(chi, dtype=float)

    for t in range(nf):
        convergence_lagrange = []
        convergence_qoe = []
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

        for _ in range(max_iterations):
            # 先根据当前码率估算 SBS/IU 的负载，用于拉格朗日乘子
            sbs_load = 0.0
            iu_load = np.zeros(iu_count_per_community)
            lagrange_val = 0.0
            qoe_all_user = 0.0

            for n in range(len(community_users)):
                if task_assignment[n, t] == 0 and r_decision[n, t] < chi[-1]:
                    sbs_load += r_decision[n, t]
            for n in range(iu_count_per_community):
                for m in range(len(community_users)):
                    if task_assignment[m, t] == n + 1 and r_decision[m, t] < chi[-1]:
                        iu_load[n] += r_decision[m, t]

            for i in range(len(community_users)):
                # 计算每个用户的 QoE、约束违反程度
                if task_assignment[i, t] == 0:
                    mu_val = mu_sbs[0, t]
                    fuzai = sbs_load
                    rongliang = p_sbs
                else:
                    mu_val = mu_iu[int(task_assignment[i, t]) - 1, t]
                    fuzai = iu_load[int(task_assignment[i, t]) - 1]
                    rongliang = p_iu

                if r_decision[i, t] >= chi[-1] - 1e-6:
                    mu_val = 0

                VQ = r_decision[i, t]
                SW = (r_decision[i, t] - r_previous[i, t]) ** 2
                T_w = initial_wait_times[i]
                Bu_current = buffer_state[i, t]
                T_remaining = nf * delta_t - (t) * delta_t
                bk_rate = download_rates[i, t]

                if bk_rate > r_decision[i, t]:
                    EI = 1
                else:
                    if T_remaining > 0:
                        numerator = Bu_current + T_remaining * bk_rate
                        denominator = r_decision[i, t] * T_remaining
                        if denominator > 0:
                            EI = min(1, numerator / denominator)
                        else:
                            EI = 1
                    else:
                        EI = 1

                qoe_i = alpha_qoe * VQ - beta_qoe * SW - gamma_qoe * T_w + delta_qoe * EI
                violate = lambda_val[i, t] * (r_decision[i, t] - download_rates[i, t]) + mu_val * (
                    fuzai - rongliang
                )
                lagrange_val += qoe_i - violate
                qoe_all_user += qoe_i

            convergence_qoe.append(qoe_all_user)
            convergence_lagrange.append(lagrange_val)
            if len(convergence_lagrange) > 1:
                change = abs(convergence_lagrange[-1] - convergence_lagrange[-2])
                if change < epsilon_conv:
                    break

            for i in range(len(community_users)):
                # 根据一维解析式或三次方程解更新码率
                user_idx = community_users[i]
                if iu_flags[user_idx] == 1 and cache_decision[user_idx, requested_videos[i]] == 1:
                    r_decision[i, t] = chi[-1]
                    continue

                T_remaining = nf * delta_t - (t) * delta_t
                Bu_current = buffer_state[i, t]
                bk_rate = download_rates[i, t]

                if bk_rate > r_decision[i, t]:
                    EI = 1
                else:
                    if T_remaining > 0:
                        numerator = Bu_current + T_remaining * bk_rate
                        denominator = r_decision[i, t] * T_remaining
                        if denominator > 0:
                            EI = min(1, numerator / denominator)
                        else:
                            EI = 1
                    else:
                        EI = 1

                if task_assignment[i, t] == 0:
                    mu_val = mu_sbs[0, t]
                else:
                    mu_val = mu_iu[int(task_assignment[i, t]) - 1, t]
                if r_decision[i, t] >= chi[-1] - 1e-6:
                    mu_val = 0

                if EI >= 0.999:
                    r_new = (alpha_qoe - lambda_val[i, t] - mu_val) / (2 * beta_qoe) + r_previous[i, t]
                else:
                    a = 2 * beta_qoe * T_remaining
                    b = -(alpha_qoe + 2 * beta_qoe * r_previous[i, t] - lambda_val[i, t] - mu_val) * T_remaining
                    c = 0
                    d = delta_qoe * (Bu_current + T_remaining * bk_rate)
                    coeffs = [a, b, c, d]
                    roots_complex = np.roots(coeffs)
                    real_roots = np.real(roots_complex[np.abs(np.imag(roots_complex)) < 1e-10])
                    valid_roots = real_roots[(real_roots > 0) & (real_roots <= chi.max())]
                    if valid_roots.size > 0:
                        r_new = valid_roots[0]
                    else:
                        r_new = (alpha_qoe - lambda_val[i, t] - mu_val) / (2 * beta_qoe) + r_previous[i, t]

                r_new = min(r_new, download_rates[i, t])
                r_new = max(r_new, 0)
                r_decision[i, t] = r_new

            for i in range(len(community_users)):
                lambda_val[i, t] = max(0, lambda_val[i, t] + eta * (r_decision[i, t] - download_rates[i, t]))

            sbs_load = 0.0
            iu_load = np.zeros(iu_count_per_community)
            for n in range(len(community_users)):
                if task_assignment[n, t] == 0 and r_decision[n, t] < chi[-1]:
                    sbs_load += r_decision[n, t]
            mu_sbs[0, t] = max(0, mu_sbs[0, t] + eta * (sbs_load - p_sbs))

            for n in range(iu_count_per_community):
                for m in range(len(community_users)):
                    if task_assignment[m, t] == n + 1 and r_decision[m, t] < chi[-1]:
                        iu_load[n] += r_decision[m, t]
                mu_iu[n, t] = max(0, mu_iu[n, t] + eta * (iu_load[n] - p_iu))

        r_final = np.zeros(len(community_users))
        # 将连续码率映射到离散档位，并重新检查 SBS/IU 容量是否超限
        for i in range(len(community_users)):
            if r_decision[i, t] <= chi[0]:
                lower = chi[0]
                higher = chi[0]
            elif r_decision[i, t] >= chi[-1]:
                lower = chi[-1]
                higher = chi[-1]
            else:
                upper_idx_candidates = np.where(chi >= r_decision[i, t])[0]
                if upper_idx_candidates.size == 0:
                    lower = chi[-1]
                    higher = chi[-1]
                else:
                    upper_idx = upper_idx_candidates[0]
                    if upper_idx == 0:
                        lower = chi[0]
                        higher = chi[0]
                    else:
                        lower = chi[upper_idx - 1]
                        higher = chi[upper_idx]

            Bu_current = buffer_state[i, t]
            qoe_lower = calculate_qoe(
                Bu_current,
                lower,
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
            qoe_higher = calculate_qoe(
                Bu_current,
                higher,
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
            r_final[i] = higher if qoe_higher >= qoe_lower else lower

        sbs_users = np.where(task_assignment[:, t] == 0)[0]
        sbs_users_need_transcode = [idx for idx in sbs_users if r_final[idx] < chi[-1]]
        sbs_total_load = np.sum(r_final[sbs_users_need_transcode])

        if sbs_total_load > p_sbs:
            qoe_losses = []
            user_resolution_info = []
            for idx in sbs_users_need_transcode:
                current_res = r_final[idx]
                current_res_idx = np.where(chi == current_res)[0]
                if current_res_idx.size > 0 and current_res_idx[0] > 0:
                    lower_res = chi[current_res_idx[0] - 1]
                    Bu_current = buffer_state[idx, t]
                    qoe_current = calculate_qoe(
                        Bu_current,
                        current_res,
                        r_previous[idx, t],
                        initial_wait_times[idx],
                        download_rates[idx, t],
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
                    qoe_lower = calculate_qoe(
                        Bu_current,
                        lower_res,
                        r_previous[idx, t],
                        initial_wait_times[idx],
                        download_rates[idx, t],
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
                    qoe_loss = qoe_lower - qoe_current
                    qoe_losses.append(qoe_loss)
                    user_resolution_info.append([idx, current_res, lower_res, qoe_loss])

            if qoe_losses:
                sort_indices = np.argsort(qoe_losses)[::-1]
                sorted_info = np.array(user_resolution_info)[sort_indices]
                for row in sorted_info:
                    user_i = int(row[0])
                    lower_res = row[2]
                    r_final[user_i] = lower_res
                    new_sbs_load = 0.0
                    for check_user in sbs_users_need_transcode:
                        if r_final[check_user] < chi[-1]:
                            new_sbs_load += r_final[check_user]
                    if new_sbs_load <= p_sbs:
                        break

        for j in range(iu_count_per_community):
            iu_users = np.where(task_assignment[:, t] == j + 1)[0]
            iu_users_need_transcode = [idx for idx in iu_users if r_final[idx] < chi[-1]]

            if iu_users_need_transcode:
                iu_total_load = np.sum(r_final[iu_users_need_transcode])
                if iu_total_load > p_iu:
                    qoe_losses = []
                    user_resolution_info = []
                    for idx in iu_users_need_transcode:
                        current_res = r_final[idx]
                        current_res_idx = np.where(chi == current_res)[0]
                        if current_res_idx.size > 0 and current_res_idx[0] > 0:
                            lower_res = chi[current_res_idx[0] - 1]
                            Bu_current = buffer_state[idx, t]
                            qoe_current = calculate_qoe(
                                Bu_current,
                                current_res,
                                r_previous[idx, t],
                                initial_wait_times[idx],
                                download_rates[idx, t],
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
                            qoe_lower = calculate_qoe(
                                Bu_current,
                                lower_res,
                                r_previous[idx, t],
                                initial_wait_times[idx],
                                download_rates[idx, t],
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
                            qoe_loss = qoe_lower - qoe_current
                            qoe_losses.append(qoe_loss)
                            user_resolution_info.append([idx, current_res, lower_res, qoe_loss])

                    if qoe_losses:
                        sort_indices = np.argsort(qoe_losses)[::-1]
                        sorted_info = np.array(user_resolution_info)[sort_indices]
                        for row in sorted_info:
                            user_i = int(row[0])
                            lower_res = row[2]
                            r_final[user_i] = lower_res
                            new_iu_load = 0.0
                            for check_user in iu_users_need_transcode:
                                if r_final[check_user] < chi[-1]:
                                    new_iu_load += r_final[check_user]
                            if new_iu_load > p_iu and (np.where(chi == lower_res)[0][0] > 0):
                                lower_idx = np.where(chi == lower_res)[0][0]
                                r_final[user_i] = chi[max(0, lower_idx - 1)]
                            elif new_iu_load <= p_iu:
                                break

        r_decision[:, t] = r_final

        final_total_qoe = 0.0
        for i in range(len(community_users)):
            Bu_current = buffer_state[i, t]
            qoe_i = calculate_qoe(
                Bu_current,
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
