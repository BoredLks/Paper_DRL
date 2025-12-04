"""Algorithm 6 translated from MATLAB algorithm_6.m."""

# 算法 6：与算法 0 的缓存/速率流程完全一致，但视频码率选择阶段
# 使用“固定档位”策略——即不进行迭代优化，自适应模块被关闭，用于对照。

from __future__ import annotations

from .cal_Initial_position import cal_Initial_position
from .cal_Select_iu import cal_Select_iu
from .cal_Movielens import cal_Movielens
from .cal_sg_edge_weights import cal_sg_edge_weights
from .cal_pl_edge_weights import cal_pl_edge_weights
from .cal_cache_preference import cal_cache_preference
from .cal_cache_decision_0 import cal_cache_decision_0
from .cal_target_community_users import cal_target_community_users
from .cal_requested_videos import cal_requested_videos
from .cal_download_rates_task_assignment import cal_download_rates_task_assignment
from .cal_initial_wait_times import cal_initial_wait_times
from .cal_Initialize_optimization_variables import cal_Initialize_optimization_variables
from .cal_final_alltime_qoe_without_adaption import cal_final_alltime_qoe_without_adaption
from .cal_hit_rate import cal_hit_rate


def algorithm_6(
    h,
    user_per_community,
    total_users,
    F,
    iu_count_per_community,
    nf,
    chi,
    delta_t,
    v_chi_star,
    region_size,
    sbs_coverage,
    iu_coverage,
    community_radius,
    max_movement_dist,
    D_eg,
    D_iu,
    T_small,
    gamma_m,
    alpha,
    beta,
    gamma,
    B,
    P_sbs,
    P_iu,
    N0,
    K,
    epsilon,
    slowfading_dB,
    alpha_qoe,
    beta_qoe,
    gamma_qoe,
    delta_qoe,
    p_sbs,
    p_iu,
    D_bf,
    t_cloud,
    t_propagation,
    target_community,
    data=None,
    movielens_path=None,
):
    sbs_positions, per_user_positions, now_user_positions, user_community = cal_Initial_position(
        h,
        user_per_community,
        total_users,
        region_size,
        community_radius,
        max_movement_dist,
        T_small,
    )
    iu_indices, iu_flags = cal_Select_iu(
        h,
        user_per_community,
        total_users,
        iu_count_per_community,
        iu_coverage,
        per_user_positions,
    )
    sim_matrix, user_file_req_prob = cal_Movielens(
        total_users, F, gamma_m, data=data, file_path=movielens_path
    )
    sg_edge_weights = cal_sg_edge_weights(
        h,
        total_users,
        sbs_coverage,
        alpha,
        beta,
        gamma,
        sbs_positions,
        per_user_positions,
        user_community,
        sim_matrix,
    )
    pl_edge_weights = cal_pl_edge_weights(
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
        per_user_positions,
        iu_indices,
        iu_flags,
    )
    joint_edge_weights = pl_edge_weights * sg_edge_weights
    cache_preference = cal_cache_preference(h, total_users, F, user_file_req_prob, joint_edge_weights)
    cache_decision = cal_cache_decision_0(
        h, total_users, F, iu_count_per_community, nf, v_chi_star, D_eg, D_iu, iu_indices, cache_preference
    )

    community_users = cal_target_community_users(user_per_community, target_community)
    requested_videos = cal_requested_videos(user_file_req_prob, community_users)
    download_rates, task_assignment = cal_download_rates_task_assignment(
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
        now_user_positions,
        iu_indices,
        iu_flags,
        cache_decision,
        community_users,
        requested_videos,
    )
    initial_wait_times = cal_initial_wait_times(
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
        now_user_positions,
        iu_indices,
        iu_flags,
        cache_decision,
        community_users,
        requested_videos,
        download_rates,
    )
    (
        r_decision,
        r_previous,
        _,
        _,
        _,
        buffer_state,
    ) = cal_Initialize_optimization_variables(
        chi,
        iu_count_per_community,
        nf,
        D_bf,
        iu_flags,
        cache_decision,
        community_users,
        requested_videos,
    )
    # 与算法 0 不同之处：调用“无自适应”版本的 QoE 评估，码率固定为 chi[1]
    final_alltime_qoe = cal_final_alltime_qoe_without_adaption(
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
    )
    cache_hit_rate = cal_hit_rate(
        total_users, nf, target_community, cache_decision, community_users, requested_videos, task_assignment
    )
    return final_alltime_qoe, cache_hit_rate
