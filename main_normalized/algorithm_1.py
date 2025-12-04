"""Algorithm 1 translated from MATLAB algorithm_1.m."""

# 算法 1（LMPC）流程与算法 0 类似，但缓存策略改为“本地最热门”：
# 1. 根据物理距离统计每个节点能覆盖到的邻居用户；
# 2. 统计邻居请求概率作为热门度，SBS/IU 按热门度缓存；
# 其余步骤（速率、QoE 优化等）与算法 0 一致。

from __future__ import annotations

from .cal_Initial_position import cal_Initial_position
from .cal_Select_iu import cal_Select_iu
from .cal_user_file_req_prob import cal_user_file_req_prob
from .cal_cache_decision_1 import cal_cache_decision_1
from .cal_target_community_users import cal_target_community_users
from .cal_requested_videos import cal_requested_videos
from .cal_download_rates_task_assignment import cal_download_rates_task_assignment
from .cal_initial_wait_times import cal_initial_wait_times
from .cal_Initialize_optimization_variables import cal_Initialize_optimization_variables
from .cal_final_alltime_qoe import cal_final_alltime_qoe
from .cal_hit_rate import cal_hit_rate


def algorithm_1(
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
    eta,
    epsilon_conv,
    max_iterations,
):
    # Step1: 构造场景，得到所有用户/SBS 的运动轨迹
    sbs_positions, per_user_positions, now_user_positions, _ = cal_Initial_position(
        h,
        user_per_community,
        total_users,
        region_size,
        community_radius,
        max_movement_dist,
        T_small,
    )
    # Step2: 每个社区随机选出若干 IU 作为协作节点
    iu_indices, iu_flags = cal_Select_iu(
        h,
        user_per_community,
        total_users,
        iu_count_per_community,
        iu_coverage,
        per_user_positions,
    )
    user_file_req_prob = cal_user_file_req_prob(total_users, F, gamma_m)
    # Step3: 采用局部最热门策略，依据邻域请求概率缓存内容
    cache_decision = cal_cache_decision_1(
        h,
        total_users,
        F,
        iu_count_per_community,
        sbs_coverage,
        iu_coverage,
        nf,
        v_chi_star,
        D_eg,
        D_iu,
        sbs_positions,
        per_user_positions,
        iu_indices,
        iu_flags,
        user_file_req_prob,
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
        lambda_val,
        mu_sbs,
        mu_iu,
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
    final_alltime_qoe = cal_final_alltime_qoe(
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
    )
    cache_hit_rate = cal_hit_rate(
        total_users, nf, target_community, cache_decision, community_users, requested_videos, task_assignment
    )
    return final_alltime_qoe, cache_hit_rate
