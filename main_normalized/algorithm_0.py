"""Algorithm 0 translated from MATLAB algorithm_0.m."""

# 算法 0 是论文中的“社交-物理双域 + QoE 优化”的完整流程：
# 1. 随机生成拓扑、选择 IU；
# 2. 通过 Movielens 相似度构建社交图，物理链路得到链路权重；
# 3. 结合社交/物理权重计算缓存偏好并下发缓存策略；
# 4. 计算目标社区的下载速率、排队和 QoE，并输出平均 QoE 与命中率。

from __future__ import annotations

import numpy as np

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
from .cal_final_alltime_qoe import cal_final_alltime_qoe
from .cal_hit_rate import cal_hit_rate


def algorithm_0(
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
    eta,
    epsilon_conv,
    max_iterations,
    data=None,
    movielens_path=None,
):
    # Step 1: 随机生成 SBS/用户初始位置与移动轨迹，同时返回每个用户所属社区
    sbs_positions, per_user_positions, now_user_positions, user_community = cal_Initial_position(
        h,
        user_per_community,
        total_users,
        region_size,
        community_radius,
        max_movement_dist,
        T_small,
    )
    # Step 2: 为每个社区选出固定数量的 IU（重要用户/设备）
    iu_indices, iu_flags = cal_Select_iu(
        h,
        user_per_community,
        total_users,
        iu_count_per_community,
        iu_coverage,
        per_user_positions,
    )
    # Step 3: 利用 Movielens 评分得到用户相似度矩阵和请求分布
    sim_matrix, user_file_req_prob = cal_Movielens(
        total_users, F, gamma_m, data=data, file_path=movielens_path
    )
    # Step 4: 计算社交图边权（亲密度、相似度、重要度）
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
    # Step 5: 计算物理链路边权（无线链路容量/可靠性）
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
    # Step 6: 社交图和物理图逐元素相乘，得到综合边权，进一步计算内容偏好并做缓存
    joint_edge_weights = pl_edge_weights * sg_edge_weights
    cache_preference = cal_cache_preference(h, total_users, F, user_file_req_prob, joint_edge_weights)
    cache_decision = cal_cache_decision_0(
        h, total_users, F, iu_count_per_community, nf, v_chi_star, D_eg, D_iu, iu_indices, cache_preference
    )

    # Step 7: 取出目标社区用户及其请求的视频
    community_users = cal_target_community_users(user_per_community, target_community)
    requested_videos = cal_requested_videos(user_file_req_prob, community_users)
    # Step 8: 仿真无线传输，得到每个用户在每个时隙的下载速率和任务（SBS/IU 分配）
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
    # Step 9: 根据缓存位置及速率，估算开始播放前的等待时间
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
    # Step10: 初始化 QoE 优化算法所需的变量（码率决策、乘子、缓冲等）
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
    # Step11: 运行论文中的迭代缓存/码率优化，得到总 QoE
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
    # Step12: 根据最终缓存与任务信息计算命中率
    cache_hit_rate = cal_hit_rate(
        total_users, nf, target_community, cache_decision, community_users, requested_videos, task_assignment
    )
    return final_alltime_qoe, cache_hit_rate
