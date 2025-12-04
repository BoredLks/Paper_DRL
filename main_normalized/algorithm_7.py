"""Algorithm 7: GMPC Caching with DRL-based Video Adaptation."""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt

from .cal_Initial_position import cal_Initial_position
from .cal_Select_iu import cal_Select_iu
from .cal_user_file_req_prob import cal_user_file_req_prob
from .cal_cache_decision_4 import cal_cache_decision_4
from .cal_target_community_users import cal_target_community_users
from .cal_requested_videos import cal_requested_videos
from .cal_download_rates_task_assignment import cal_download_rates_task_assignment
from .cal_initial_wait_times import cal_initial_wait_times
from .cal_final_alltime_qoe_drl import cal_final_alltime_qoe_drl
from .cal_hit_rate import cal_hit_rate


def _maybe_log_learning_curve(history: list[float]) -> None:
    """根据环境变量决定是否将训练曲线保存成图片"""
    enable = os.environ.get("DRL_LOG_CURVE", "1").lower()
    if enable not in ("1", "true", "yes"):
        return
    if not history:
        return
    directory = os.environ.get("DRL_CURVE_DIR", "Figure")
    os.makedirs(directory, exist_ok=True)
    custom_file = os.environ.get("DRL_CURVE_FILE")
    if custom_file:
        if os.path.isabs(custom_file):
            path = custom_file
        else:
            path = os.path.join(directory, custom_file)
    else:
        slug = f"drl_curve_{int(time.time()*1000)}_{os.getpid()}.png"
        path = os.path.join(directory, slug)
    plt.figure(figsize=(6, 4))
    plt.plot(history, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Episode QoE")
    plt.title("DRL Learning Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def algorithm_7(
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
    movielens_path=None,
    *,
    training_episodes: int = 200,
    model_path: str | None = None,
    train_model: bool = True,
):
    # Step1: 构造仿真场景，得到 SBS、全部用户的初始位置以及运动轨迹
    sbs_positions, per_user_positions, now_user_positions, _ = cal_Initial_position(
        h,
        user_per_community,
        total_users,
        region_size,
        community_radius,
        max_movement_dist,
        T_small,
    )
    # Step2: 在每个社区中选出 IU（重要用户），记录其数组索引和标识
    iu_indices, iu_flags = cal_Select_iu(
        h,
        user_per_community,
        total_users,
        iu_count_per_community,
        iu_coverage,
        per_user_positions,
    )
    user_file_req_prob = cal_user_file_req_prob(total_users, F, gamma_m)
    # Step3: GMPC 全局热门内容缓存策略，给出 SBS/IU 缓存矩阵
    cache_decision = cal_cache_decision_4(
        h,
        user_per_community,
        total_users,
        F,
        iu_count_per_community,
        nf,
        v_chi_star,
        D_eg,
        D_iu,
        iu_indices,
        user_file_req_prob,
    )

    # 仅抽取目标社区用户作为 DRL 控制对象
    community_users = cal_target_community_users(user_per_community, target_community)
    requested_videos = cal_requested_videos(user_file_req_prob, community_users)
    # 先生成一个“评估样本”，训练结束后会在该样本上算 QoE
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
    eval_sample = (download_rates, task_assignment, initial_wait_times)

    def episode_sampler():
        """生成一个新的随机 episode，包括位置、链路速率和初始化等待时间"""
        sbs_positions_ep, _, now_user_positions_ep, _ = cal_Initial_position(
            h,
            user_per_community,
            total_users,
            region_size,
            community_radius,
            max_movement_dist,
            T_small,
        )
        download_rates_ep, task_assignment_ep = cal_download_rates_task_assignment(
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
            sbs_positions_ep,
            now_user_positions_ep,
            iu_indices,
            iu_flags,
            cache_decision,
            community_users,
            requested_videos,
        )
        initial_wait_ep = cal_initial_wait_times(
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
            now_user_positions_ep,
            iu_indices,
            iu_flags,
            cache_decision,
            community_users,
            requested_videos,
            download_rates_ep,
        )
        return download_rates_ep, task_assignment_ep, initial_wait_ep

    # 真正执行 DRL 训练：episode_sampler 提供环境随机性
    log_interval = int(os.environ.get("DRL_LOG_INTERVAL", "10"))

    final_alltime_qoe, training_history = cal_final_alltime_qoe_drl(
        nf,
        chi,
        delta_t,
        alpha_qoe,
        beta_qoe,
        gamma_qoe,
        delta_qoe,
        D_bf,
        community_users,
        requested_videos,
        cache_decision,
        iu_flags,
        p_sbs,
        p_iu,
        iu_count_per_community,
        episode_sampler,
        eval_sample=eval_sample,
        training_episodes=training_episodes,
        model_path=model_path,
        train_model=train_model,
        log_interval=log_interval,
    )
    # 如果需要查看训练效果，保存学习曲线
    _maybe_log_learning_curve(training_history)
    cache_hit_rate = cal_hit_rate(
        total_users, nf, target_community, cache_decision, community_users, requested_videos, task_assignment
    )
    return final_alltime_qoe, cache_hit_rate
