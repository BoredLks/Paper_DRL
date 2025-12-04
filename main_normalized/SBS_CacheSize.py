"""Python translation of MATLAB SBS_CacheSize.m."""

from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from _algo_runner import build_args_kwargs, run_mean_for_algorithm  # type: ignore
    from cal_Movielens import DEFAULT_ML_PATH  # type: ignore
    from sim_utils import apply_global_seed  # type: ignore
else:
    from ._algo_runner import build_args_kwargs, run_mean_for_algorithm
    from .cal_Movielens import DEFAULT_ML_PATH
    from .sim_utils import apply_global_seed


def main():
    start_time = time.time()
    try:
        seed_value = int(os.environ.get("SIM_RANDOM_SEED", "42"))
    except ValueError:
        seed_value = 42
    apply_global_seed(seed_value)

    movielens_path = os.environ.get("ML_DATA_PATH", DEFAULT_ML_PATH)
    movielens_data = np.loadtxt(movielens_path)

    h = 3
    user_per_community = 50
    total_users = h * user_per_community
    F = 200
    Per_m = 0.3
    iu_count_per_community = round(user_per_community * Per_m)
    nf = 100
    chi = np.array([1.5, 2, 5, 10, 12], dtype=float)
    chi_star = chi[-1]
    delta_t = 1
    v_chi_star = chi_star * delta_t / 8

    region_size = 1500
    sbs_coverage = 150
    iu_coverage = 50
    community_radius = 150
    max_movement_dist = 2

    T_small = 100
    gamma_m = 0.56

    alpha = 0.2
    beta = 0.4
    gamma_val = 0.4

    B = 2
    P_sbs = 1
    P_iu = 0.3
    N0 = 1e-7
    K = 1
    epsilon = 3
    slowfading_dB = 3

    alpha_qoe = 0.4
    beta_qoe = 0.2
    gamma_qoe = 0.2
    delta_qoe = 0.2

    if abs(alpha_qoe + beta_qoe + gamma_qoe + delta_qoe - 1) > 1e-6:
        raise ValueError("QoE权重参数之和必须为 1")

    D_eg = 30 * nf * v_chi_star
    D_iu = 10 * nf * v_chi_star

    p_sbs = 100
    p_iu = 10
    D_bf = 10
    t_cloud = 1
    t_propagation = 0.1

    target_community = 1
    eta = 0.01
    epsilon_conv = 0.1
    max_iterations = 200
    run_num = 1000

    qoe_algorithm_number = np.zeros((7, 10))

    for number in range(1, 11):
        print(f"处理参数 {number}/10...")

        D_eg_current = 5 * number * nf * v_chi_star

        for algorithm_num in range(1, 8):
            print(f"  算法 {algorithm_num}/7...")
            params = {
                "h": h,
                "user_per_community": user_per_community,
                "total_users": total_users,
                "F": F,
                "iu_count_per_community": iu_count_per_community,
                "nf": nf,
                "chi": chi,
                "delta_t": delta_t,
                "v_chi_star": v_chi_star,
                "region_size": region_size,
                "sbs_coverage": sbs_coverage,
                "iu_coverage": iu_coverage,
                "community_radius": community_radius,
                "max_movement_dist": max_movement_dist,
                "D_eg": D_eg_current,
                "D_iu": D_iu,
                "T_small": T_small,
                "gamma_m": gamma_m,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma_val,
                "B": B,
                "P_sbs": P_sbs,
                "P_iu": P_iu,
                "N0": N0,
                "K": K,
                "epsilon": epsilon,
                "slowfading_dB": slowfading_dB,
                "alpha_qoe": alpha_qoe,
                "beta_qoe": beta_qoe,
                "gamma_qoe": gamma_qoe,
                "delta_qoe": delta_qoe,
                "p_sbs": p_sbs,
                "p_iu": p_iu,
                "D_bf": D_bf,
                "t_cloud": t_cloud,
                "t_propagation": t_propagation,
                "target_community": target_community,
                "eta": eta,
                "epsilon_conv": epsilon_conv,
                "max_iterations": max_iterations,
                "movielens_path": movielens_path,
                "movielens_data": movielens_data,
            }
            args_common, kwargs = build_args_kwargs(algorithm_num, params)
            qoe_algorithm_number[algorithm_num - 1, number - 1] = run_mean_for_algorithm(
                algorithm_num, run_num, args_common, kwargs
            )
            print(
                f"    算法 {algorithm_num} 完成，平均QoE: "
                f"{qoe_algorithm_number[algorithm_num - 1, number - 1]:.6f}"
            )

    plt.figure(figsize=(8, 6))
    x_data = np.arange(1, number + 1) * 5
    plt.plot(x_data, qoe_algorithm_number[0, :], "-o", linewidth=3)
    plt.plot(x_data, qoe_algorithm_number[1, :], "-*", linewidth=3)
    plt.plot(x_data, qoe_algorithm_number[4, :], "-d", linewidth=3)
    plt.plot(x_data, qoe_algorithm_number[3, :], "-s", linewidth=3)
    plt.plot(x_data, qoe_algorithm_number[2, :], "-x", linewidth=3)
    plt.plot(x_data, qoe_algorithm_number[5, :], "-v", linewidth=3)
    plt.plot(x_data, qoe_algorithm_number[6, :], "-^", linewidth=3)
    plt.xlim([5, number * 5])
    plt.xticks(x_data)
    plt.xlabel("Cache size of SBS (Files)", fontsize=18)
    plt.ylabel("QoE", fontsize=18)
    plt.legend(["Proposed", "LMPC", "GMPC", "HC", "RAC", "NCC", "NAV"], loc="best", fontsize=12)
    plt.grid(True)
    plt.savefig("Figure/SBS_CacheSize.png", bbox_inches="tight")

    print("\n所有计算完成！")
    print(f"耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
