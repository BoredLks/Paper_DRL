"""Com_Parameter_DRL: sweep communication parameter K with DRL-based algorithm (algorithm 8)."""

from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir)
    sys.path.append(os.path.dirname(base_dir))
    from _algo_runner import build_args_kwargs, run_mean_for_algorithm, run_algorithm_by_index  # type: ignore
    from cal_Movielens import DEFAULT_ML_PATH  # type: ignore
    from sim_utils import apply_global_seed, resolve_movielens_path  # type: ignore
else:
    from ._algo_runner import build_args_kwargs, run_mean_for_algorithm, run_algorithm_by_index
    from .cal_Movielens import DEFAULT_ML_PATH
    from .sim_utils import apply_global_seed, resolve_movielens_path


def main():
    start_time = time.time()
    try:
        seed_value = int(os.environ.get("SIM_RANDOM_SEED", "42"))
    except ValueError:
        seed_value = 42
    apply_global_seed(seed_value)
    movielens_path = resolve_movielens_path(DEFAULT_ML_PATH)

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

    alpha_qoe = 0.45
    beta_qoe = 0.05
    gamma_qoe = 0.3
    delta_qoe = 0.2

    if abs(alpha_qoe + beta_qoe + gamma_qoe + delta_qoe - 1) > 1e-6:
        raise ValueError("QoE权重参数之和必须为 1")

    D_eg = 30 * nf * v_chi_star
    D_iu = 10 * nf * v_chi_star

    p_sbs = 100
    p_iu = 10
    D_bf = 10
    t_cloud = 60
    t_propagation = 30

    target_community = 1
    eta = 0.001
    epsilon_conv = 0.1
    max_iterations = 1000
    run_num = 10

    qoe_algorithm_number = np.zeros((8, 10))

    for number in range(1, 11):
        print(f"处理参数 {number}/10...")
        K_current = 0.1 * number  # 通信参数 K 逐步增大

        # 如需遍历全部算法，设置为 range(1, 9)
        algorithms_to_run = [8]
        for algorithm_num in algorithms_to_run:
            print(f"  算法 {algorithm_num}/8...")

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
                "D_eg": D_eg,
                "D_iu": D_iu,
                "T_small": T_small,
                "gamma_m": gamma_m,
                "alpha": alpha,
                "beta": beta,
                "gamma_val": gamma_val,
                "B": B,
                "P_sbs": P_sbs,
                "P_iu": P_iu,
                "N0": N0,
                "K_current": K_current,
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
            }
            args_common, kwargs = build_args_kwargs(algorithm_num, params)

            if algorithm_num == 8:
                model_dir = os.path.join("models", "drl")
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"K_{K_current:.2f}.pth")
                training_eps = int(os.environ.get("DRL_TRAIN_EPISODES", "300"))
                qoe_vals = []

                # 第一次训练并保存模型
                res = run_algorithm_by_index(
                    algorithm_num,
                    *args_common,
                    training_episodes=training_eps,
                    model_path=model_path,
                    train_model=True,
                    **kwargs,
                )
                qoe_vals.append(res[0] if isinstance(res, tuple) else res)

                # 其余 run_num-1 次仅加载模型评估
                for _ in range(1, run_num):
                    res = run_algorithm_by_index(
                        algorithm_num,
                        *args_common,
                        training_episodes=0,
                        model_path=model_path,
                        train_model=False,
                        **kwargs,
                    )
                    qoe_vals.append(res[0] if isinstance(res, tuple) else res)

                qoe_algorithm_number[algorithm_num - 1, number - 1] = float(np.mean(qoe_vals))
            else:
                qoe_algorithm_number[algorithm_num - 1, number - 1] = run_mean_for_algorithm(
                    algorithm_num,
                    run_num,
                    args_common,
                    kwargs,
                    sequential=False,
                )
            print(f"    算法 {algorithm_num} 完成，平均QoE: {qoe_algorithm_number[algorithm_num - 1, number - 1]:.6f}")

    plt.figure(figsize=(8, 6))
    x_data = np.arange(1, number + 1) * 0.1
    plt.plot(x_data, qoe_algorithm_number[7, :], "-p", linewidth=3, label="GMPC+DRL")
    plt.xlim([0.1, number * 0.1])
    plt.xticks(x_data)
    plt.xlabel("Communication Parameter K", fontsize=18)
    plt.ylabel("QoE", fontsize=18)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True)
    plt.savefig("Figure/Com_Parameter_DRL.png", bbox_inches="tight")

    print("\n所有计算完成！")
    print(f"耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
