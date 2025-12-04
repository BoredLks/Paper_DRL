"""Shared helper to dispatch algorithm functions based on index."""

from __future__ import annotations

import copy
import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor


def _import_algorithms():
    """Support running as script (__package__ empty) or as package import."""
    if __package__ in (None, ""):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)
        pkg_name = os.path.basename(base_dir)
        # 允许通过包名导入以支持文件内的相对导入
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        pkg_prefix = f"{pkg_name}."
    else:
        pkg_prefix = __package__ + "."
    algo_mod = lambda name: importlib.import_module(pkg_prefix + name)  # noqa: E731
    return (
        algo_mod("algorithm_0").algorithm_0,
        algo_mod("algorithm_1").algorithm_1,
        algo_mod("algorithm_2").algorithm_2,
        algo_mod("algorithm_3").algorithm_3,
        algo_mod("algorithm_4").algorithm_4,
        algo_mod("algorithm_5").algorithm_5,
        algo_mod("algorithm_6").algorithm_6,
        algo_mod("algorithm_7").algorithm_7,
    )


algorithm_0, algorithm_1, algorithm_2, algorithm_3, algorithm_4, algorithm_5, algorithm_6, algorithm_7 = (
    _import_algorithms()
)


def run_algorithm_by_index(idx, *args, **kwargs):
    if idx == 1:
        return algorithm_0(*args, **kwargs)
    if idx == 2:
        return algorithm_1(*args, **kwargs)
    if idx == 3:
        return algorithm_2(*args, **kwargs)
    if idx == 4:
        return algorithm_3(*args, **kwargs)
    if idx == 5:
        return algorithm_4(*args, **kwargs)
    if idx == 6:
        return algorithm_5(*args, **kwargs)
    if idx == 7:
        return algorithm_6(*args, **kwargs)
    if idx == 8:
        return algorithm_7(*args, **kwargs)
    raise ValueError(f"Unknown algorithm index {idx}")


def _run_job(job):
    idx, args, kwargs = job
    return run_algorithm_by_index(idx, *args, **kwargs)


def run_jobs_parallel(jobs, max_workers: int | None = None):
    """Run jobs (idx, args, kwargs) in parallel; returns list of results."""
    workers = max_workers or max(1, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_run_job, jobs))
    return results


def run_mean_for_algorithm(
    idx,
    run_num,
    args,
    kwargs=None,
    max_workers=None,
    progress_callback=None,
    sequential=False,
):
    """Run algorithm idx run_num times and return mean QoE.

    If sequential is True, jobs execute in the current process (needed when algorithms
    manage GPU resources or need fine-grained progress reporting). Otherwise jobs run
    in parallel via ProcessPoolExecutor.
    """
    kwargs = kwargs or {}
    qoe_vals = []

    def _append_result(res, done_count):
        if isinstance(res, tuple):
            qoe_vals.append(res[0])
        else:
            qoe_vals.append(res)
        if progress_callback:
            progress_callback(done_count, run_num, res)

    if sequential:
        for i in range(1, run_num + 1):
            if progress_callback:
                progress_callback(i, run_num, "start")
            args_copy = copy.deepcopy(args)
            kwargs_copy = copy.deepcopy(kwargs)
            res = run_algorithm_by_index(idx, *args_copy, **kwargs_copy)
            _append_result(res, i)
    else:
        jobs = [(idx, args, kwargs) for _ in range(run_num)]
        results = run_jobs_parallel(jobs, max_workers=max_workers)
        for i, res in enumerate(results, 1):
            _append_result(res, i)

    return float(sum(qoe_vals) / len(qoe_vals))


def _ml_kwargs(params):
    """Collect Movielens-related kwargs."""
    kwargs = {}
    if params.get("movielens_data") is not None:
        kwargs["data"] = params["movielens_data"]
    if params.get("movielens_path"):
        kwargs["movielens_path"] = params["movielens_path"]
    return kwargs


def _gamma_value(params):
    return params.get("gamma_val", params.get("gamma"))


def _k_value(params):
    return params.get("K_current", params.get("K"))


def build_args_kwargs(idx, p):
    """Build (args, kwargs) for a given algorithm index based on params dict p."""
    gamma_val = _gamma_value(p)
    k_val = _k_value(p)
    if idx == 1:
        args = (
            p["h"],
            p["user_per_community"],
            p["total_users"],
            p["F"],
            p["iu_count_per_community"],
            p["nf"],
            p["chi"],
            p["delta_t"],
            p["v_chi_star"],
            p["region_size"],
            p["sbs_coverage"],
            p["iu_coverage"],
            p["community_radius"],
            p["max_movement_dist"],
            p["D_eg"],
            p["D_iu"],
            p["T_small"],
            p["gamma_m"],
            p["alpha"],
            p["beta"],
            gamma_val,
            p["B"],
            p["P_sbs"],
            p["P_iu"],
            p["N0"],
            k_val,
            p["epsilon"],
            p["slowfading_dB"],
            p["alpha_qoe"],
            p["beta_qoe"],
            p["gamma_qoe"],
            p["delta_qoe"],
            p["p_sbs"],
            p["p_iu"],
            p["D_bf"],
            p["t_cloud"],
            p["t_propagation"],
            p["target_community"],
            p["eta"],
            p["epsilon_conv"],
            p["max_iterations"],
        )
        kwargs = _ml_kwargs(p)
    elif idx in (2, 3, 4):
        args = (
            p["h"],
            p["user_per_community"],
            p["total_users"],
            p["F"],
            p["iu_count_per_community"],
            p["nf"],
            p["chi"],
            p["delta_t"],
            p["v_chi_star"],
            p["region_size"],
            p["sbs_coverage"],
            p["iu_coverage"],
            p["community_radius"],
            p["max_movement_dist"],
            p["D_eg"],
            p["D_iu"],
            p["T_small"],
            p["gamma_m"],
            p["B"],
            p["P_sbs"],
            p["P_iu"],
            p["N0"],
            k_val,
            p["epsilon"],
            p["slowfading_dB"],
            p["alpha_qoe"],
            p["beta_qoe"],
            p["gamma_qoe"],
            p["delta_qoe"],
            p["p_sbs"],
            p["p_iu"],
            p["D_bf"],
            p["t_cloud"],
            p["t_propagation"],
            p["target_community"],
            p["eta"],
            p["epsilon_conv"],
            p["max_iterations"],
        )
        kwargs = {}
    elif idx == 5:
        args = (
            p["h"],
            p["user_per_community"],
            p["total_users"],
            p["F"],
            p["iu_count_per_community"],
            p["nf"],
            p["chi"],
            p["delta_t"],
            p["v_chi_star"],
            p["region_size"],
            p["sbs_coverage"],
            p["iu_coverage"],
            p["community_radius"],
            p["max_movement_dist"],
            p["D_eg"],
            p["D_iu"],
            p["T_small"],
            p["gamma_m"],
            p["B"],
            p["P_sbs"],
            p["P_iu"],
            p["N0"],
            k_val,
            p["epsilon"],
            p["slowfading_dB"],
            p["alpha_qoe"],
            p["beta_qoe"],
            p["gamma_qoe"],
            p["delta_qoe"],
            p["p_sbs"],
            p["p_iu"],
            p["D_bf"],
            p["t_cloud"],
            p["t_propagation"],
            p["target_community"],
            p["eta"],
            p["epsilon_conv"],
            p["max_iterations"],
        )
        kwargs = {}
    elif idx == 8:
        args = (
            p["h"],
            p["user_per_community"],
            p["total_users"],
            p["F"],
            p["iu_count_per_community"],
            p["nf"],
            p["chi"],
            p["delta_t"],
            p["v_chi_star"],
            p["region_size"],
            p["sbs_coverage"],
            p["iu_coverage"],
            p["community_radius"],
            p["max_movement_dist"],
            p["D_eg"],
            p["D_iu"],
            p["T_small"],
            p["gamma_m"],
            p["B"],
            p["P_sbs"],
            p["P_iu"],
            p["N0"],
            k_val,
            p["epsilon"],
            p["slowfading_dB"],
            p["alpha_qoe"],
            p["beta_qoe"],
            p["gamma_qoe"],
            p["delta_qoe"],
            p["p_sbs"],
            p["p_iu"],
            p["D_bf"],
            p["t_cloud"],
            p["t_propagation"],
            p["target_community"],
            p["eta"],
            p["epsilon_conv"],
            p["max_iterations"],
        )
        kwargs = _ml_kwargs(p)
    elif idx == 6:
        args = (
            p["h"],
            p["user_per_community"],
            p["total_users"],
            p["F"],
            p["iu_count_per_community"],
            p["nf"],
            p["chi"],
            p["delta_t"],
            p["v_chi_star"],
            p["region_size"],
            p["sbs_coverage"],
            p["iu_coverage"],
            p["community_radius"],
            p["max_movement_dist"],
            p["D_eg"],
            p["T_small"],
            p["gamma_m"],
            p["alpha"],
            p["beta"],
            gamma_val,
            p["B"],
            p["P_sbs"],
            p["P_iu"],
            p["N0"],
            k_val,
            p["epsilon"],
            p["slowfading_dB"],
            p["alpha_qoe"],
            p["beta_qoe"],
            p["gamma_qoe"],
            p["delta_qoe"],
            p["p_sbs"],
            p["p_iu"],
            p["D_bf"],
            p["t_cloud"],
            p["t_propagation"],
            p["target_community"],
            p["eta"],
            p["epsilon_conv"],
            p["max_iterations"],
        )
        kwargs = _ml_kwargs(p)
    else:  # idx == 7
        args = (
            p["h"],
            p["user_per_community"],
            p["total_users"],
            p["F"],
            p["iu_count_per_community"],
            p["nf"],
            p["chi"],
            p["delta_t"],
            p["v_chi_star"],
            p["region_size"],
            p["sbs_coverage"],
            p["iu_coverage"],
            p["community_radius"],
            p["max_movement_dist"],
            p["D_eg"],
            p["D_iu"],
            p["T_small"],
            p["gamma_m"],
            p["alpha"],
            p["beta"],
            gamma_val,
            p["B"],
            p["P_sbs"],
            p["P_iu"],
            p["N0"],
            k_val,
            p["epsilon"],
            p["slowfading_dB"],
            p["alpha_qoe"],
            p["beta_qoe"],
            p["gamma_qoe"],
            p["delta_qoe"],
            p["p_sbs"],
            p["p_iu"],
            p["D_bf"],
            p["t_cloud"],
            p["t_propagation"],
            p["target_community"],
        )
        kwargs = _ml_kwargs(p)
    return args, kwargs
