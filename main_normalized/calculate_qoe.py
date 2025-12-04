"""QoE calculation helper converted from MATLAB calculate_qoe.m."""

from __future__ import annotations

import numpy as np


def calculate_qoe(
    Bu_current,
    resolution,
    r_prev,
    wait_time,
    download_rate,
    alpha,
    beta,
    gamma,
    delta,
    nf,
    delta_t,
    timeslot,
    chi,
    initial_wait_times,
):
    """
    Compute QoE with normalization of VQ, SW, TW (aligned to MATLAB implementation).
    """
    chi_arr = np.asarray(chi, dtype=float)
    VQ = resolution
    SW = (resolution - r_prev) ** 2
    T_w = wait_time

    chi_min = float(np.min(chi_arr)) if chi_arr.size > 0 else 0.0
    chi_max = float(np.max(chi_arr)) if chi_arr.size > 0 else 1.0
    res_range = chi_max - chi_min
    if res_range <= 0:
        res_range = 1.0
    T_w_max = float(np.max(initial_wait_times)) if np.size(initial_wait_times) > 0 else 0.0

    VQ_norm = VQ / chi_max if chi_max != 0 else 0.0
    delta_r = (resolution - r_prev) / res_range
    SW_norm = delta_r**2
    SW_norm = min(1.0, max(0.0, SW_norm))
    if T_w_max > 0:
        T_w_norm = min(T_w / T_w_max, 1.0)
    else:
        T_w_norm = 0.0

    T_remaining = nf * delta_t - (timeslot - 1) * delta_t
    Bh = download_rate
    if Bh > resolution:
        EI = 1.0
    else:
        if T_remaining > 0:
            numerator = Bu_current + T_remaining * Bh
            denominator = resolution * T_remaining
            if denominator > 0:
                EI = min(1.0, numerator / denominator)
            else:
                EI = 1.0
        else:
            EI = 1.0

    qoe = alpha * VQ_norm - beta * SW_norm - gamma * T_w_norm + delta * EI
    return qoe
