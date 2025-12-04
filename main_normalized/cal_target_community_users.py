"""Target community user selection translated from MATLAB cal_target_community_users.m."""

from __future__ import annotations

import numpy as np


def cal_target_community_users(user_per_community, target_community):
    """返回目标社区内的用户编号区间（社区按顺序排列）"""
    start_idx = (target_community - 1) * user_per_community
    end_idx = target_community * user_per_community
    community_users = np.arange(start_idx, end_idx)
    return community_users
