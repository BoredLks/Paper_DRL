"""Requested videos generator translated from MATLAB cal_requested_videos.m."""

from __future__ import annotations

import numpy as np


def cal_requested_videos(user_file_req_prob, community_users):
    """根据每个用户的请求概率分布随机抽取本轮所需的视频 ID"""
    requested_videos = np.zeros(len(community_users), dtype=int)
    for i, user_idx in enumerate(community_users):
        cumulative_prob = np.cumsum(user_file_req_prob[user_idx, :])
        rand_val = np.random.rand()
        requested_videos[i] = int(np.searchsorted(cumulative_prob, rand_val))
    return requested_videos
