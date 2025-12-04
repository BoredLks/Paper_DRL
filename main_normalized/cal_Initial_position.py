"""Initial position generator translated from MATLAB cal_Initial_position.m."""

from __future__ import annotations

import numpy as np


def cal_Initial_position(
    h,
    user_per_community,
    total_users,
    region_size,
    community_radius,
    max_movement_dist,
    T_small,
):
    sbs_positions = np.zeros((h, 2))
    min_distance = 2 * community_radius
    max_attempts = 1000

    # 放置第一个 SBS
    valid_position = False
    attempts = 0
    while not valid_position and attempts < max_attempts:
        attempts += 1
        candidate_pos = np.array(
            [
                np.random.rand() * (region_size - 2 * community_radius) + community_radius,
                np.random.rand() * (region_size - 2 * community_radius) + community_radius,
            ]
        )
        if (
            community_radius <= candidate_pos[0] <= region_size - community_radius
            and community_radius <= candidate_pos[1] <= region_size - community_radius
        ):
            sbs_positions[0, :] = candidate_pos
            valid_position = True
    if not valid_position:
        raise RuntimeError("无法为SBS 1找到有效位置")

    # 放置剩余的 SBS
    for i in range(1, h):
        valid_position = False
        attempts = 0
        while not valid_position and attempts < max_attempts:
            attempts += 1
            candidate_pos = np.array(
                [
                    np.random.rand() * (region_size - 2 * community_radius) + community_radius,
                    np.random.rand() * (region_size - 2 * community_radius) + community_radius,
                ]
            )
            if (
                community_radius <= candidate_pos[0] <= region_size - community_radius
                and community_radius <= candidate_pos[1] <= region_size - community_radius
            ):
                distances = np.linalg.norm(sbs_positions[:i, :] - candidate_pos, axis=1)
                if np.all(distances > min_distance):
                    sbs_positions[i, :] = candidate_pos
                    valid_position = True
        if not valid_position:
            raise RuntimeError(f"无法为SBS {i+1}找到有效位置")

    user_positions = np.zeros((total_users, 2, T_small * 2))
    user_community = np.zeros(total_users, dtype=int)

    for m in range(h):
        start_idx = m * user_per_community
        end_idx = start_idx + user_per_community
        user_community[start_idx:end_idx] = m + 1  # 保持社区标签为 1 基

        for u in range(start_idx, end_idx):
            for t in range(T_small * 2):
                if t == 0:
                    valid_position = False
                    while not valid_position:
                        r = np.sqrt(np.random.rand()) * community_radius
                        theta = 2 * np.pi * np.random.rand()
                        x = sbs_positions[m, 0] + r * np.cos(theta)
                        y = sbs_positions[m, 1] + r * np.sin(theta)
                        if 0 <= x <= region_size and 0 <= y <= region_size:
                            user_positions[u, :, t] = [x, y]
                            valid_position = True
                else:
                    valid_position = False
                    attempts = 0
                    while not valid_position and attempts < 100:
                        attempts += 1
                        prev_position = user_positions[u, :, t - 1]
                        move_distance = np.random.rand() * max_movement_dist
                        move_angle = 2 * np.pi * np.random.rand()
                        new_x = prev_position[0] + move_distance * np.cos(move_angle)
                        new_y = prev_position[1] + move_distance * np.sin(move_angle)
                        dist_to_sbs = np.sqrt(
                            (new_x - sbs_positions[m, 0]) ** 2
                            + (new_y - sbs_positions[m, 1]) ** 2
                        )
                        if (
                            0 <= new_x <= region_size
                            and 0 <= new_y <= region_size
                            and dist_to_sbs <= community_radius
                        ):
                            user_positions[u, :, t] = [new_x, new_y]
                            valid_position = True

                    if not valid_position:
                        prev_position = user_positions[u, :, t - 1]
                        small_move = 10
                        to_center = sbs_positions[m, :] - prev_position
                        to_center_normalized = to_center / np.linalg.norm(to_center)
                        new_position = prev_position + small_move * to_center_normalized
                        new_position = new_position + np.random.randn(2) * 5
                        new_position[0] = max(0, min(region_size, new_position[0]))
                        new_position[1] = max(0, min(region_size, new_position[1]))
                        actual_move_dist = np.linalg.norm(new_position - prev_position)
                        if actual_move_dist > max_movement_dist:
                            direction = (new_position - prev_position) / actual_move_dist
                            new_position = prev_position + direction * max_movement_dist
                        user_positions[u, :, t] = new_position

    per_user_positions = user_positions[:, :, :T_small]
    now_user_positions = user_positions[:, :, T_small:]
    return sbs_positions, per_user_positions, now_user_positions, user_community
