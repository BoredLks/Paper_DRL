"""Movielens-based preference computation translated from MATLAB cal_Movielens.m."""

from __future__ import annotations

import functools
import os
import numpy as np

# 数据集默认路径：与 MATLAB 脚本一致，使用当前目录下的 ml-100k/u.data
DEFAULT_ML_PATH = os.path.join(os.path.dirname(__file__), "ml-100k", "u.data")


def _build_matrix(data: np.ndarray):
    """构造评分矩阵和余弦相似度矩阵，兼容直接传入 data 或文件读取。"""
    user_ids = data[:, 0].astype(int)
    movie_ids = data[:, 1].astype(int)
    ratings = data[:, 2]

    num_users = int(user_ids.max())
    num_movies = int(movie_ids.max())

    R = np.zeros((num_users, num_movies))
    for u, m, r in zip(user_ids, movie_ids, ratings):
        # MATLAB 数据是 1-index，这里转为 0-index
        R[u - 1, m - 1] = r

    denom = np.sqrt(np.sum(R**2, axis=1, keepdims=True))
    denom[denom == 0] = 1
    R_norm = R / denom
    sim_matrix = R_norm @ R_norm.T
    return R, sim_matrix


@functools.lru_cache(maxsize=1)
def _load_movielens_matrix(file_path: str):
    """读取 Movielens 数据集，构造用�?影片评分矩阵与余弦相似度。"""
    data = np.loadtxt(file_path)
    return _build_matrix(data)


def cal_Movielens(total_users, F, gamma_m, data: np.ndarray | None = None, file_path: str | None = None):
    """
    使用 Movielens 数据计算兴趣相似度与用户请求概率。

    参数 data 可直接传入已读取的评分表；否则使用 file_path 或默认路径加载。
    """
    if data is not None:
        R, sim_matrix = _build_matrix(np.asarray(data))
    else:
        path = file_path or DEFAULT_ML_PATH
        R, sim_matrix = _load_movielens_matrix(path)

    num_users, num_movies = R.shape
    user_file_req_prob = np.zeros((total_users, F))

    # 因为仿真用户数量�?Movielens 不一致，这里循环映射用户/文件 ID
    user_mapping = np.mod(np.arange(total_users), num_users)
    file_mapping = np.mod(np.arange(F), num_movies)

    all_ratings = R[R > 0]
    global_mean = np.mean(all_ratings) if all_ratings.size > 0 else 2.5

    for u in range(total_users):
        movielens_user_id = user_mapping[u]
        user_ratings = R[movielens_user_id, :]

        file_scores = np.zeros(F)
        for k in range(F):
            movielens_movie_id = file_mapping[k]
            rating = user_ratings[movielens_movie_id]
            if rating > 0:
                file_scores[k] = rating  # 用户对该电影有真实评�?
            else:
                movie_ratings = R[:, movielens_movie_id]
                movie_ratings = movie_ratings[movie_ratings > 0]
                if movie_ratings.size > 0:
                    file_scores[k] = np.mean(movie_ratings)
                else:
                    file_scores[k] = global_mean
                file_scores[k] += 0.1 * np.random.randn()  # 加一点扰动避免全部相�?

        sorted_indices = np.argsort(file_scores)[::-1]
        zipf_denominator = np.sum((np.arange(1, F + 1)) ** (-gamma_m))

        rank_positions = np.empty(F, dtype=int)
        rank_positions[sorted_indices] = np.arange(1, F + 1)

        # 将排名映射到 Zipf 分布，得到用户对每个文件的请求概�?
        user_file_req_prob[u, :] = (rank_positions**(-gamma_m)) / zipf_denominator

    return sim_matrix, user_file_req_prob
