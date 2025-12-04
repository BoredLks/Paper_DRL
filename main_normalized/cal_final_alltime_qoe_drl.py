"""QoE calculation with an on-policy DRL environment (aligned to MATLAB DRL variant)."""

from __future__ import annotations

import os
import random
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .calculate_qoe import calculate_qoe
try:
    from .debug_utils import SimulationLogger
except ImportError:
    class SimulationLogger:
        def __init__(self, *args, **kwargs): pass
        def log_step(self, *args, **kwargs): pass
        def plot_and_save(self, *args, **kwargs): pass

EpisodeSample = Tuple[np.ndarray, np.ndarray, np.ndarray]

BUFFER_SIZE = 20000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 5e-5
TARGET_UPDATE = 500
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HISTORY_LEN = 4

class QNetwork(nn.Module):
    """Shared feed-forward Q-network with Action Masking support."""
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x, mask=None):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        if mask is not None:
            # 极小的负数，保证 argmax 绝不会选中无效动作
            q_values = q_values.masked_fill(mask == 0, -1e9)
        return q_values

class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer = []
        self.capacity = buffer_size
    def push(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.device = DEVICE
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.t_step = 0

    def step(self, state, mask, action, reward, next_state, next_mask, done):
        self.memory.push((state, mask, action, reward, next_state, next_mask, done))
        self.t_step += 1
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample(BATCH_SIZE)
            self.learn(experiences)
        if self.t_step % TARGET_UPDATE == 0:
            self.update_target_network()

    def act(self, state: np.ndarray, mask: np.ndarray, eps: float = 0.1) -> int:
        if random.random() > eps:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state_tensor, mask_tensor)
            self.qnetwork_local.train()
            return int(torch.argmax(action_values, dim=1).item())
        else:
            # 随机探索时，必须严格遵守 mask
            valid_indices = np.where(mask == 1)[0]
            if len(valid_indices) > 0:
                return int(random.choice(valid_indices))
            else:
                return 0 # 保底

    def learn(self, experiences):
        states, masks, actions, rewards, next_states, next_masks, dones = zip(*experiences)
        states_t = torch.from_numpy(np.vstack(states)).float().to(self.device)
        masks_t = torch.from_numpy(np.vstack(masks)).bool().to(self.device) # Use mask in training? Not strictly needed for Local but good for consistency
        actions_t = torch.from_numpy(np.array(actions, dtype=np.int64).reshape(-1, 1)).to(self.device)
        rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32).reshape(-1, 1)).to(self.device)
        next_states_t = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        next_masks_t = torch.from_numpy(np.vstack(next_masks)).bool().to(self.device)
        dones_t = torch.from_numpy(np.array(dones, dtype=np.uint8).reshape(-1, 1)).float().to(self.device)

        # Target Network: Double DQN logic or Standard DQN with Mask
        # 使用 next_mask 确保 target max q 不会选到下一时刻也不合法的动作
        q_targets_next = self.qnetwork_target(next_states_t, next_masks_t).detach().max(1, keepdim=True)[0]
        q_targets = rewards_t + (GAMMA * q_targets_next * (1 - dones_t))
        
        q_expected = self.qnetwork_local(states_t).gather(1, actions_t)
        
        loss = nn.MSELoss()(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), GRAD_CLIP)
        self.optimizer.step()

    def update_target_network(self):
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.qnetwork_local.state_dict(), path)
    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.qnetwork_local.load_state_dict(state)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

def cal_final_alltime_qoe_drl(
    nf: int, chi, delta_t: float, alpha_qoe: float, beta_qoe: float, gamma_qoe: float, delta_qoe: float,
    D_bf: float, community_users, requested_videos, cache_decision, iu_flags, p_sbs: float, p_iu: float,
    iu_count_per_community: int, episode_sampler: Callable[[], EpisodeSample],
    eval_sample: EpisodeSample | None = None, training_episodes: int = 200,
    model_path: str | None = None, train_model: bool = True, log_interval: int = 10,
) -> tuple[float, list[float]]:

    chi = np.array(chi, dtype=float)
    user_count = len(community_users)
    chi_levels = len(chi)
    chi_max = chi[-1] if chi_levels > 0 else 1.0
    cache_hits = np.zeros(user_count, dtype=bool)
    for idx, user_idx in enumerate(community_users):
        cache_hits[idx] = iu_flags[user_idx] == 1 and cache_decision[user_idx, requested_videos[idx]] == 1

    state_size = 7 + HISTORY_LEN
    agent = DQNAgent(state_size, chi_levels)

    eps_start, eps_end = 1.0, 0.05
    eps_decay = (eps_start - eps_end) / max(1, training_episodes * 0.8)
    eps_value = eps_start

    # ... (Helper functions: _resolution_index, _normalize_rate, _capacity_ratio, _assignment_norm 保持不变) ...
    def _resolution_index(value: float) -> float:
        idx = np.argmin(np.abs(chi - value))
        return idx / max(1, chi_levels - 1) if chi_levels > 1 else 0.0
    def _normalize_rate(rate: float) -> float:
        return rate / chi_max if chi_max > 0 else 0.0
    def _capacity_ratio(remaining: float, limit: float) -> float:
        if limit <= 0: return 0.0
        return max(0.0, min(1.0, remaining / limit))
    def _assignment_norm(value: float) -> float:
        if value <= 0: return 0.0
        return (value + 1) / (iu_count_per_community + 1)

    def _build_state(buffer_val, rate_history_queue, prev_res_val, assignment_val, capacity_val, cache_flag, wait_val, wait_norm_base):
        wait_norm = wait_val / wait_norm_base if wait_norm_base > 0 else 0.0
        base_features = [
            buffer_val / D_bf if D_bf > 0 else 0.0,
            _normalize_rate(rate_history_queue[-1]), 
            _resolution_index(prev_res_val),
            _assignment_norm(assignment_val),
            max(0.0, min(1.0, capacity_val)),
            1.0 if cache_flag else 0.0,
            wait_norm,
        ]
        history_features = [_normalize_rate(r) for r in rate_history_queue[:-1]]
        return np.array(base_features + history_features, dtype=np.float32)

    def _generate_mask(available_capacity):
        mask = np.zeros(chi_levels, dtype=np.float32)
        valid_indices = np.where(chi <= available_capacity + 1e-9)[0]
        if valid_indices.size > 0:
            mask[valid_indices] = 1.0
        else:
            mask[0] = 1.0 # 无论如何保底给个最低档
        return mask

    def _apply_capacity_limit(current_res, download_rate, available_capacity):
        # 仅供 Greedy 使用：Greedy 依然受限于速率
        greedy_allowed = min(download_rate, available_capacity)
        valid_indices = np.where(chi <= greedy_allowed + 1e-9)[0]
        if valid_indices.size == 0: return chi[0]
        return chi[valid_indices[-1]] # Greedy 总是选最大的

    def _run_episode(sample: EpisodeSample, eps: float, training: bool, 
                     do_logging: bool = False, episode_idx: int = 0) -> tuple[float, float]:
        
        download_rates, task_assignment, initial_wait = sample
        buffer_state = np.zeros((user_count, nf), dtype=float)
        buffer_state[:, 0] = D_bf
        r_previous = np.full((user_count, nf), chi[0], dtype=float)
        r_decision = np.zeros((user_count, nf), dtype=float)

        qoe_total = 0.0
        greedy_qoe_total = 0.0
        wait_norm_base = max(1.0, float(np.max(initial_wait) if initial_wait.size > 0 else 1.0))

        if do_logging:
            logger = SimulationLogger(nf, user_count, p_sbs, p_iu, iu_count_per_community)

        rate_histories = np.zeros((user_count, HISTORY_LEN + 1))
        for u in range(user_count):
            rate_histories[u, :] = download_rates[u, 0]

        for t in range(nf):
            if t > 0:
                r_previous[:, t] = r_decision[:, t - 1]
                rate_histories[:, :-1] = rate_histories[:, 1:]
                rate_histories[:, -1] = download_rates[:, t]

            # 每一秒开始，算力重置为满血 (除了被 Greedy 占用的部分，这里假设 Greedy 和 RL 是平行世界)
            remaining_sbs = float(p_sbs)
            remaining_iu = np.full(iu_count_per_community, float(p_iu))
            
            # Greedy 也要重置
            greedy_rem_sbs = float(p_sbs)
            greedy_rem_iu = np.full(iu_count_per_community, float(p_iu))
            
            # 日志统计
            current_sbs_load = 0.0
            current_iu_loads = {k: 0.0 for k in range(iu_count_per_community)}

            for i in range(user_count):
                assignment_val = task_assignment[i, t]
                is_sbs = assignment_val <= 0
                
                # 1. 计算当前(t时刻)剩余算力
                current_limit = remaining_sbs if is_sbs else remaining_iu[min(iu_count_per_community - 1, int(round(assignment_val)))]
                # Mask 基于当前剩余算力生成 -> 保证物理约束
                current_mask = _generate_mask(current_limit)

                wait_time = initial_wait[i] if t == 0 else 0.0
                state = _build_state(
                    buffer_state[i, t], rate_histories[i], r_previous[i, t], assignment_val,
                    _capacity_ratio(current_limit, p_sbs if is_sbs else p_iu),
                    cache_hits[i], wait_time, wait_norm_base
                )

                # 2. 动作选择
                action_idx = agent.act(state, current_mask, eps if training else 0.0)
                raw_resolution = chi[action_idx]
                
                # # 3. 双重保险：虽然 Mask 应该保证了，但我们再强制截断一次，确保绝对不超标
                # # 注意：这里只截断算力，不截断网速！
                # valid_capacity_indices = np.where(chi <= current_limit + 1e-9)[0]
                # if valid_capacity_indices.size > 0:
                #     # 如果 raw_resolution 超过了 current_limit，就强制降级
                #     if raw_resolution > current_limit + 1e-9:
                #         # 这理论上不应发生（如果Mask工作正常），但为了稳健性：
                #         safe_resolution = chi[valid_capacity_indices[-1]]
                #     else:
                #         safe_resolution = raw_resolution
                # else:
                #     safe_resolution = chi[0] # 保底

                # 4. 计算 QoE
                step_qoe = calculate_qoe(
                    buffer_state[i, t], raw_resolution, r_previous[i, t], wait_time, download_rates[i, t],
                    alpha_qoe, beta_qoe, gamma_qoe, delta_qoe, nf, delta_t, t + 1, chi, initial_wait
                )
                qoe_total += step_qoe

                # 5. 更新环境 (Buffer)
                # 允许 safe_resolution > download_rate，导致 Buffer 减少
                next_buffer = buffer_state[i, t] + (download_rates[i, t] - raw_resolution) * delta_t
                next_buffer = float(np.clip(next_buffer, 0, D_bf)) # 物理上 buffer 不能 < 0
                if t < nf - 1:
                    buffer_state[i, t + 1] = next_buffer
                r_decision[i, t] = raw_resolution

                # 6. 奖励计算 (加入防卡顿惩罚！)
                reward_for_agent = step_qoe / 100.0
                
                # 逻辑修正：
                # 只有当 (初始Buffer + 下载量) < 播放量 时，才是真卡顿。
                # 或者简单点：看 calculate_qoe 返回的 EI (完整性) 是否 < 1.0

                # 重新计算一次 EI 用于判断是否卡顿 (稍微有点耗时但准确)
                # 或者直接用 next_buffer 判断，但要放宽条件：
                # 如果 next_buffer 接近 0，且 Buffer 在下降 (rate < resolution)，才罚。

                # 最推荐的修改：
                # 既然 next_buffer = 0 且 rate >= resolution 是允许的（边下边播），
                # 那我们就只惩罚 [Buffer耗尽] 且 [网速不够] 的情况。

                # 方案：
                is_rebuffering = (buffer_state[i, t] + download_rates[i, t] * delta_t) < (raw_resolution * delta_t)

                if is_rebuffering:
                    reward_for_agent -= 200.0
                # else: 不罚！哪怕 next_buffer 是 0，只要没卡顿就不罚。

                # 7. 扣除算力
                if is_sbs:
                    remaining_sbs = max(0.0, remaining_sbs - raw_resolution)
                    current_sbs_load += raw_resolution
                else:
                    iu_idx = min(iu_count_per_community - 1, int(round(assignment_val)))
                    remaining_iu[iu_idx] = max(0.0, remaining_iu[iu_idx] - raw_resolution)
                    current_iu_loads[iu_idx] += raw_resolution

                # 8. 生成 Next State
                # 关键修正：Next State 的算力特征应该是【下一时刻的满血算力】，而不是当前时刻剩下的残渣！
                # 近似：假设下一时刻大家还没开始抢，算力是满的 (Ratio = 1.0)
                # 这样 Agent 就知道：只要熬过这一秒，下一秒我就又有资源了
                next_full_capacity = p_sbs if is_sbs else p_iu
                next_mask = _generate_mask(next_full_capacity)
                
                next_state = _build_state(
                    next_buffer, rate_histories[i], raw_resolution, assignment_val,
                    1.0, # Next capacity ratio approx 1.0 (reset)
                    cache_hits[i], 0.0, wait_norm_base
                )
                
                done = t == nf - 1 and i == user_count - 1
                if training:
                    agent.step(state, current_mask, action_idx, reward_for_agent, next_state, next_mask, done)

                # --- Greedy 逻辑 ---
                limit_greedy = greedy_rem_sbs if is_sbs else greedy_rem_iu[min(iu_count_per_community - 1, int(round(assignment_val)))]
                greedy_res = _apply_capacity_limit(chi[-1], download_rates[i, t], limit_greedy)
                greedy_step_qoe = calculate_qoe(
                    buffer_state[i, t], greedy_res, r_previous[i, t], wait_time, download_rates[i, t],
                    alpha_qoe, beta_qoe, gamma_qoe, delta_qoe, nf, delta_t, t + 1, chi, initial_wait
                )
                greedy_qoe_total += greedy_step_qoe
                if is_sbs:
                    greedy_rem_sbs = max(0.0, greedy_rem_sbs - greedy_res)
                else:
                    iu_idx = min(iu_count_per_community - 1, int(round(assignment_val)))
                    greedy_rem_iu[iu_idx] = max(0.0, greedy_rem_iu[iu_idx] - greedy_res)

            if do_logging:
                logger.log_step(t, buffer_state[:, t], r_decision[:, t], download_rates[:, t], 
                                task_assignment[:, t], current_sbs_load, current_iu_loads)

        if do_logging:
            logger.plot_and_save(episode_idx)

        return qoe_total, greedy_qoe_total

    training_history = []
    if train_model:
        for ep in range(1, training_episodes + 1):
            episode_sample = episode_sampler()
            should_log = (ep % 100 == 0) or (ep == 1)
            episode_reward, greedy_reward = _run_episode(
                episode_sample, eps_value, training=True, do_logging=should_log, episode_idx=ep
            )
            training_history.append(episode_reward)
            eps_value = max(eps_end, eps_value - eps_decay)
            
            if log_interval > 0 and (ep % log_interval == 0 or ep == training_episodes):
                recent = training_history[-log_interval:]
                print(f"[DRL] Ep {ep}/{training_episodes} | Eps: {eps_value:.2f} | QoE: {episode_reward:.1f} (Avg: {np.mean(recent):.1f}) | Greedy: {greedy_reward:.1f}")
        
        if model_path: agent.save(model_path)
    else:
        if model_path is None or not os.path.exists(model_path): raise FileNotFoundError("Model not found.")
        agent.load(model_path)

    eval_sample = eval_sample or episode_sampler()
    final_qoe, _ = _run_episode(eval_sample, eps=0.0, training=False, do_logging=True, episode_idx=9999)
    return final_qoe, training_history