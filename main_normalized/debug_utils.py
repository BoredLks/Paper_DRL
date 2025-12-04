"""
debug_utils.py: Visualization tools for DRL verification.
用于记录和绘制蒙特卡洛仿真中的微观数据（Buffer、Rate、Load）。
"""

import os
import numpy as np
import matplotlib.pyplot as plt

class SimulationLogger:
    """
    用于记录单次仿真（Episode）中所有关键状态的日志工具。
    """
    def __init__(self, nf, user_count, p_sbs, p_iu, iu_count):
        self.nf = nf
        self.time_steps = np.arange(nf)
        self.p_sbs = p_sbs
        self.p_iu = p_iu
        self.user_count = user_count
        self.iu_count = iu_count
        
        # 用户数据容器
        self.user_data = {u: {
            "buffer": [],
            "resolution": [],
            "download_rate": [],
            "assignment": []
        } for u in range(user_count)}
        
        # 节点负载容器
        self.sbs_load = []
        self.iu_loads = {i: [] for i in range(iu_count)}

    def log_step(self, t, buffer_states, decisions, download_rates, assignments, 
                 current_sbs_load, current_iu_loads):
        """在每个时间片结束时调用，记录当前帧的数据"""
        # 记录每个用户的状态
        for u in range(self.user_count):
            self.user_data[u]["buffer"].append(buffer_states[u])
            self.user_data[u]["resolution"].append(decisions[u])
            self.user_data[u]["download_rate"].append(download_rates[u])
            self.user_data[u]["assignment"].append(assignments[u])
            
        # 记录节点总负载
        self.sbs_load.append(current_sbs_load)
        for i in range(self.iu_count):
            # 如果某个IU这一秒没活干，负载就是0
            load = current_iu_loads.get(i, 0.0)
            self.iu_loads[i].append(load)

    def plot_and_save(self, episode_idx, save_dir="Debug_Figures"):
        """生成并保存三张核心约束验证图"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        # 为了清晰，我们只画第 0 个用户作为代表
        # 如果你想看随机用户，可以改为 target_user = np.random.randint(self.user_count)
        target_user = 0
        
        fig = plt.figure(figsize=(12, 12))
        
        # --- 子图 1: Buffer 状态 (防卡顿验证) ---
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(self.time_steps, self.user_data[target_user]["buffer"], 
                 color='blue', linewidth=2, label="Buffer Level (Mb)")
        ax1.axhline(0, color='red', linestyle='--', linewidth=2, label="Rebuffering Limit (0)")
        ax1.set_ylabel("Buffer Size (Mb)")
        ax1.set_title(f"Ep {episode_idx} - User {target_user} Buffer Status")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # --- 子图 2: 码率 vs 下载速率 (透支策略验证) ---
        ax2 = fig.add_subplot(3, 1, 2)
        res = np.array(self.user_data[target_user]["resolution"])
        rate = np.array(self.user_data[target_user]["download_rate"])
        
        ax2.plot(self.time_steps, res, color='green', marker='.', label="Selected Resolution (Action)")
        ax2.plot(self.time_steps, rate, color='orange', linestyle='--', label="Download Rate (Constraint)")
        
        # 红色区域：透支 Buffer (Res > Rate) - 这是 DRL 应该学会的高级策略
        ax2.fill_between(self.time_steps, res, rate, where=(res > rate), 
                         color='red', alpha=0.2, interpolate=True, label="Draining Buffer (Res > Rate)")
        # 绿色区域：积累 Buffer (Res < Rate)
        ax2.fill_between(self.time_steps, res, rate, where=(res <= rate), 
                         color='green', alpha=0.1, interpolate=True, label="Filling Buffer (Res < Rate)")
        
        ax2.set_ylabel("Bitrate (Mbps)")
        ax2.set_title(f"Ep {episode_idx} - Resolution Selection vs Bandwidth")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        # --- 子图 3: 节点计算负载 (硬约束验证) ---
        ax3 = fig.add_subplot(3, 1, 3)
        # 画 SBS 负载
        ax3.plot(self.time_steps, self.sbs_load, color='purple', linewidth=2, label="SBS Load")
        ax3.axhline(self.p_sbs, color='purple', linestyle='--', alpha=0.5, label=f"SBS Max ({self.p_sbs})")
        
        # 画负载最高的 IU（避免线条太多）
        # 计算每个 IU 在这一轮的总负载
        iu_total_loads = {i: sum(self.iu_loads[i]) for i in range(self.iu_count)}
        busy_iu_idx = max(iu_total_loads, key=iu_total_loads.get)
        
        ax3.plot(self.time_steps, self.iu_loads[busy_iu_idx], 
                 color='brown', alpha=0.7, linestyle='-.', label=f"Busiest IU (Index {busy_iu_idx}) Load")
        ax3.axhline(self.p_iu, color='brown', linestyle=':', alpha=0.5, label=f"IU Max ({self.p_iu})")
        
        ax3.set_ylabel("Compute Load (Mbps)")
        ax3.set_xlabel("Time Step (t)")
        ax3.set_title(f"Ep {episode_idx} - Edge Computing Constraints (Action Masking Check)")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = os.path.join(save_dir, f"verification_ep_{episode_idx}.png")
        plt.savefig(filename)
        plt.close()
        print(f"  --> [Debug] Constraint verification plot saved to {filename}")