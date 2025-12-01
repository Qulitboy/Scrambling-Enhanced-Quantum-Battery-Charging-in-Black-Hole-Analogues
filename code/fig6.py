import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time

# 人民币配色和科学标记
# RMB_COLORS = ['#2d1c4d', '#684e94', '#b186a3', '#e7ddb8']   
RMB_COLORS = ['#0077BE', '#C2185B', '#FFC107', '#424242']
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 28,
    # 'axes.titlesize': 25,
    # 'axes.labelsize': 25,
    # 'xtick.labelsize': 23,
    # 'ytick.labelsize': 23,
    # 'legend.fontsize': 25,
    'figure.dpi': 300,
    # 'figure.autolayout': False,
    # 'axes.spines.top': False,
    # 'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.25,
    'lines.linewidth': 1.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4
})

SCI_MARKERS = [
    {'marker': 'o', 'size': 8},
    {'marker': '*', 'size': 10},
    {'marker': 'D', 'size': 8},
    {'marker': '^', 'size': 8}
]

def create_hamiltonian(xh, L):
    """创建哈密顿量矩阵"""
    d = 0.01  # 空间步长
    H = np.zeros((L, L), dtype=complex)
    
    for n in range(L - 1):
        x = (n + 0.5) * d  # 格点位置
        # 添加安全措施防止除以零
        coupling = -x**2 * (1 - xh/(x + 1e-10)) / (4 * d) if x > 1e-6 else 0
        H[n, n+1] = coupling
    
    H = H + H.T  # 对称化哈密顿量
    np.fill_diagonal(H, 0.1)  # 添加对角元素防止奇异
    return H

def optimize_power(H0, psi0, U, Nt, dt, energy_band):
    """优化功率计算"""
    psi_t = psi0.copy()
    max_power = 0
    opt_time = 0
    E0 = (psi0.conj().T @ H0 @ psi0).real
    
    for step in range(1, Nt + 1):
        psi_t = U @ psi_t  # 时间演化
        E_t = (psi_t.conj().T @ H0 @ psi_t).real
        power = (E_t - E0) / (step * dt) / energy_band
        
        if power > max_power:
            max_power = power
            opt_time = step * dt
            
    return max_power, opt_time

# ============ 主程序 =============
start_time = time.time()
Nt = 500  # 时间步数
d_val = 0.01  # 空间步长
xh_min, xh_max = 0.1, 5  # 淬火参数范围（避免0）
xh_values = [0.5, 1.5,2.5, 3.0]  # 淬火参数（选择4个代表性值）
system_sizes = np.linspace(300, 500, 10, dtype=int)  # 系统尺寸从100到300

# 存储结果
power_results = np.zeros((len(xh_values), len(system_sizes)))
time_results = np.zeros((len(xh_values), len(system_sizes)))

# 固定初始淬火位置
xh0 = 0.0
total_time = 20.0  # 固定总演化时间
dt = total_time / Nt  # 时间步长

# 主循环：外层淬火参数，内层系统尺寸
for xh_idx, xh_quench in enumerate(xh_values):
    print(f"Processing quench parameter: xh={xh_quench:.1f}")
    
    for size_idx, L in enumerate(system_sizes):
        # 创建哈密顿量
        H0 = create_hamiltonian(xh0, L)
        H1 = create_hamiltonian(xh_quench, L)
        
        # 计算基态
        eigvals, eigvecs = np.linalg.eigh(H0)
        energy_band = np.max(eigvals) - np.min(eigvals)
        psi0 = eigvecs[:, 0].reshape(-1, 1)  # 基态
        
        # 时间演化算子
        U = expm(-1j * H1 * dt)
        
        # 优化计算
        power, opt_time = optimize_power(H0, psi0, U, Nt, dt, energy_band)
        power_results[xh_idx, size_idx] = power
        time_results[xh_idx, size_idx] = opt_time

# ============ 结果可视化 =============
# plt.figure(figsize=(8, 10))

# # 最佳充电功率 vs 系统尺寸
# plt.subplot(2, 1, 1)
# for idx, xh in enumerate(xh_values):
#     plt.plot(system_sizes, power_results[idx, :], 
#              label=f'$x_h$={xh:.1f}',
#              marker=SCI_MARKERS[idx]['marker'],
#              markersize=SCI_MARKERS[idx]['size'],
#              color=RMB_COLORS[idx],
#              linewidth=2)

# plt.xlabel('System Size $L$', fontsize=20)
# plt.ylabel('$P_{max}}$', fontsize=20)
# plt.title('Maximum Charging Power vs System Size', fontsize=25)
# plt.grid(alpha=0.3)
# plt.legend(title='Quench Parameter', fontsize=10)

# # 最佳充电时间 vs 系统尺寸
# plt.subplot(2, 1, 2)
# for idx, xh in enumerate(xh_values):
#     plt.plot(system_sizes, time_results[idx, :], 
#              label=f'$x_h$={xh:.1f}',
#              marker=SCI_MARKERS[idx]['marker'],
#              markersize=SCI_MARKERS[idx]['size'],
#              color=RMB_COLORS[idx],
#              linewidth=2,
#              linestyle='--')

# plt.xlabel('System Size $L$', fontsize=12)
# plt.ylabel(r'$\tau_{*}$', fontsize=12)
# plt.title('Optimal Charging Time vs System Size', fontsize=14)
# plt.grid(alpha=0.3)
# plt.legend(title='Quench Parameter', fontsize=10)

# plt.tight_layout()

#====================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), sharex=True)

# 第一个子图：P_max vs L
for idx, xh in enumerate(xh_values):
    ax1.plot(
        system_sizes, 
        power_results[idx, :], 
        label=f'$x_h$={xh:.1f}',
        marker=SCI_MARKERS[idx]['marker'],
        markersize=SCI_MARKERS[idx]['size'],
        color=RMB_COLORS[idx],
        linewidth=2
    )

ax1.set_ylabel(r'$P_{max}$', fontsize=30)
ax1.set_title('Maximum Charging Power', fontsize=30)
ax1.grid(alpha=0.3)
ax1.legend(title='Quench Parameter', fontsize=30)

# 第二个子图：τ* vs L
for idx, xh in enumerate(xh_values):
    ax2.plot(
        system_sizes, 
        time_results[idx, :], 
        label=f'$x_h$={xh:.1f}',
        marker=SCI_MARKERS[idx]['marker'],
        markersize=SCI_MARKERS[idx]['size'],
        color=RMB_COLORS[idx],
        linewidth=2,
        linestyle='--'
    )

ax2.set_xlabel('System Size $L$', fontsize=30)  # 仅底部显示x轴标签
ax2.set_ylabel(r'$\tau_{*}$', fontsize=30)
ax2.set_title('Optimal Charging Time', fontsize=30)
ax2.grid(alpha=0.3)
ax2.legend(title='Quench Parameter', fontsize=30)

# 调整子图间距
plt.tight_layout()


# plt.savefig('fig6.pdf', dpi=300)

print(f"Total execution time: {time.time()-start_time:.2f} seconds")
plt.show()