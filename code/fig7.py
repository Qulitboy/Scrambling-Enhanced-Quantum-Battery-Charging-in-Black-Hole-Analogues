import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import scipy.io
import time
import random
import matplotlib.ticker as ticker
#=============
"""最佳充电时间和最佳充电功率"""
#================
# Load custom colormap
# newCmap1 = scipy.io.loadmat('myColormap.mat')['newCmap1']
# Pauli matrices (保留但未使用，可能后续扩展需要)
# sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
# sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
# sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

plt.rcParams.update({
    'font.family': 'serif',        # 主字体
    'font.serif': ['Times New Roman'],  # Times字体族
    'font.size': 18,               # 基础字号
    'axes.labelsize': 20,          # 坐标轴标签字号
    'axes.titlesize': 22,          # 标题字号
    'xtick.labelsize': 20,         # X轴刻度字号
    'ytick.labelsize': 20,         # Y轴刻度字号
    'legend.fontsize': 20,         # 图例字号
    'legend.frameon': True,        # 图例边框
    'legend.framealpha': 0.8,      # 图例透明度
    'legend.edgecolor': 'black',   # 图例边框颜色
    'grid.alpha': 0.15,            # 网格透明度
    'grid.linestyle': '--',        # 网格线型
    'figure.dpi': 300,             # 输出分辨率
    'savefig.dpi': 600,            # 保存分辨率
    'axes.linewidth': 0.8,         # 坐标轴线宽
    'lines.linewidth': 2,        # 数据线宽
    'xtick.direction': 'in',       # 刻度朝向
    'ytick.direction': 'in',       # 刻度朝向
    'xtick.major.size': 4,         # 刻度长度
    'ytick.major.size': 4          # 刻度长度
})


def Creat_H(xht):
    Lh = int(xht / d)  
    Ht = np.zeros((Lout, Lout), dtype=complex)
    for n in range(Lout - 1):
        x1 = (Lh+n + 0.5) * d  
        Ht[n, n+1] = -x1**2 * (1 - xht/x1) / (4 * d)
        # if x1 >= xh1:  # 只在x1 >= xh1时设置元素
        #     Ht[n, n+1] = -x1**2 * (1 - xh1/x1) / (4 * d) 
    Ht = Ht + Ht.T  # 对称化
    # Ht += beta1      # 添加对角元素
    return Ht


#================RDM==================

#=================end=================
def Opt_P(H0):
    pst = psi_0.copy()
    P0 = 0  # 初始化为0
    topt = 0
    e0 = (np.conj(psi_0).T @ H0 @ psi_0).real
    for i in range(1, Nt+1):
        pst = U @ pst
        # pst /= np.linalg.norm(pst)  # 若U是酉矩阵可注释
        dE = (np.conj(pst).T @ H0 @ pst).real - e0
        Pt = dE / (i * dt)/energy_band
        if Pt > P0:
            P0 = Pt
            topt = i * dt
    return P0, topt  # 返回最大值而非最后的值



# Start timer
start_time = time.time()
Nt = 500
Lout=250
tm = 2000/Lout   # Total evolution time

dt = tm / Nt  # Time per step
d = 0.01
xhmax=5
xhmin=0.1
NL=20
xh0=0
dm=int(np.ceil(np.sqrt(NL)))
P_op = np.zeros([NL])
t_op = np.zeros([NL])
regu=np.zeros([NL])
xh_value= np.linspace(xhmin,xhmax,NL)
H0 = Creat_H(xh0)
# 计算本征值并排序
a1, V0 = np.linalg.eig(H0)
sorted_indices = np.argsort(a1.real)
a1 = a1[sorted_indices]
energy_band=np.real(np.max(a1)-np.min(a1))
V0 = V0[:, sorted_indices]
# 选择第三个本征态（原MATLAB的第三列）
psi_0 = V0[:, 0].reshape(-1, 1)
for q_idx,xht in enumerate(xh_value):
    # 创建哈密顿量并确保维度一致
   
    H1 = Creat_H(xht)
    regu[q_idx]=np.linalg.norm(H1, 2)
    H1 /= np.linalg.norm(H1, 2)
    
    U = expm(-1j * H1 * dt)
    L = H0.shape[0]  # 矩阵维度
    P_op[q_idx],t_op[q_idx]=Opt_P(H0)
#====================variance sigma_E(t)====================
a=xh_value
b=regu
c = [a_i / b_i if b_i != 0 else float('nan') for a_i, b_i in zip(a, b)]


# fig, ax = plt.subplots(figsize=(5.5, 4.5))  # 黄金比例尺寸
# line_p = ax.plot(c, P_op, 
#                 color='#1f77b4',  # 科学蓝
#                 linestyle='-', 
#                 marker='o',
#                 markersize=4,
#                 markerfacecolor='white',
#                 markeredgewidth=1.2,
#                 label=r'$P_{\mathrm{opt}}$')

# line_t = ax.plot(c, t_op, 
#                 color='#d62728',  # 科学红
#                 linestyle='--', 
#                 marker='s',
#                 markersize=4,
#                 markerfacecolor='white',
#                 markeredgewidth=1.2,
#                 label=r'$t_{\mathrm{op}}$')
# # ======================
# ax.set_xlabel(r'$x_{ht}$', labelpad=8)
# ax.set_ylabel('Value', labelpad=10)  # 根据实际含义修改ylabel
# ax.yaxis.set_major_locator(ticker.MaxNLocator(6))  # 最多5个刻度
# ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))  # 每个主刻
# # 科学计数法自动格式化
# ax.ticklabel_format(axis='both', style='sci', scilimits=(-2,3), useMathText=True)

# # 网格线（期刊常用浅灰色虚线）
# ax.grid(True, linestyle='--', alpha=0.3)

# # ======================
# # 图例和标题优化
# # ======================
# legend = ax.legend(loc='best', frameon=True, fancybox=False)
# legend.get_frame().set_linewidth(0.7)

# # 可选：添加学术标题（多数期刊不鼓励图表标题）
# # ax.set_title('Optimization Parameters', pad=15, fontsize=12)

# # ======================
# # 布局和输出
# # ======================
# plt.tight_layout(pad=2.0)

# # 显示图形
# plt.show()


# ======================
# 科研级绘图参数全局设置
# ======================
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman'],
#     'font.size': 11,
#     'axes.titlesize': 13,
#     'axes.labelsize': 12,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'legend.fontsize': 10,
#     'legend.frameon': True,
#     'legend.framealpha': 0.85,
#     'legend.edgecolor': 'black',
#     'grid.alpha': 0.15,
#     'figure.dpi': 300,
#     'savefig.dpi': 600,
#     'axes.linewidth': 0.8,
#     'lines.linewidth': 1.8  # 加粗线条提高可读性
# })

# ======================
# 创建双轴系统
# ======================
# fig, ax1 = plt.subplots(figsize=(6, 5))  # 稍宽以适应双轴标签

# # 创建共享X轴的第二Y轴
# ax2 = ax1.twinx()

# # ======================
# # 数据绘制 - 双色系统
# # ======================
# # 科学配色方案 (Nature期刊风格)
# color_p = '#1f77b4'  # 左轴蓝色
# color_t = '#d62728'  # 右轴红色

# # 左轴数据 (P_opt)
# line_p = ax1.plot(c, P_op, 
#                  color=color_p,
#                  linestyle='-',
#                  marker='o',
#                  markersize=5,
#                  markerfacecolor='white',
#                  markeredgewidth=1.2,
#                  label=r'$P_{\mathrm{max}}$')

# # 右轴数据 (t_op)
# line_t = ax2.plot(c, t_op, 
#                  color=color_t,
#                  linestyle='--',
#                  marker='s',
#                  markersize=5,
#                  markerfacecolor='white',
#                  markeredgewidth=1.2,
#                  label=r'$t_{*}$')

# line_tm = ax2.plot(np.linspace(0,0.006,len(c)),[tm for _ in range(len(c))], 
#                     color='y',
#                     linestyle='-',
#                     linewidth=1,
#                     label=r'$t_{*}$')
# # ======================
# # 坐标轴系统美化
# # ======================
# # X轴设置
# ax1.set_xlabel(r'$x_{ht}$', fontsize=12, labelpad=10)
# ax1.tick_params(axis='x', which='both', direction='in', top=False)
# ax1.xaxis.set_major_locator(ticker.MaxNLocator(6))  # 4个刻度
# # ax1.xaxis.set_major_locator(ticker.AutoMinorLocator(2))  # 4个刻度
# # 左Y轴设置 (P_opt)
# ax1.set_ylabel(r'$P_{\mathrm{max}}$', 
#               fontsize=12, 
#               color=color_p,
#               labelpad=12)
# ax1.tick_params(axis='y', labelcolor=color_p, direction='in')
# ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))  # 5个刻度

# # 右Y轴设置 (t_op)
# ax2.set_ylabel(r'$t_{*}$', 
#               fontsize=12, 
#               color=color_t,
#               labelpad=12)
# ax2.set_ylim([0,1.5*tm])
# ax2.tick_params(axis='y', labelcolor=color_t, direction='in')
# ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))  # 4个刻度

# # 科学计数法格式化
# ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,3), useMathText=True)
# ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,3), useMathText=True)

# # ======================
# # 图例整合与位置优化
# # ======================
# # 合并双轴图例
# lines = line_p + line_t
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper center', 
#           bbox_to_anchor=(0.5, 1.15),
#           ncol=2, frameon=True, fancybox=False)

# # ======================
# # 网格与布局优化
# # ======================
# # 仅左轴显示网格（避免视觉混乱）
# ax1.grid(True, linestyle='--', alpha=0.2)

# # 调整坐标轴位置避免重叠
# ax1.spines['left'].set_position(('outward', 8))
# ax2.spines['right'].set_position(('outward', 8))

# # 设置轴线颜色匹配曲线
# ax1.spines['left'].set_color(color_p)
# ax2.spines['right'].set_color(color_t)

# # 最终布局调整
# plt.tight_layout()
# fig.subplots_adjust(top=0.85)  # 为图例留出空间

# # 显示图形
# plt.show()
fig, ax1 = plt.subplots(figsize=(7, 6))  # 稍宽以适应双轴标签

# 创建共享X轴的第二Y轴
ax2 = ax1.twinx()

# ======================
# 数据绘制 - 双色系统
# ======================
# 科学配色方案 (Nature期刊风格)
color_p = '#1f77b4'  # 左轴蓝色
color_t = '#d62728'  # 右轴红色

# 左轴数据 (P_opt)
line_p = ax1.plot(c, P_op, 
                 color=color_p,
                 linestyle='-',
                 marker='o',
                 markersize=5,
                 markerfacecolor='white',
                 markeredgewidth=1.2,
                 label=r'$P_{\mathrm{max}}$')

# 右轴数据 (t_op)
line_t = ax2.plot(c, t_op, 
                 color=color_t,
                 linestyle='--',
                 marker='s',
                 markersize=5,
                 markerfacecolor='white',
                 markeredgewidth=1.2,
                 label=r'$\tau_{*}$')

line_tm = ax2.plot(np.linspace(0,0.006,len(c)),[tm for _ in range(len(c))], 
                    color='y',
                    linestyle='-',
                    linewidth=1,
                    label=r'$\tau_{*}$')
# ======================
# 坐标轴系统美化
# ======================
# X轴设置
ax1.set_xlabel(r'$x_{ht}$', fontsize=20, labelpad=10)
ax1.tick_params(axis='x', which='both', direction='in', top=False)
ax1.xaxis.set_major_locator(ticker.MaxNLocator(4))  # 4个刻度
# ax1.xaxis.set_major_locator(ticker.AutoMinorLocator(2))  # 4个刻度
# 左Y轴设置 (P_opt)
ax1.set_ylabel(r'$P_{\mathrm{max}}$', 
              fontsize=20, 
              color=color_p,
              labelpad=12)
ax1.tick_params(axis='y', labelcolor=color_p, direction='in')
ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))  # 5个刻度

# 右Y轴设置 (t_op)
ax2.set_ylabel(r'$\tau_{*}$', 
              fontsize=20, 
              color=color_t,
              labelpad=12)
ax2.set_ylim([0,1.5*tm])
ax2.tick_params(axis='y', labelcolor=color_t, direction='in')
ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))  # 4个刻度

# 科学计数法格式化
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,3), useMathText=True)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,3), useMathText=True)

# ======================
# 图例整合与位置优化
# ======================
# 合并双轴图例
lines = line_p + line_t
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', 
          bbox_to_anchor=(0.5, 1.18),
          ncol=2, frameon=True, fancybox=False)

# ======================
# 网格与布局优化
# ======================
# 仅左轴显示网格（避免视觉混乱）
ax1.grid(True, linestyle='--', alpha=0.2)

# 调整坐标轴位置避免重叠
ax1.spines['left'].set_position(('outward', 8))
ax2.spines['right'].set_position(('outward', 8))

# 设置轴线颜色匹配曲线
ax1.spines['left'].set_color(color_p)
ax2.spines['right'].set_color(color_t)

# 最终布局调整
plt.tight_layout()
fig.subplots_adjust(top=0.85)  # 为图例留出空间
fig.savefig('fig7.pdf', bbox_inches='tight')
# 显示图形
plt.show()

# 保存出版级图片
# fig.savefig('dual_axis_plot.pdf', bbox_inches='tight')
# fig.savefig('dual_axis_plot.png', dpi=600, bbox_inches='tight')