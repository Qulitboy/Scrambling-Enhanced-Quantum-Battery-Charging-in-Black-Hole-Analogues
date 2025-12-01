import numpy as np
import matplotlib as mpl
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
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 15,
    'figure.dpi': 300,
    'figure.autolayout': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.25,
    'lines.linewidth': 1.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4
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
tm = 250/Lout   # Total evolution time

dt = tm / Nt  # Time per step
d = 0.01
xhmax=5
xhmin=0
NL=20
dm=int(np.ceil(np.sqrt(NL)))
P_op = np.zeros([NL,NL])
t_op = np.zeros([NL,NL])

xh_value= np.linspace(xhmin,xhmax,NL)
for p_idx,xh0 in enumerate(xh_value):
    for q_idx,xht in enumerate(xh_value):
        # 创建哈密顿量并确保维度一致
        H0 = Creat_H(xh0)
        H1 = Creat_H(xht)
        # 计算本征值并排序
        a1, V0 = np.linalg.eig(H0)
        sorted_indices = np.argsort(a1.real)
        a1 = a1[sorted_indices]
        energy_band=np.real(np.max(a1)-np.min(a1))
        V0 = V0[:, sorted_indices]
        # 选择第三个本征态（原MATLAB的第三列）
        psi_0 = V0[:, 0].reshape(-1, 1)
        U = expm(-1j * H1 * dt)
        L = H0.shape[0]  # 矩阵维度
        P_op[p_idx,q_idx],t_op[p_idx,q_idx]=Opt_P(H0)
#====================variance sigma_E(t)====================


plt.figure(figsize=(6, 5))
plt.pcolormesh(xh_value, xh_value, P_op, shading='auto' , edgecolor='face')  # 添加shading参数确保正确渲染

# 设置科研论文常用字体 (Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

plt.xlabel(r'$x_{ht}$', fontsize=30)
plt.ylabel(r'$x_{h0}$', fontsize=30)

# 创建colorbar并设置刻度
cbar = plt.colorbar(label=r'$P_{max}$')
cbar.locator = ticker.MaxNLocator(nbins=4)  # 设置5个主要刻度
cbar.formatter = ticker.FormatStrFormatter('%.2f')  # 强制显示2位小数
cbar.update_ticks()

# 设置刻度标签字体
cbar.ax.tick_params(labelsize=25)
cbar.ax.set_ylabel(r'$P_{max}$', fontsize=30, fontname='Times New Roman')

# plt.title(r'$P_{max}$', fontsize=12, fontname='Times New Roman')
plt.xticks(fontname='Times New Roman', fontsize=25)
plt.yticks(fontname='Times New Roman', fontsize=25)
# plt.savefig('fig4a.pdf', dpi=300)
plt.show()

# 第二个图同理修改
plt.figure(figsize=(6, 5))
plt.pcolormesh(xh_value, xh_value, t_op, shading='auto', edgecolor='face')

plt.xlabel(r'$x_{ht}$', fontsize=30)
plt.ylabel(r'$x_{h0}$', fontsize=30)

cbar = plt.colorbar(label=r'$t_*$')
cbar.locator = ticker.MaxNLocator(nbins=4)  # 设置4个刻度
cbar.update_ticks()

cbar.ax.tick_params(labelsize=25)
cbar.ax.set_ylabel(r'$\tau_*$', fontsize=30, fontname='Times New Roman')

# plt.title(r'$t_*$', fontsize=12, fontname='Times New Roman')
plt.xticks(fontname='Times New Roman', fontsize=25)
plt.yticks(fontname='Times New Roman', fontsize=25)
# plt.savefig('fig4b.pdf', dpi=300)
plt.show()
#====================================extralplot
#=========================end

# # 创建网格
# x = np.arange(t_op.shape[1])
# y = np.arange(t_op.shape[0])
# x, y = np.meshgrid(x, y)

# # 
# fig = plt.figure(figsize=(8, 6))  # 设置图形大小
# ax = fig.add_subplot(111, projection='3d')
# # 绘制三维表面图
# p2 = ax.plot_surface(x, y, t_op, cmap='viridis', edgecolor='none', rstride=1, cstride=1, alpha=0.8)

# # 设置坐标轴标签
# ax.set_xlabel(r'$x_{ht}$', fontsize=12, labelpad=10)  # 使用LaTeX格式，增加字体大小和标签间距
# ax.set_ylabel(r'$x_{h0}$', fontsize=12, labelpad=10)
# ax.set_zlabel(r'$t_{\mathrm{op}}$', fontsize=12, labelpad=10)

# # 设置坐标轴刻度字体大小
# ax.tick_params(axis='both', which='major', labelsize=10)

# # 添加颜色条
# cbar = fig.colorbar(p2, pad=0.15, shrink=0.6)
# cbar.set_label(r'$t_{\mathrm{op}}$', fontsize=12, rotation=270, labelpad=15)

# # 设置图形标题
# ax.set_title('3D Surface Plot of t_op', fontsize=14, pad=20)

# # 优化图形布局
# plt.tight_layout()

# # 显示图形
# plt.show()