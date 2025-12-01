import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm
# import scipy.io
import time
import random
#=============
"""绘制xh0--xht--sgm"""
#=================
# Load custom colormap
# newCmap1 = scipy.io.loadmat('myColormap.mat')['newCmap1']
# 设置Nature风格参数
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

    Ht = Ht + Ht.T  # 对称化
    # Ht += beta1      # 添加对角元素
    return Ht

#==================================
def sgm_E(H0):
    pst = psi_0.copy()  # 避免修改原始态
    dE = np.zeros(Nt)   # 一维数组存储结果
    e0 = np.conj(psi_0).T @ H0 @ psi_0  # 初始能量期望
    for i in range(Nt):
        dE[i] = (np.conj(pst).T @ H0 @ pst).real - e0.real  # 取实部避免复数误差
        pst = U @ pst  # 演化到下一时间步
        # opt_t
        # pst /= np.linalg.norm(pst)
        # 可选：定期归一化态矢量（数值稳定性）
    # dE = dE/energy_band*2
    dE = dE/energy_band
    var1=np.var(dE[100:])
    E_max=max(dE)
    return var1,E_max


# Start timer
start_time = time.time()
Nt = 500     # Number of time steps
Lout =250
xhmax=5
xhmin=0
NL=10
dm=int(np.ceil(np.sqrt(NL)))
sgme = np.zeros([NL,NL])
Emax = np.zeros([NL,NL])

xh_value= np.linspace(xhmin,xhmax,NL)
for p_idx,xh0 in enumerate(xh_value):
    for q_idx,xht in enumerate(xh_value):
        tm = 500/Lout   # Total evolution time

        dt = tm / Nt  # Time per step
        d = 0.01

        # 创建哈密顿量并确保维度一致
        H0 = Creat_H(xh0)
        # H0 /= np.linalg.norm(H0, 2)
        H1 = Creat_H(xht)
        # H1 /= np.linalg.norm(H1, 2)
        # 计算本征值并排序
        a1, V0 = np.linalg.eig(H0)
        sorted_indices = np.argsort(a1.real)
        a1 = a1[sorted_indices]
        energy_band=np.max(a1)-np.min(a1)
        V0 = V0[:, sorted_indices]
        # 选择第三个本征态（原MATLAB的第三列）
        psi_0 = V0[:, 0].reshape(-1, 1)
        U = expm(-1j * H1 * dt)
        L = H0.shape[0]  # 矩阵维度
        sgme[p_idx,q_idx],Emax[p_idx,q_idx]=sgm_E(H0)
#====================variance sigma_E(t)====================


# plt.figure(figsize=(10, 10))
# # 使用 imshow 绘制，设置 interpolation 为 'bilinear' 来实现平滑过渡
# # plt.pcolor(sgme)
# plt.pcolormesh(xh_value,xh_value,sgme)
# plt.xlabel('xht')
# plt.ylabel('xh0')
# plt.colorbar(label='sgm')
# # 反转 y 轴
# # plt.gca().invert_yaxis()
# plt.title('sgm')
# plt.show()

# plt.figure(figsize=(10, 10))
# # 使用 imshow 绘制，设置 interpolation 为 'bilinear' 来实现平滑过渡
# # plt.pcolor(Emax)
# plt.pcolormesh(xh_value,xh_value,Emax)
# plt.xlabel('xht')
# plt.ylabel('xh0')
# plt.colorbar(label='EMAX')
# # 反转 y 轴
# # plt.gca().invert_yaxis()
# plt.title('Emax')
# plt.show()
#============================================dipict================
# 假设 data1 是一个二维 NumPy 数组


# # 创建网格
# x = np.linspace(0,xhmax,Emax.shape[1])
# y = np.linspace(0,xhmax,Emax.shape[0])
# x, y = np.meshgrid(x, y)

# # 
# fig = plt.figure(figsize=(8, 6))  # 设置图形大小
# ax = fig.add_subplot(111, projection='3d')
# # 绘制三维表面图
# p2 = ax.plot_surface(x, y, Emax, cmap='viridis', edgecolor='none', rstride=1, cstride=1, alpha=0.8)

# # 设置坐标轴标签
# ax.set_xlabel(r'$x_{h0}$', fontsize=12, labelpad=10)  # 使用LaTeX格式，增加字体大小和标签间距
# ax.set_ylabel(r'$x_{ht}$', fontsize=12, labelpad=10)
# ax.set_zlabel(r'$E_{\mathrm{max}}$', fontsize=12, labelpad=10)

# # 设置坐标轴刻度字体大小
# ax.tick_params(axis='both', which='major', labelsize=10)

# # 添加颜色条
# cbar = fig.colorbar(p2, pad=0.15, shrink=0.6)
# cbar.set_label(r'$E_{\mathrm{max}}$', fontsize=12, rotation=270, labelpad=15)

# # 设置图形标题
# ax.set_title('3D Surface Plot of Emax', fontsize=14, pad=20)

# # 优化图形布局
# plt.tight_layout()

# # 显示图形
# plt.show()

#================end=====================================
#=============extral


# 获取矩阵的行和列
rows, cols = Emax.shape

# 创建网格
x = np.linspace(0,xhmax,cols)
y = np.linspace(0,xhmax,rows)
x, y = np.meshgrid(x, y)

# 将矩阵展平为一维数组
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = Emax.flatten()


# 创建图形
fig = plt.figure(figsize=(6.5, 6))  # Nature推荐尺寸
ax = fig.add_subplot(111, projection='3d')

# 使用Viridis颜色映射（更科学）
norm = mpl.colors.Normalize(vmin=z_flat.min(), vmax=z_flat.max())
colors = plt.cm.Spectral(norm(z_flat))

# 绘制柱状图（减小alpha值，增加边缘线宽）
p3 = ax.bar3d(x_flat, y_flat, np.zeros_like(z_flat),
              dx=0.2, dy=0.2, dz=z_flat,
              color=colors, edgecolor='black', linewidth=0.3, alpha=0.7)

# 设置坐标轴标签（使用LaTeX格式）
ax.set_xlabel(r'$x_{h0}$', labelpad=12, fontsize=20)
ax.set_ylabel(r'$x_{ht}$', labelpad=12, fontsize=20)

# 设置刻度（优化显示）
ax.set_xticks(np.arange(0, xhmax+1, step=1))
ax.set_yticks(np.arange(0, xhmax+1, step=1))
ax.set_zticks(np.arange(0, z_flat.max(), 0.001))
# 科学计数法显示z轴
# ax.ticklabel_format(axis='z', style='sci', scilimits=(-3,3))
# ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax.zaxis.offsetText.set_fontsize(10)

# 设置z轴范围
z_max = z_flat.max() * 1.05  # 增加5%空间
ax.set_zlim(0, z_max)

# 添加网格
ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "alpha":0.3})
ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "alpha":0.3})
ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "alpha":0.3})

# 调整视角
ax.view_init(elev=35, azim=-55)

# 设置刻度标签
ax.tick_params(axis='x', labelsize=15, pad=8)
ax.tick_params(axis='y', labelsize=15, pad=8)
ax.tick_params(axis='z', labelsize=15, pad=10)

# 移除z轴标签（将在外部添加）
ax.set_zlabel('')

# 添加外部z轴标签
fig.text(0.95, 0.5, r'$E_{\mathrm{max}}$', rotation=90, 
         va='center', ha='center', fontsize=20)
# # 添加颜色条
# cax = fig.add_axes([0.95, 0.3, 0.02, 0.3])  # [x, y, width, height]
# sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral, norm=norm)
# sm.set_array([])
# cbar = fig.colorbar(sm, cax=cax)
# cbar.set_label(r'$E_{\mathrm{max}}$ ', rotation=270, labelpad=15, fontsize=20)
# cbar.ax.tick_params(labelsize=9)


# 优化布局
plt.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.95)

# 保存高质量图片（符合Nature要求）
# plt.savefig('fig3.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
# plt.savefig('3d_plot.tif', dpi=300, bbox_inches='tight', pad_inches=0.05, format='tiff')

plt.show()