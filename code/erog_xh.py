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
# 设置Nature期刊绘图风格
plt.style.use('default')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['figure.titlesize'] = 10
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['figure.figsize'] = (3.54, 2.65)  # Nature单栏图尺寸 (89mm)
mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['xtick.major.width'] = 0.8
mpl.rcParams['ytick.major.width'] = 0.8
mpl.rcParams['xtick.minor.width'] = 0.4
mpl.rcParams['ytick.minor.width'] = 0.4
mpl.rcParams['xtick.major.size'] = 3.5
mpl.rcParams['ytick.major.size'] = 3.5
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.minor.size'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.05
def Creat_H(xht,Lout,d):
    Lh = int(xht / d)  
    Ht = np.zeros((Lout, Lout), dtype=complex)
    for n in range(Lout - 1):
        x1 = (Lh+n + 0.5) * d  
        Ht[n, n+1] = -x1**2 * (1 - xht/x1) / (4 * d)

    Ht = Ht + Ht.T  # 对称化
    # Ht += beta1      # 添加对角元素
    return Ht

def fermion_chain_ergotropy(rho, H,a1):
    """
    计算费米子链的可提取功
    """
    
    
    # 对角化密度矩阵
    rho_eigvals,_ = np.linalg.eigh(rho)
    # 对角化哈密顿量 a1


    
    # 按概率降序排列密度矩阵本征态
    idx_rho = np.argsort(rho_eigvals)[::-1]
    eigvals_rho = rho_eigvals[idx_rho]

    
    # 计算ergotropy
    initial_energy = np.trace(rho @ H)

    final_energy = np.sum(eigvals_rho * a1)
    
    return (initial_energy - final_energy)/energy_band
    # return (initial_energy - final_energy)

def erogmax(psi_in,H0,Nt):
    erogload=np.zeros([Nt,1])
    for i in range(Nt):
        rho_in=np.outer(psi_in,psi_in.T.conj())
        erogload[i]=fermion_chain_ergotropy(rho_in, H0,a1)
        psi_in=U @ psi_in
    return np.max(erogload)
# Start timer
start_time = time.time()
Nt = 200     # Number of time steps
Lout =250
xhmax=5
xhmin=0
NL=10
d = 0.01
erog=np.zeros([NL,NL])
xh_value= np.linspace(xhmin,xhmax,NL)
for p_idx,xh0 in enumerate(xh_value):
    for q_idx,xht in enumerate(xh_value):

        H0 = Creat_H(xh0,Lout,d)
        H1 = Creat_H(xht,Lout,d)
        tm =np.abs( Lout/np.min(H0)/2)  # Total evolution time
        dt = tm / Nt  # Time per step
        # 计算本征值并排序
        a1, V0 = np.linalg.eigh(H0)
        sorted_indices = np.argsort(a1.real)
        a1 = a1[sorted_indices]
        energy_band=np.max(a1)-np.min(a1)
        V0 = V0[:, sorted_indices]
        # 选择第三个本征态（原MATLAB的第三列）
        psi_0 = V0[:, 0].reshape(-1, 1)
        U = expm(-1j * H1 * dt)
        erog[p_idx,q_idx]=erogmax(psi_0,H0,Nt)





# 获取矩阵的行和列
rows, cols = erog.shape

# 创建网格
x = np.linspace(0,xhmax,cols)
y = np.linspace(0,xhmax,rows)
x, y = np.meshgrid(x, y)

# 将矩阵展平为一维数组
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = erog.flatten()



## %%
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
# ax.set_zticks(np.arange(0, z_flat.max(), 1))
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
fig.text(0.95, 0.5, r'$\mathcal{E}_{\mathrm{max}}$', rotation=90, 
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
print('程序运行时间:%s秒' % ((time.time()-start_time)))
plt.show()