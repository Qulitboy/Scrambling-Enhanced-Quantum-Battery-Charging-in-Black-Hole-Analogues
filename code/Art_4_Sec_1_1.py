import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, eigh
import time
from scipy.optimize import curve_fit
import matplotlib as mpl

# 定义优雅的配色方案
COLOR_THEORY = '#2a5a8c'  # 深蓝色 - 理论线
COLOR_DATA = '#0de378'    # 亮绿色 - 数据点
COLOR_ERROR = '#d62728'   # 红色 - 误差棒
COLOR_BACKGROUND = '#f8f9fa'  # 浅灰色背景

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

# 定义指数拟合函数用于斜率计算
def exp_fit(t, a, b):
    return a * t + b

start_time = time.time()

# 参数设置
d = 0.005
Lout = 251
dt = 0.02
beta = 1.0
beta1 = beta * np.eye(Lout)
xh_values = np.arange(0.1, 5.2, 0.9)
lmdfit = np.zeros(len(xh_values))
lmdfit_err = np.zeros(len(xh_values))  # 存储拟合误差

# 预计算常数
precomputed = {}

for idx, xh1 in enumerate(xh_values):
    # 计算参数
    Lh = int(xh1 / d)
    L0 = xh1 / 15.0
    n0 = (xh1 + 2 * L0) / d
    Th = xh1 / (4 * np.pi)
    lamda = xh1 / 2.0
    Tm = (12 + 4 * np.log(xh1)) / xh1
    Nt = int(Tm / dt)
    Tn = np.linspace(dt, Tm, Nt)
    Co_all = np.zeros(Nt)
    
    # 使用缓存避免重复计算
    key = (Lh, Lout)
    if key not in precomputed:
        # 哈密顿量构造
        Hout = np.zeros((Lout, Lout))
        n_range = np.arange(Lout)
        x1 = (Lh + n_range + 0.5) * d
        diag_vals = np.zeros(Lout)
        
        # 填充非对角元素
        for n in range(Lout - 1):
            Hout[n, n+1] = -x1[n]**2 * (1 - xh1/x1[n]) / (4 * d)
            diag_vals[n] = (d / L0) * np.exp(-(d**2) * ((Lh + n + 1 - n0)**2) / L0**2)
        
        # 填充对角元素
        diag_vals[-1] = (d / L0) * np.exp(-(d**2) * (Lh + Lout - n0)**2 / L0**2)
        Hout = Hout + Hout.T + np.diag(beta * np.ones(Lout))
        
        # 基态计算
        Omega, Vn0 = eigh(Hout)  # 使用更快的eigh代替eig
        precomputed[key] = (Hout, diag_vals, Omega, Vn0)
    else:
        Hout, diag_vals, Omega, Vn0 = precomputed[key]
    
    # 筛选特征值
    ap = Omega >= beta
    Eout = Omega[ap]
    Vn1 = Vn0[:, ap]
    
    # 计算密度矩阵
    rhox = np.zeros((Lout, Lout))
    for i in range(len(Eout)):
        v = Vn1[:, i]
        rhox += np.exp(-Eout[i]/Th) * np.outer(v, v)
    
    rhox = rhox / np.trace(rhox)
    rho = rhox
    
    # 时间演化算符
    U0 = expm(-1j * Hout * dt)
    U1 = expm(1j * Hout * dt)
    N_n0 = np.diag(diag_vals)
    N_nv = N_n0.copy()
    
    # 时间演化循环 - 向量化部分计算
    for n in range(Nt):
        N_nv = U0 @ N_nv @ U1
        Co_all[n] = -np.real(np.trace(rho @ (N_nv @ N_n0 - N_n0 @ N_nv)@ ( N_n0 @ N_nv  -   N_nv @ N_n0 )))
        # Co_all[n] = np.abs(-np.trace(rho @ (N_nv @ N_n0 - N_n0 @ N_nv)))
    # 使用曲线拟合计算斜率及误差
    y_data = np.log(Co_all / Co_all[0])
    try:
        popt, pcov = curve_fit(exp_fit, Tn, y_data, p0=[1.0, 0.0])
        lmdfit[idx] = popt[0]
        lmdfit_err[idx] = np.sqrt(pcov[0, 0])  # 斜率的误差估计
    except RuntimeError:
        lmdfit[idx] = np.polyfit(Tn, y_data, 1)[0]
        lmdfit_err[idx] = 0.0

# 最终拟合
k, b = np.polyfit(xh_values, lmdfit, 1)
k_err = np.std(lmdfit - (k * xh_values + b)) / np.sqrt(len(xh_values))  # 斜率误差估计

# 计算理论值
theoretical = xh_values 

# 计算点到理论线的绝对偏差
absolute_deviation = np.abs(lmdfit - theoretical)

# 计算相对偏差百分比
relative_deviation = 100 * absolute_deviation / theoretical

# 创建图表
fig, ax = plt.subplots(figsize=(3.54, 2.65))



# 绘制理论线（优雅深蓝色虚线）
ax.plot(xh_values, theoretical, '--', color=COLOR_THEORY, 
        linewidth=2.0, alpha=0.9, 
        label=r'Theoretical: $\lambda_fit = x_h$',
        dashes=(5, 2))  # 5点实线+2点空白

# 绘制数据点带误差棒（表示到理论线的偏差）
ax.scatter(xh_values, lmdfit, color='red',marker='x',
            label='Numerical data', zorder=10)

# 添加连接线（亮绿色细线）
ax.plot(xh_values, lmdfit, '-', color=COLOR_DATA, 
        linewidth=1.2, alpha=0.7, zorder=5)

# 设置标题和标签
ax.set_xlabel(r'$x_h$', fontsize=10, labelpad=2)
ax.set_ylabel(r'$\lambda$', fontsize=10, labelpad=2)
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)

# 设置图例
legend = ax.legend(loc='best', frameon=True, fancybox=True, 
          framealpha=0.9, edgecolor='#cccccc')
legend.get_frame().set_linewidth(0.5)

# 添加网格
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5, color='#dddddd')

# 添加理论线方程
ax.text(0.95, 0.15, r'$\lambda_{fit} = x_h$', 
        transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.8, pad=0.3, edgecolor='none'))



# 添加最终斜率信息
fit_text = f"Slope: {k:.3f} ± {k_err:.3f}"
ax.text(0.95, 0.35, fit_text, transform=ax.transAxes,
        fontsize=8, ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.8, pad=0.3, edgecolor='none'))

# 优化布局
plt.tight_layout(pad=0.5)

# 保存图像
# plt.savefig('otoc_fit_analysis.png', dpi=600)

# 显示图表
plt.show()

end_time = time.time()
print(f"Optimized time elapsed: {end_time - start_time:.2f} seconds")