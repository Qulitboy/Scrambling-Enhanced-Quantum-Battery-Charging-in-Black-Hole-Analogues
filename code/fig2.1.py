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
mpl.rcParams['font.size'] = 15
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['figure.titlesize'] = 18
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
xh_values = np.arange(0.8, 5.1, 0.6)
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

# 计算R²值 (添加在绘图代码前)
residuals = lmdfit - (k*xh_values + b)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((lmdfit - np.mean(lmdfit))**2)
r_squared = 1 - (ss_res / ss_tot)
# 绘图部分代码
# =============================================================================
# 创建双面板图表 (主图和偏差分析)
fig = plt.figure(figsize=(7, 5.5))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
ax_main = fig.add_subplot(gs[0])
ax_dev = fig.add_subplot(gs[1], sharex=ax_main)

# 主图：理论值 vs 计算值
# =============================================================================
# 绘制理论线
ax_main.plot(xh_values, theoretical, color=COLOR_THEORY, linewidth=2, 
             label=r'Theoretical $\lambda = x_h$', zorder=1)

# 绘制计算值带误差棒
ax_main.errorbar(xh_values, lmdfit, yerr=lmdfit_err, fmt='o', 
                 color=COLOR_DATA, ecolor=COLOR_ERROR, elinewidth=1.5,
                 capsize=4, capthick=1.5, markersize=6, markeredgewidth=1,
                 label='Numerical fit', zorder=3)

# 绘制线性拟合线
ax_main.plot(xh_values, k*xh_values + b, '--', color='#9c27b0', 
             linewidth=1.8, alpha=0.9, 
             label=f'Linear fit: $k={k:.3f} \\pm {k_err:.3f}$', zorder=2)

# 设置主图标题和标签
ax_main.set_title('Lyapunov Exponent in Gravitational Analogue System', fontsize=18, pad=12)
ax_main.set_ylabel(r'$\lambda$', fontsize=15, labelpad=8)
ax_main.grid(True, linestyle=':', alpha=0.3, linewidth=0.8)

# 添加图例
ax_main.legend(loc='upper left', frameon=True, framealpha=0.9)

# 添加拟合信息文本框
fit_text = (f'$R^2$ = {r_squared:.4f}\n'
            f'Slope = {k:.3f} $\\pm$ {k_err:.3f}\n'
            f'Intercept = {b:.3f}')
ax_main.text(0.95, 0.05, fit_text, transform=ax_main.transAxes,
             ha='right', va='bottom', fontsize=15,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 偏差分析图
# =============================================================================
# 计算相对偏差误差
rel_error = 100 * (lmdfit_err / theoretical)

# 绘制相对偏差
ax_dev.errorbar(xh_values, relative_deviation, yerr=rel_error, fmt='s',
                color='#ff7f0e', ecolor='#d62728', capsize=3, 
                markersize=5, alpha=0.8, markeredgewidth=0.5)

# 绘制零偏差参考线
ax_dev.axhline(y=0, color='#2c3e50', linestyle='--', linewidth=1, alpha=0.7)

# 设置偏差图标签
ax_dev.set_ylabel('Deviation (%)', fontsize=15, labelpad=8)
ax_dev.set_xlabel(r'System parameter $x_h$', fontsize=15, labelpad=8)
ax_dev.grid(True, linestyle=':', alpha=0.3, linewidth=0.8)

# 设置坐标轴范围
ax_main.set_xlim(min(xh_values)-0.1, max(xh_values)+0.1)
ax_main.set_ylim(min(theoretical)-0.5, max(theoretical)+0.5)
ax_dev.set_ylim(-np.max(relative_deviation)*1.8, np.max(relative_deviation)*1.8)

# 隐藏主图的x轴刻度标签
plt.setp(ax_main.get_xticklabels(), visible=False)

# 添加整体标题
# fig.suptitle('OTOC Lyapunov Exponent Analysis', fontsize=14, y=0.98)

# 添加网格背景
for ax in [ax_main, ax_dev]:
    ax.set_facecolor(COLOR_BACKGROUND)
    ax.grid(True, linestyle=':', alpha=0.4)


# 保存图片
# plt.savefig('lyapunov_exponent_analysis.png', dpi=350, bbox_inches='tight')

plt.show()

end_time = time.time()
print(f"Optimized time elapsed: {end_time - start_time:.2f} seconds")