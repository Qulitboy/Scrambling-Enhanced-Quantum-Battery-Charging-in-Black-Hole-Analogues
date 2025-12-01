import numpy as np
import matplotlib.pyplot as plt
import time
#================蝴蝶效应，算符和xht无关==================
RMB_COLORS = ['#2d1c4d', #5元
             '#684e94',  
             '#b186a3',
             '#e7ddb8']   
# RMB_COLORS = ['#8d4b45', #20元
#              '#dba880',  
#              '#f0c986',
#              '#f1e0dc'#第四个比较浅
#              ]   
MARKERS = ['o', 's', 'D', '^']  # 圆形、方形、菱形、三角形
plt.rcParams.update({
    'font.family': 'Times New Roman',
    # 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 15,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 15,
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

# 科研级标记配置1
SCI_MARKERS = [
    {'marker': 'o', 'drawstyle': '-', 'size': 30},  # 实心圆
    {'marker': '*', 'drawstyle': '--', 'size': 30},  # star
    {'marker': 'D', 'drawstyle': '-.', 'size': 30},  # 实心菱形
    {'marker': '^', 'drawstyle': ':', 'size': 30}   # 空心三角
]

def Creat_H(xht, d, Lout):
    Lh = int(xht / d)
    n = np.arange(Lout - 1)
    x1 = (Lh + n + 0.5) * d
    upper_diag = -x1**2 * (1 - xht / x1) / (4 * d)
    Ht = np.zeros((Lout, Lout), dtype=complex)
    rows, cols = n, n + 1
    Ht[rows, cols] = upper_diag
    Ht += Ht.T  # Symmetrize the matrix
    return Ht

def oper_w(Lout, xht, d):
    Lh = int(xht / d)
    # L0 = xht / 15.0
    # n0 = (xht + 2 * L0) / d
    L0=10
    n0=20
    n = np.arange(Lout)
    terms = Lh + n + 1 - n0
    Nn0 = (d / L0) * np.exp(-(d**2) * terms**2 / L0**2)
    return Nn0.astype(complex)  # Ensure complex type

def comu(a, b):
    return np.dot(a, b) - np.dot(b, a)

def butt_w(opw, H, order):
    for _ in range(order):
        opw = comu(opw, H)
    return np.linalg.norm(opw)  # Frobenius norm as scalar



# Parameters
d = 0.01
xh0=1
Lout = 251
xhvalues = np.linspace(0, 1, 10)
odr_values = list(range(3, 7))
bt_fly = np.zeros((len(odr_values), len(xhvalues)))
colors=['']
start_time = time.time()
plt.figure(figsize=(8,6)) 
for idx, order in enumerate(odr_values):
    for jdx, xht in enumerate(xhvalues):
        H = Creat_H(xht, d, Lout)
        op_diag = oper_w(Lout, xh0, d)
        op = np.diag(op_diag)  # Convert to diagonal matrix
        bt_fly[idx, jdx] = butt_w(op, H, order)
    
    
    plt.scatter(xhvalues, bt_fly[idx, :],
            marker=SCI_MARKERS[idx]['marker'],
               s=SCI_MARKERS[idx]['size'],
               facecolors=(RMB_COLORS[idx]),label=f"k={odr_values[idx]}")
    plt.plot(xhvalues, bt_fly[idx, :],
               color=(RMB_COLORS[idx]),linestyle=SCI_MARKERS[idx]['drawstyle'],)
    plt.xlabel(r'$x_{ht}$',fontsize=28)
    plt.ylabel(r'$||[H_1,W]_k||$',fontsize=22)
    plt.yscale('log')
    # plt.title('Butterfly')
    plt.legend(ncol=2, framealpha=0.95, fontsize=22)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
# plt.savefig('fig5.pdf', dpi=300)    
plt.show()
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")