import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 数据准备 ====================
# 定义k1-k39的数值（从用户提供的数据）
k = np.array([
    0.2796294846681433, 0.28295165086533625, 0.2831928752345816, 0.2842876182233199, 0.2844829653689831,
    0.2846020267776552,
    0.2853746900185002, 0.28546120653482965, 0.29044005981184273, 0.2905951285383373, 0.29301794699437983,
    0.30186844035226124,
    0.3125298527882659, 0.31721215567926053, 0.31721714105286974, 0.3172512053706637, 0.3173693843935628,
    0.3185265737213199,
    0.3185901244633732, 0.3292550747667668, 0.3293197040141575, 0.32932353981384394, 0.3309724672605064,
    0.3312154382846182,
    0.34728969759604106, 0.3473442396564915, 0.3858786517388793, 0.4069290704302152, 0.40712151028749766,
    0.4372930758897808,
    0.43784919906061665, 0.44760412648378967, 0.5362072036499123, 0.5557151856288806, 0.6130792072265121,
    0.6131788271452184,
    0.7884773735396972, 0.7887702149087181, 0.7896598259944754
])

n = np.arange(1, 40)  # 聚合物序号1-39
p = np.log(2 ** (n))  # 计算尺寸参数p = ln(2^(n+1))


# ==================== 模型定义 ====================
def segmented_power_law(n, a1, x1, a2, x2, a3, x3, trans1, trans2, k1, k2):
    """
    三段幂律模型 + Sigmoid过渡
    参数说明：
    a1, x1: 第一段系数和指数
    a2, x2: 第二段系数和指数
    a3, x3: 第三段系数和指数
    trans1, trans2: 过渡点位置（p值）
    k1, k2: 过渡锐度参数
    """
    # 定义各段基础函数
    p = np.log(2 ** (n))

    region1 = a1 * (p ** x1)
    region2 = a2 * (p ** x2)
    region3 = a3 * (p ** x3)

    # 计算Sigmoid过渡权重
    w1 = 1 / (1 + np.exp(-k1 * (p - trans1)))  # 第一过渡区权重
    w2 = 1 / (1 + np.exp(-k2 * (p - trans2)))  # 第二过渡区权重

    # 组合各段函数
    return (1 - w1) * region1 + (w1 - w2) * region2 + w2 * region3


# ==================== 参数拟合 ====================
# 初始参数猜测（基于数据观察）
initial_guess = [
    0.28, 0.03,  # 第一段 a1, x1
    0.3, 0.4,  # 第二段 a2, x2
    0.01, 2.0,  # 第三段 a3, x3
    np.log(2 ** 15), np.log(2 ** 28),  # 过渡点 trans1, trans2
    1.0, 1.0  # 过渡锐度 k1, k2
]

# 设置参数边界
bounds = (
    [0.2, 0.0, 0.2, 0.2, 0.001, 1.5, 10, 20, 0.1, 0.1],  # 下限
    [0.4, 0.1, 0.4, 0.6, 0.1, 2.5, 15, 25, 5.0, 5.0]  # 上限
)

# 执行拟合
popt, pcov = curve_fit(
    segmented_power_law,
    p, k,
    p0=initial_guess,
    bounds=bounds,
    maxfev=10000
)

# ==================== 结果解析 ====================
# 提取拟合参数
a1, x1, a2, x2, a3, x3, trans1, trans2, k1, k2 = popt

# 计算各过渡点对应的n值
n_trans1 = (np.exp(trans1) / np.log(2)) - 1
n_trans2 = (np.exp(trans2) / np.log(2)) - 1

print("=" * 40)
print("分段幂律拟合结果：")
print(f"第一段 (p < {trans1:.2f}, n < {n_trans1:.1f}): k = {a1:.6f} * p^{x1:.6f}")
print(f"第二段 ({trans1:.2f} ≤ p ≤ {trans2:.2f}, {n_trans1:.1f} ≤ n ≤ {n_trans2:.1f}): k = {a2:.6f} * p^{x2:.6f}")
print(f"第三段 (p > {trans2:.2f}, n > {n_trans2:.1f}): k = {a3:.6f} * p^{x3:.6f}")
print(f"过渡锐度参数：k1={k1:.2f}, k2={k2:.2f}")
print("=" * 40)

# ==================== 可视化 ====================
plt.figure(figsize=(12, 7))

# 绘制原始数据
plt.scatter(p, k, color='navy', zorder=5, label='实验数据')

# 生成拟合曲线
p_fine = np.linspace(min(p), max(p), 300)
k_fit = segmented_power_law(p_fine, *popt)

# 绘制整体拟合曲线
plt.plot(p_fine, k_fit, 'r-', lw=2, label='分段幂律拟合')

# 标记过渡区域
plt.axvline(trans1, color='gray', linestyle='--', alpha=0.6)
plt.axvline(trans2, color='gray', linestyle='--', alpha=0.6)
plt.fill_betweenx([min(k), max(k)], trans1, trans2, color='yellow', alpha=0.1, label='过渡区域')

# 标注各段公式
plt.text(trans1 / 2, 0.3,
         f'$k = {a1:.6f}p^{{{x1:.6f}}}$\n(n < {n_trans1:.1f})',
         ha='center', va='center')
plt.text((trans1 + trans2) / 2, 0.45,
         f'$k = {a2:.6f}p^{{{x2:.6f}}}$\n({n_trans1:.1f} ≤ n ≤ {n_trans2:.1f})',
         ha='center', va='center')
plt.text(trans2 + 2, 0.7,
         f'$k = {a3:.6f}p^{{{x3:.6f}}}$\n(n > {n_trans2:.1f})',
         ha='left', va='center')

# 坐标轴设置
plt.xlabel(r'尺寸参数 $p = \ln(2^{n+1})$', fontsize=12)
plt.ylabel('速率常数 $k$', fontsize=12)
plt.title('聚合物反应速率常数的分段幂律关系', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# ==================== 附加分析 ====================
# 计算决定系数R²
residuals = k - segmented_power_law(n, *popt)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((k - np.mean(k)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"拟合质量评估：")
print(f"- R² = {r_squared:.4f}")
print(f"- 平均绝对误差: {np.mean(np.abs(residuals)):.4f}")
print(f"- 最大残差: {np.max(np.abs(residuals)):.4f}")

# 显示图形
plt.show()