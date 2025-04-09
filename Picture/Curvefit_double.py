import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

# ==================== 数据准备 ====================
k = np.array([
    0.279629, 0.282952, 0.283193, 0.284288, 0.284483, 0.284602,
    0.285375, 0.285461, 0.290440, 0.290595, 0.293018, 0.301868,
    0.312530, 0.317212, 0.317217, 0.317251, 0.317369, 0.318527,
    0.318590, 0.329255, 0.329320, 0.329324, 0.330972, 0.331215,
    0.347290, 0.347344, 0.385879, 0.406929, 0.407122, 0.437293,
    0.437849, 0.447604, 0.536207, 0.555715, 0.613079, 0.613179,
    0.788477, 0.788770, 0.789660
])

n = np.arange(1, 40)  # 聚合物等级1-39（直接作为x轴）
p = np.log(2**n)

# ==================== 二段式模型定义 ====================
def two_segment_model(p, a1, x1, a2, x2, p_trans, k_trans):
    """
    直接以n为自变量的二段式模型
    参数：
    a1, x1 - 平缓区参数
    a2, x2 - 增长区参数
    n_trans - 过渡点（直接是n值）
    k_trans - 过渡锐度
    """
    # 基础函数
    region1 = a1 * (p ** x1)
    region2 = a2 * (p ** x2)

    # Sigmoid过渡
    w = 1 / (1 + np.exp(-k_trans * (p - p_trans)))

    return (1 - w) * region1 + w * region2


# ==================== 参数拟合 ====================
# 初始参数设置（基于数据观察）
initial_guess = [0.28, 0.01, 0.01, 1.8, np.log(2**22), 1.0]
bounds = (
    [0.2,  0.0,  0.001, 1.5, np.log(2**15), 0.1],  # 下限
    [0.3,  0.1,  0.1,   2.5, np.log(2**28), 5.0]    # 上限
)

# 执行拟合
popt, _ = curve_fit(two_segment_model, p, k,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=10000)

a1, x1, a2, x2, p_trans, k_trans = popt
print(f"平缓区参数 a1={a1}, x1={x1}")
print(f"增长区参数 a2={a2}, x2={x2}")

n_trans = p_trans / np.log(2)

# ==================== 可视化 ====================
plt.figure(figsize=(12, 7))

# 原始数据
plt.scatter(n, k, color='navy', s=50, label='实验数据')

# 拟合曲线
n_fine = np.linspace(1, 39, 200)
p_fine = np.log(2**n_fine)
k_fit = two_segment_model(p_fine, *popt)
plt.plot(n_fine, k_fit, 'r-', lw=2, label='二段式拟合')

# 标记过渡区
trans_width = (2 / k_trans) / np.log(2) - 1  # 过渡区宽度
plt.axvline(n_trans, color='red', ls='--', alpha=0.6)
plt.fill_betweenx([min(k), max(k)],
                  n_trans - trans_width, n_trans + trans_width,
                  color='yellow', alpha=0.1, label='活跃区')

# 标注参数
plt.text(0.15, 0.85,
         f'平缓区: $k = {a1:.3f}p^{{{x1:.3f}}}$\n'
         f'增长区: $k = {a2:.3f}p^{{{x2:.3f}}}$\n'
         f'过渡点: n = {n_trans:.1f} ± {trans_width:.1f}',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
         transform=plt.gca().transAxes,
         ha='left', va='top')

plt.xlabel('聚合物等级 n', fontsize=12)
plt.ylabel('速率常数 k', fontsize=12)
plt.title('基于聚合物等级n的二段式模型拟合', fontsize=14)
plt.legend()
plt.grid(True)

# ==================== 结果输出 ====================
residuals = k - two_segment_model(p, *popt)
r_squared = 1 - np.sum(residuals ** 2) / np.sum((k - np.mean(k)) ** 2)

print("=" * 40)
print(f"平缓区 (n < {n_trans:.1f}): k = {a1:.4f} * p^{x1:.4f}")
print(f"增长区 (n ≥ {n_trans:.1f}): k = {a2:.4f} * p^{x2:.4f}")
print(f"活跃区: n = {n_trans:.1f} ± {trans_width:.1f}")
print(f"过渡锐度: k_trans = {k_trans:.2f}")
print(f"拟合优度 R² = {r_squared:.4f}")
print("=" * 40)

plt.tight_layout()
plt.show()