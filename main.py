import numpy as np
from numba import jit
import json
from scipy.optimize import minimize
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from SHADE_optimizer import SHADE

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, x_values, scale_factor):
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations

# 定义非线性微分方程组
@jit(nopython=True)
def equations(p, t, k_values):
    dpdt = np.zeros_like(p)
    k = k_values[:40]
    k_inv = k_values[40:]
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] + k_inv[0] * p[2] - k[1] * p[1] ** 2
    for i in range(2, 40):
        dpdt[i] = k[i - 1] * p[i - 1] ** 2 + k_inv[i - 1] * p[i + 1] - k_inv[i - 2] * p[i] - k[i] * p[i] ** 2
    dpdt[40] = k[39] * p[39] ** 2 - k_inv[38] * p[40]
    return dpdt

# 定义目标函数
def objective(k):
    # 正向系数递增性惩罚项
    k_forward = k[1:40]
    penalty = 0.0
    # 计算所有相邻k的递减量，若k[i+1] < k[i]则施加惩罚
    for i in range(len(k_forward) - 1):
        if k_forward[i + 1] < k_forward[i]:
            penalty += (k_forward[i] - k_forward[i + 1]) ** 2  # 平方惩罚项
    penalty_weight = 1e6  # 惩罚权重（根据问题规模调整）
    total_penalty = penalty_weight * penalty

    initial_p = [10.0] + [0] * 40
    t = np.linspace(0, 1000, 1000)
    # 求解微分方程
    sol = odeint(equations, initial_p, t, args=(k,))
    # 选取t>=900时的所有解（假设t=1000时有1000个点，索引900对应t=900）
    selected_sol = sol[900:, :]
    # 理想浓度
    ideal_p = np.array([0] + list(target_p))
    # 计算所有选中时间点的误差平方和
    sum_error = np.sum((selected_sol - ideal_p) ** 2)

    return sum_error + total_penalty

# 设置变量边界
bounds = np.array([(0.1, 0.2)] + [(0, 1.0)] * 39 + [(0, 0.2)] * 39)

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=20.5, sigma=10, total_concentration=1.0, x_values=np.arange(1, 41), scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 41)]  # 定义图像横坐标
print("理想最终浓度:")
print(json.dumps({f'P{i}': float(c) for i, c in enumerate(target_p, start=1)}, indent=4))

# 运行差分进化算法
result = SHADE(objective, bounds=bounds, pop_size=400, max_gen=5000, H=400, tol=1e-6)
best_solution, best_fitness = result.optimize()
print("全局优化得到的系数k:")
print(json.dumps({f'k{i}': float(c) for i, c in enumerate(best_solution[:40], start=0)}, indent=4))
print("全局优化得到的系数k_inv:")
print(json.dumps({f'k{i}_inv': float(c) for i, c in enumerate(best_solution[40:], start=1)}, indent=4))
print("全局优化精度:", best_fitness)

# 梯度优化，进一步提高精度
print("开始梯度优化")

result_final = minimize(objective, best_solution, method='L-BFGS-B', bounds=bounds, tol=1e-8)
optimal_k = result_final.x
final_precision = result_final.fun

print("系数k:")
print(json.dumps({f'k{i}': float(c) for i, c in enumerate(optimal_k[:40], start=0)}, indent=4))
print("系数k_inv:")
print(json.dumps({f'k{i}_inv': float(c) for i, c in enumerate(optimal_k[40:], start=1)}, indent=4))
print("最终精度:", final_precision)

# 使用得到的系数求解
initial_p = [10.0] + [0] * 40
t = np.linspace(0, 1000, 1000)
sol = odeint(equations, initial_p, t, args=(best_solution,))

# 绘制理想稳态浓度曲线
plt.figure(figsize=(15, 8))
plt.xlabel("P-Species")
plt.ylabel("P-Concentrations")
plt.title("Ideal Concentrations and Actual Concentrations")
plt.xticks(range(len(x_values)), x_values, rotation=90)
final_concentrations = sol[-1, 1:]
plt.plot(range(len(x_values)), target_p, label = 'Ideal Concentrations', marker='o', linestyle='-', color='blue')
plt.plot(range(len(x_values)), final_concentrations, label = 'Actual Concentrations', marker='o', linestyle='-', color='red')
plt.grid(True)
plt.show()

# 绘图函数
plt.figure(figsize=(15, 8))
plt.plot(t, sol[:, 0], label='p0')
for i in range(1, 11):
    plt.plot(t, sol[:, i], label=f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P0-P10 Concentration over Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 8))
for i in range(11, 21):
    plt.plot(t, sol[:, i], label=f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P11-P20 Concentration over Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 8))
for i in range(21, 31):
    plt.plot(t, sol[:, i], label=f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P21-P30 Concentration over Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 8))
for i in range(31, 41):
    plt.plot(t, sol[:, i], label=f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P31-P40 Concentration over Time')
plt.grid(True)
plt.show()