import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 定义核心函数
def equations(p, t, k_values):
    dpdt = np.zeros_like(p)
    k = k_values[:40]
    k_inv = k_values[40:]
    dpdt[0] = -k[0] * p[0]
    dpdt[1] = k[0] * p[0] + k_inv[0] * p[2] - k[1] * (p[1] ** 2)
    for i in range(2, 40):
        dpdt[i] = k[i - 1] * (p[i - 1] ** 2) + k_inv[i - 1] * p[i + 1] - k_inv[i - 2] * p[i] - k[i] * (p[i] ** 2)
    dpdt[40] = k[39] * (p[39] ** 2) - k_inv[38] * p[40]
    return dpdt

def plot_concentration_combined(t, sol):
    # 创建一个包含 5 行 1 列的子图布局
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))  # figsize 为 (宽, 高)

    # 绘制 P0-P10 的浓度变化
    for i in range(11):
        axs[0].plot(t, sol[:, i], label=f'P{i}')
    axs[0].set_title('P0-P10 Concentration')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Concentration')
    axs[0].legend()  # 调整图例位置
    axs[0].grid(True)

    # 绘制 P11-P20 的浓度变化
    for i in range(11, 21):
        axs[1].plot(t, sol[:, i], label=f'P{i}')
    axs[1].set_title('P11-P20 Concentration')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Concentration')
    axs[1].legend()
    axs[1].grid(True)

    # 绘制 P21-P30 的浓度变化
    for i in range(21, 31):
        axs[2].plot(t, sol[:, i], label=f'P{i}')
    axs[2].set_title('P21-P30 Concentration')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Concentration')
    axs[2].legend()
    axs[2].grid(True)

    # 绘制 P31-P40 的浓度变化
    for i in range(31, 41):
        axs[3].plot(t, sol[:, i], label=f'P{i}')
    axs[3].set_title('P31-P40 Concentration')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Concentration')
    axs[3].legend()
    axs[3].grid(True)

    # 自动调整布局，避免重叠
    plt.tight_layout()
    plt.show()

def extreme_perturb_test(initial_k, base_sol):
    """对k1-k39进行±90%极端扰动测试"""
    extreme_results = []
    k_indices = range(1, 40)  # 只测试k1-k39

    plt.figure(figsize=(15, 10))

    for i, k_idx in enumerate(k_indices):
        # 原始系数备份
        perturbed_k = initial_k.copy()

        # 施加-90%和+90%扰动
        for ratio in [0.1, 1.9]:
            perturbed_k[k_idx] = initial_k[k_idx] * ratio

            # 求解ODE
            sol = odeint(equations, initial_p, t, args=(perturbed_k,))
            extreme_results.append((k_idx, ratio, sol[-1, 1:]))

            # 绘制扰动曲线（浅色半透明）
            plt.plot(range(1, 41), sol[-1, 1:],
                     color='grey', alpha=0.1, linestyle='-')

    # 绘制原始曲线（加粗突出）
    plt.plot(range(1, 41), base_sol[-1, 1:],
             'r-', linewidth=3, label='Original')

    # 标记异常情况
    max_dev = 0
    for k_idx, ratio, conc in extreme_results:
        dev = np.max(np.abs(conc - base_sol[-1, 1:]))
        if dev > max_dev:
            max_dev = dev
            worst_k = k_idx
            worst_ratio = ratio

    plt.plot(range(1, 41), extreme_results[worst_k * 2][2],
             'b--', linewidth=2, label=f'Max Dev: k{worst_k} {worst_ratio}x')

    # 图表美化
    plt.xlabel('Polymer Size')
    plt.ylabel('Concentration')
    plt.title('Extreme Perturbation (±90%) Effects on Steady State')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 重新计算最坏扰动情况并绘制时间演化
    perturbed_k = initial_k.copy()
    perturbed_k[worst_k] = initial_k[worst_k] * worst_ratio
    worst_sol = odeint(equations, initial_p, t, args=(perturbed_k,))
    plot_concentration_combined(t, worst_sol)

    # 文字报告
    print("\n极端扰动测试结果：")
    print(f"最敏感系数: k{worst_k} ({worst_ratio}x扰动时产生最大偏差)")
    print(f"最大浓度偏差量: {max_dev:.2f}")
    print("灰色线条显示所有k1-k39的±90%扰动结果")


# 运行测试
if __name__ == '__main__':
    # 初始条件和参数
    initial_p = np.zeros(41)
    initial_p[0] = 10
    t = np.linspace(0, 1000, 1000)

    k = {"k0": 0.1, "k1": 0.2796294846681433, "k2": 0.28295165086533625, "k3": 0.2831928752345816,
         "k4": 0.2842876182233199, "k5": 0.2844829653689831, "k6": 0.2846020267776552,
         "k7": 0.2853746900185002, "k8": 0.28546120653482965, "k9": 0.29044005981184273, "k10": 0.2905951285383373,
         "k11": 0.29301794699437983, "k12": 0.30186844035226124,
         "k13": 0.3125298527882659, "k14": 0.31721215567926053, "k15": 0.31721714105286974, "k16": 0.3172512053706637,
         "k17": 0.3173693843935628, "k18": 0.3185265737213199,
         "k19": 0.3185901244633732, "k20": 0.3292550747667668, "k21": 0.3293197040141575, "k22": 0.32932353981384394,
         "k23": 0.3309724672605064, "k24": 0.3312154382846182,
         "k25": 0.34728969759604106, "k26": 0.3473442396564915, "k27": 0.3858786517388793, "k28": 0.4069290704302152,
         "k29": 0.40712151028749766, "k30": 0.4372930758897808,
         "k31": 0.43784919906061665, "k32": 0.44760412648378967, "k33": 0.5362072036499123, "k34": 0.5557151856288806,
         "k35": 0.6130792072265121, "k36": 0.6131788271452184,
         "k37": 0.7884773735396972, "k38": 0.7887702149087181, "k39": 0.7896598259944754}
    k_inv = {"k1_inv": 0.01443437964224509, "k2_inv": 0.01784449874328581, "k3_inv": 0.021592353105510755,
             "k4_inv": 0.02595100787328786, "k5_inv": 0.030784584285417818,
             "k6_inv": 0.03613989279905994, "k7_inv": 0.04210034219829905, "k8_inv": 0.04843718536730609,
             "k9_inv": 0.056129982389919335, "k10_inv": 0.06331667436694616,
             "k11_inv": 0.07126676260834372, "k12_inv": 0.08114890683141902, "k13_inv": 0.09192597823217666,
             "k14_inv": 0.10107312299398491, "k15_inv": 0.10839666451573603,
             "k16_inv": 0.11512339379848416, "k17_inv": 0.1210579610260175, "k18_inv": 0.12646852626869348,
             "k19_inv": 0.13034324089396937, "k20_inv": 0.13742337548949485,
             "k21_inv": 0.13882732619037358, "k22_inv": 0.13884056743247208, "k23_inv": 0.13813921417712163,
             "k24_inv": 0.135515820739825, "k25_inv": 0.1378665920402216,
             "k26_inv": 0.13250532079809435, "k27_inv": 0.1399921757047886, "k28_inv": 0.13905491621913857,
             "k29_inv": 0.12969864493878502, "k30_inv": 0.1285987269678805,
             "k31_inv": 0.11766649206442223, "k32_inv": 0.10882985936447974, "k33_inv": 0.1167715494430669,
             "k34_inv": 0.10733099769955946, "k35_inv": 0.10394764710764955,
             "k36_inv": 0.09034301580205746, "k37_inv": 0.09995339116167218, "k38_inv": 0.08520343173840925,
             "k39_inv": 0.0719522340192938}
    best_solution = list(k.values()) + list(k_inv.values())

    initial_k = list(best_solution)

    base_sol = odeint(equations, initial_p, t, args=(initial_k,))
    extreme_perturb_test(initial_k, base_sol)
