import os
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 创建保存图像的文件夹
save_dir = r'C:\Users\13119\Desktop\敏感性测试图'
os.makedirs(save_dir, exist_ok=True)


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


# 扰动测试主函数
def single_perturb_test(initial_k, base_sol):
    sensitivity_matrix = np.zeros((40, 40))  # 存储敏感性数据（40个k x 40个P）
    max_changes = []  # 记录每个k扰动导致的最大浓度变化

    for k_idx in range(40):  # 遍历k0-k39
        for perturb_dir in [0.7, 1.3]:  # ±30%扰动
            # 复制原始系数并施加扰动
            perturbed_k = initial_k.copy()
            perturbed_k[k_idx] *= perturb_dir

            # 求解ODE
            sol = odeint(equations, initial_p, t, args=(perturbed_k,))
            final_conc = sol[-1, 1:]  # 取P1-P40的最终浓度

            # 计算浓度变化量（绝对值百分比变化）
            delta = (final_conc - base_sol[-1, 1:]) / base_sol[-1, 1:] * 100
            delta = np.abs(delta)

            # 更新敏感性矩阵（取最大值）
            sensitivity_matrix[k_idx] = np.maximum(sensitivity_matrix[k_idx], delta)

            # 保存对比图
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, 41), base_sol[-1, 1:], width=0.4, label='Original')
            plt.bar(np.arange(1, 41) + 0.4, final_conc, width=0.4, label=f'k{k_idx} {perturb_dir}x')
            plt.xlabel('Polymer Size')
            plt.ylabel('Concentration')
            plt.title(f'k{k_idx} Perturbation {perturb_dir}x Effect')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'k{k_idx}_perturb_{perturb_dir}.png'))
            plt.close()

            # 记录最大变化
            max_p = np.argmax(delta) + 1  # +1因为P从1开始
            max_changes.append((k_idx, max_p, np.max(delta)))

    # 输出文字结果
    print("关键系数分析：")
    sorted_changes = sorted(max_changes, key=lambda x: x[2], reverse=True)[:10]
    for k_idx, p_idx, change in sorted_changes:
        print(f"k{k_idx}扰动导致 P{p_idx} 变化最大：{change:.1f}%")

    # 绘制热图
    plt.figure(figsize=(12, 8))
    plt.imshow(sensitivity_matrix, cmap='viridis', aspect='auto',
               extent=[1, 40, 39, 0], vmin=0, vmax=np.max(sensitivity_matrix))
    plt.colorbar(label='Concentration Change (%)')
    plt.xlabel('Polymer Size (P)')
    plt.ylabel('Rate Constant (k)')
    plt.title('Sensitivity Heatmap of k0-k39 Perturbations')
    plt.yticks(range(40), [f'k{i}' for i in range(40)])
    plt.show()


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
    # 执行单系数扰动测试
    single_perturb_test(initial_k, base_sol)