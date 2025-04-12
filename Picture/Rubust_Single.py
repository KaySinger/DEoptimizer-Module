import os
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

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

def get_perturbation_settings():
    """获取用户自定义的扰动设置"""
    settings = []
    while True:
        print("\n请选择扰动类型（输入q结束设置）：")
        print("1. 正向扰动（+%）")
        print("2. 负向扰动（-%）")
        choice = input("请输入选择（1/2/q）：").strip().lower()

        if choice == 'q':
            break

        if choice not in ['1', '2']:
            print("输入无效，请重新选择")
            continue

        try:
            percentage = float(input(f"请输入扰动幅度（{'正' if choice == '1' else '负'}向百分比，如30表示30%）："))
            if percentage <= 0:
                print("幅度必须为正数")
                continue

            if choice == '1':
                settings.append(('+', 1 + percentage / 100))
            else:
                settings.append(('-', 1 - percentage / 100))

            print(f"已添加 {'+' if choice == '1' else '-'}{percentage}% 扰动")

        except ValueError:
            print("请输入有效数字")

    return settings if settings else [('+', 1.3), ('-', 0.7)]  # 默认±30%


# 扰动测试主函数
def single_perturb_test(initial_k, base_sol):
    # 获取用户自定义扰动设置
    perturb_settings = get_perturbation_settings()

    sensitivity_matrix = np.zeros((40, 40))  # 存储敏感性数据
    all_changes = []  # 存储所有扰动数据

    for k_idx in range(40):  # 遍历k0-k39
        for direction, factor in perturb_settings:  # 使用用户定义的扰动
            # 复制并扰动系数
            perturbed_k = initial_k.copy()
            perturbed_k[k_idx] *= factor

            # 求解ODE
            sol = odeint(equations, initial_p, t, args=(perturbed_k,))
            final_conc = sol[-1, 1:]  # P1-P40

            # 计算变化百分比（保留符号）
            delta = (final_conc - base_sol[-1, 1:]) / base_sol[-1, 1:] * 100

            # 存储结果（格式：k_idx, 方向, 因子, delta）
            all_changes.append((k_idx, direction, factor, delta))

            # 更新热图矩阵（绝对值）
            sensitivity_matrix[k_idx] = np.maximum(sensitivity_matrix[k_idx], np.abs(delta))

            # 生成对比曲线图
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, 41), base_sol[-1, 1:], 'b-o',
                     linewidth=1.5, markersize=4,
                     markeredgecolor='b', markerfacecolor='none',
                     label='Original')

            # 根据扰动方向选择颜色
            line_color = 'g' if direction == '+' else 'r'
            plt.plot(range(1, 41), final_conc, f'{line_color}--s',
                     linewidth=1.5, markersize=4,
                     markeredgecolor=line_color, markerfacecolor='none',
                     label=f'k{k_idx} {direction}{abs((factor - 1) * 100):.0f}%')

            plt.xlabel('Polymer Size', fontsize=12)
            plt.ylabel('Concentration', fontsize=12)
            title_direction = "正向" if direction == "+" else "负向"
            plt.title(f'k{k_idx} {title_direction}{abs((factor - 1) * 100):.0f}%扰动效果', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()

            # 根据扰动方向保存到不同子文件夹
            direction_dir = os.path.join(save_dir, f"{title_direction}扰动")
            os.makedirs(direction_dir, exist_ok=True)
            plt.savefig(os.path.join(direction_dir,
                                     f'k{k_idx}_{direction}{abs((factor - 1) * 100):.0f}perturb.png'),
                        dpi=300)
            plt.close()

    # 分析结果输出
    print("\n=== 扰动设置 ===")
    for i, (d, f) in enumerate(perturb_settings, 1):
        print(f"{i}. {d}{abs((f - 1) * 100):.0f}%扰动")

    print("\n=== 综合影响分析 ===")
    print("影响最大的10个系数（按平均绝对影响排序）：")
    print("排名 | 系数 | 平均影响(%) | 最大影响(P) | 变化幅度(%)")
    print("-" * 60)

    # 计算每个k的平均影响
    k_effects = {}
    for k_idx in range(40):
        total_effect = 0
        count = 0
        max_effect = 0
        max_p = 0

        for record in all_changes:
            if record[0] == k_idx:
                current_max = np.max(np.abs(record[3]))
                if current_max > max_effect:
                    max_effect = current_max
                    max_p = np.argmax(np.abs(record[3])) + 1
                total_effect += np.mean(np.abs(record[3]))
                count += 1

        if count > 0:
            k_effects[k_idx] = (total_effect / count, max_effect, max_p)

    # 排序输出
    sorted_effects = sorted(k_effects.items(), key=lambda x: x[1][0], reverse=True)[:10]
    for rank, (k_idx, (avg, max_val, max_p)) in enumerate(sorted_effects, 1):
        print(f"{rank:4} | k{k_idx:2} | {avg:9.1f} | P{max_p:2}      | {max_val:9.1f}")

    # 热图绘制（使用inferno提高对比度）
    plt.figure(figsize=(14, 8))
    plt.imshow(sensitivity_matrix, cmap='inferno', aspect='auto',
               extent=[1, 40, 39, 0], vmin=0, vmax=np.max(sensitivity_matrix))
    plt.colorbar(label='浓度变化绝对值(%)', shrink=0.8)
    plt.xlabel('聚合物尺寸 (P)', fontsize=12)
    plt.ylabel('速率常数 (k)', fontsize=12)

    # 在标题中显示使用的扰动设置
    title_desc = "、".join([f"{d}{abs((f - 1) * 100):.0f}%" for d, f in perturb_settings])
    plt.title(f'k0-k39扰动敏感性热图（{title_desc}）', fontsize=14, pad=20)

    plt.yticks(range(40), [f'k{i}' for i in range(40)], fontsize=8)
    plt.xticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'热图_{title_desc}.png'), dpi=300)
    plt.show()


# 运行测试（保持您的原有main部分不变）
if __name__ == '__main__':
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
    single_perturb_test(initial_k, base_sol)
