import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False


# 核心动力学方程保持不变
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


def random_perturb_analysis(initial_k, initial_p, t, n_runs=50, perturb_ratio=0.3):
    """整合敏感性和关键性分析的随机扰动模式"""
    # 参数设置
    n_k = 40
    n_perturb = int(n_k * 0.2)
    results = []

    # 扰动实验
    for _ in tqdm(range(n_runs)):
        perturb_indices = np.random.choice(n_k, n_perturb, False)
        perturbed_k = initial_k.copy()
        for idx in perturb_indices:
            perturbed_k[idx] *= np.random.uniform(1 - perturb_ratio, 1 + perturb_ratio)

        sol = odeint(equations, initial_p, t, args=(perturbed_k,))
        results.append(sol[-1, 1:])  # P1-P40

    results = np.array(results)

    # 综合评估
    q1 = np.percentile(results, 25, axis=0)
    q3 = np.percentile(results, 75, axis=0)
    iqr = q3 - q1
    outliers = np.sum((results < (q1 - 1.5 * iqr)) | (results > (q3 + 1.5 * iqr)), axis=0)
    std_conc = np.std(results, axis=0)

    # 动态阈值分类
    norm_iqr = (iqr - np.min(iqr)) / (np.max(iqr) - np.min(iqr) + 1e-8)
    norm_outliers = (outliers - np.min(outliers)) / (np.max(outliers) - np.min(outliers) + 1e-8)

    sensitivity_thresh = np.median(norm_iqr)
    criticality_thresh = np.median(norm_outliers)

    # 四分类系统
    classifications = []
    for i in range(40):
        sens = norm_iqr[i] > sensitivity_thresh
        crit = norm_outliers[i] > criticality_thresh

        if sens and crit:
            cls = "关键敏感物质"
        elif sens:
            cls = "非关键敏感物质"
        elif crit:
            cls = "关键非敏感物质"
        else:
            cls = "非关键非敏感物质"
        classifications.append(cls)

    # 构建结果表
    stats_df = pd.DataFrame({
        'Polymer': [f'P{i + 1}' for i in range(40)],
        'IQR': iqr,
        'Std': std_conc,
        'Outliers': outliers,
        'Classification': classifications
    }).sort_values('Polymer', key=lambda x: x.str[1:].astype(int))

    # 可视化
    plot_combined_analysis(results, classifications, stats_df)

    # 结果输出
    print(f"\n随机扰动分析结果（{n_runs}次实验，每次扰动{n_perturb}个参数）:")
    print_stats_by_category(stats_df)
    print(f"\n全体系数平均标准差: {np.mean(std_conc):.5f}")

    return stats_df


def plot_combined_analysis(results, classifications, stats_df):
    """综合可视化（箱线图+IQR条形图）"""
    # ===== 箱线图 =====
    plt.figure(figsize=(12, 7))
    color_map = {
        "关键敏感物质": "#E74C3C",
        "非关键敏感物质": "#F1C40F",
        "关键非敏感物质": "#3498DB",
        "非关键非敏感物质": "#95A5A6"
    }

    box = plt.boxplot(
        results,
        tick_labels=[f'P{i + 1}' for i in range(40)],
        patch_artist=True,
        widths=0.6
    )

    # 设置颜色
    for patch, cls in zip(box['boxes'], classifications):
        patch.set_facecolor(color_map[cls])

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in color_map.items()]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title("浓度分布与关键性分析", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ===== 敏感性条形图 =====
    plt.figure(figsize=(12, 7))
    colors = [color_map[cls] for cls in classifications]
    bars = plt.bar(range(40), stats_df['IQR'], color=colors)

    plt.title("敏感性分析（IQR指标）", fontsize=14)
    plt.xticks(range(40), stats_df['Polymer'], rotation=90, fontsize=8)
    plt.axhline(np.median(stats_df['IQR']), color='k', linestyle='--', label='中位IQR')
    plt.ylabel("IQR值")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_stats_by_category(stats_df):
    """按分类输出统计结果"""
    categories = {
        "关键敏感物质": [],
        "非关键敏感物质": [],
        "关键非敏感物质": [],
        "非关键非敏感物质": []
    }

    for _, row in stats_df.iterrows():
        categories[row['Classification']].append(row)

    for cls in categories:
        if not categories[cls]: continue
        print(f"\n{cls}:")
        for item in categories[cls]:
            print(f"  {item['Polymer']}: 标准差={item['Std']:.5f} 异常值={item['Outliers']}")
        avg_std = np.mean([x['Std'] for x in categories[cls]])
        print(f"  类别平均标准差: {avg_std:.5f}")

# ================== 主程序 ==================
if __name__ == '__main__':
    # 初始化参数
    initial_p = np.zeros(41)
    initial_p[0] = 10
    t = np.linspace(0, 1000, 1000)

    # 加载参数
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
    initial_k = list(k.values()) + list(k_inv.values())

    # 运行整合分析
    stats_df = random_perturb_analysis(initial_k, initial_p, t)

    # 保存结果
    stats_df.to_csv("combined_analysis.csv", index=False)
