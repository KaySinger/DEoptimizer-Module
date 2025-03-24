import numpy as np

class JSO:
    def __init__(self, func, bounds, pop_size=100, max_gen=1000, H=100, tol=1e-6):
        """
        jSO优化算法类

        参数：
        func: 目标函数（最小化）
        bounds: 变量边界的列表，例如 [(min1, max1), (min2, max2), ...]
        pop_size: 种群大小
        max_gen: 最大迭代次数
        H: 历史记忆大小
        tol: 收敛精度，达到时提前终止
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.H = H
        self.tol = tol
        self.min_pop_size = 4  # 最小种群大小
        self.arc_rate = 2.6  # 存档大小系数

        # 初始化历史记忆
        self.F_memory = [0.5] * H
        self.CR_memory = [0.8] * H  # 初始CR=0.8
        self.hist_idx = 0

        # 初始化种群和存档
        self.pop = self._quasi_opposition_init()
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.iteration_log = []
        self.nfe = 0  # 评估次数计数器

    def _quasi_opposition_init(self):
        """准对立初始化"""
        pop = []
        for _ in range(self.pop_size):
            # 生成准对立解
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            x_opposite = self.bounds[:, 0] + self.bounds[:, 1] - x
            # 选择更优的解
            if self.func(x) < self.func(x_opposite):
                pop.append(x)
            else:
                pop.append(x_opposite)
        return np.array(pop)

    @staticmethod
    def _repair(mutant, parent, bounds):
        """边界修复策略"""
        repaired = np.copy(mutant)
        for j in range(mutant.shape[0]):
            low, high = bounds[j]
            if repaired[j] < low:
                repaired[j] = (parent[j] + low) / 2
            elif repaired[j] > high:
                repaired[j] = (parent[j] + high) / 2
        return repaired

    @staticmethod
    def _crossover(parent, mutant, CR):
        """带强制交叉的二项交叉"""
        cross_mask = np.random.rand(parent.shape[0]) < CR
        if not np.any(cross_mask):
            cross_mask[np.random.randint(parent.shape[0])] = True
        return np.where(cross_mask, mutant, parent)

    def optimize(self):
        """执行优化过程"""
        max_nfe = self.pop_size * self.max_gen  # 最大评估次数
        for gen in range(self.max_gen):
            S_F, S_CR, S_weights = [], [], []
            new_pop = []

            # 记录当前最优值
            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            if best_val <= self.tol:
                print(f"Converged at generation {gen} with precision {best_val:.6e}")
                break

            for i in range(self.pop_size):
                # ========================= 参数生成 =========================
                # 从历史记忆随机选择索引
                r = np.random.randint(0, self.H)
                # 处理最后一位参数为0.9（论文要求）
                if r == self.H - 1:
                    mu_sf, mu_cr = 0.9, 0.9
                else:
                    mu_sf = self.F_memory[r]
                    mu_cr = self.CR_memory[r]

                # 生成F和CR（jSO的调整）
                F = np.clip(np.random.standard_cauchy() * 0.1 + mu_sf, 0, 1)
                CR = np.clip(np.random.normal(mu_cr, 0.1), 0, 1)

                # jSO的CR阶段式调整（根据当前评估次数）
                if self.nfe < 0.25 * max_nfe:
                    CR = max(CR, 0.7)
                elif self.nfe < 0.5 * max_nfe:
                    CR = max(CR, 0.6)

                # jSO的F时变调整
                if self.nfe < 0.6 * max_nfe and F > 0.7:
                    F = 0.7

                # ========================= 变异策略 =========================
                # 动态计算p值（论文公式）
                p_min, p_max = 0.05, 0.15
                p_i = p_min + (self.nfe / max_nfe) * (p_max - p_min)
                p_best_size = max(2, int(self.pop_size * p_i))

                # 选择p_best个体
                p_best_indices = np.argsort(self.fitness)[:p_best_size]
                p_best_idx = np.random.choice(p_best_indices)
                p_best = self.pop[p_best_idx]

                # 合并种群和存档
                combined_pop = np.vstack([self.pop, np.array(self.archive)]) if self.archive else self.pop.copy()

                # 选择a和b（排除当前个体和p_best）
                candidates = []
                for idx in range(len(combined_pop)):
                    if not (np.array_equal(combined_pop[idx], self.pop[i]) or
                            np.array_equal(combined_pop[idx], p_best)):
                        candidates.append(idx)

                # 处理候选不足的情况
                if len(candidates) >= 2:
                    selected = np.random.choice(candidates, 2, replace=False)
                    a, b = combined_pop[selected[0]], combined_pop[selected[1]]
                else:
                    selected = np.random.choice(self.pop_size, 2, replace=False)
                    a, b = self.pop[selected[0]], self.pop[selected[1]]

                # 时变F调整（论文第III.C节）
                if self.nfe < 0.2 * max_nfe:
                    jF = F * 0.7
                elif self.nfe < 0.4 * max_nfe:
                    jF = F * 0.8
                else:
                    jF = F * 1.2

                # 变异操作
                mutant = self._repair(
                    self.pop[i] + jF * (p_best - self.pop[i]) + F * (a - b),
                    self.pop[i],
                    self.bounds
                )

                # ========================= 交叉操作 =========================
                trial = self._crossover(self.pop[i], mutant, CR)
                trial_fitness = self.func(trial)
                self.nfe += 1  # 评估次数递增

                # 贪心选择
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    # 记录成功参数和权重
                    S_F.append(F)
                    S_CR.append(CR)
                    S_weights.append(self.fitness[i] - trial_fitness)
                    # 更新适应度和存档
                    self.fitness[i] = trial_fitness
                    self.archive.append(self.pop[i].copy())
                    # 控制存档大小（论文要求2.6倍）
                    if len(self.archive) > self.arc_rate * self.pop_size:
                        self.archive.pop(np.random.randint(0, len(self.archive)))
                else:
                    new_pop.append(self.pop[i])

            # ======================== 种群和记忆更新 ========================
            self.pop = np.array(new_pop)

            # 更新历史记忆（加权Lehmer均值）
            if S_F:
                total_weight = np.sum(S_weights)
                F_lehmer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                CR_mean = np.sum(np.array(S_CR) * S_weights) / total_weight
                # 平滑更新（论文要求）
                self.F_memory[self.hist_idx] = (F_lehmer + self.F_memory[self.hist_idx]) / 2
                self.CR_memory[self.hist_idx] = (CR_mean + self.CR_memory[self.hist_idx]) / 2
                self.hist_idx = (self.hist_idx + 1) % self.H

            # 线性种群缩减（LPSR）
            plan_pop_size = int(self.min_pop_size + (self.pop_size - self.min_pop_size) *
                                (1 - self.nfe / max_nfe))
            if plan_pop_size < self.pop_size:
                sorted_indices = np.argsort(self.fitness)
                self.pop = self.pop[sorted_indices[:plan_pop_size]]
                self.fitness = self.fitness[sorted_indices[:plan_pop_size]]
                self.pop_size = plan_pop_size

            # 输出迭代信息
            print(f"Iteration {gen + 1}, Best Fitness: {np.min(self.fitness):.6e}")

        # 返回最优解
        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log