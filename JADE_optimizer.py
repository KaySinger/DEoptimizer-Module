import numpy as np


class JADE:
    def __init__(self, func, bounds, pop_size=100, max_gen=1000, p=0.1, tol=1e-6):
        """
        JADE优化算法类

        参数：
        func: 目标函数（最小化）
        bounds: 变量边界的列表，例如 [(min1, max1), (min2, max2), ...]
        pop_size: 种群大小
        max_gen: 最大迭代次数
        p: 选择p_best的比例 (0~1)
        tol: 收敛精度，达到时提前终止
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.p = p
        self.tol = tol

        # 初始化参数
        self.F_mean = 0.5
        self.CR_mean = 0.5
        self.archive = []
        self.iteration_log = []

        # 生成初始种群
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)

    @ staticmethod
    def _repair(mutant, parent, bounds):
        """边界修复策略"""
        for j in range(mutant.shape[0]):
            low, high = bounds[j]
            if mutant[j] < low or mutant[j] > high:
                mutant[j] = (parent[j] + (low if mutant[j] < low else high)) / 2
        return mutant

    def optimize(self):
        """执行优化过程"""
        for gen in range(self.max_gen):
            F_values, CR_values = [], []
            new_pop = []

            # 检查收敛条件
            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            if best_val <= self.tol:
                print(f"Converged at generation {gen} with precision {best_val:.6e}")
                break

            for i in range(self.pop_size):
                # 生成F和CR
                F = np.clip(np.random.standard_cauchy() * 0.1 + self.F_mean, 0, 1)
                CR = np.clip(np.random.normal(self.CR_mean, 0.1), 0, 1)

                # 选择p_best
                p_best_size = max(int(self.pop_size * self.p), 1)  # 至少1个
                p_best_indices = np.argsort(self.fitness)[:p_best_size]
                p_best_idx = np.random.choice(p_best_indices)
                p_best = self.pop[p_best_idx]

                # 合并种群和存档
                combined_pop = np.vstack([self.pop, np.array(self.archive)]) if self.archive else self.pop.copy()

                # 选择a和b（排除当前个体和p_best）
                candidates = []
                for idx in range(len(combined_pop)):
                    # 避免选择当前个体或p_best
                    if not (np.array_equal(combined_pop[idx], self.pop[i]) or np.array_equal(combined_pop[idx], p_best)):
                        candidates.append(idx)

                # 确保能够选择两个不同的个体
                if len(candidates) >= 2:
                    selected = np.random.choice(candidates, 2, replace=False)
                    a, b = combined_pop[selected[0]], combined_pop[selected[1]]
                else:
                    # 回退到当前种群随机选择
                    selected = np.random.choice(self.pop_size, 2, replace=False)
                    a, b = self.pop[selected[0]], self.pop[selected[1]]

                # 变异操作
                mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (a - b)
                mutant = self._repair(mutant, self.pop[i], self.bounds)

                # 交叉操作（确保至少一个维度来自mutant）
                cross_mask = np.random.rand(self.dim) < CR
                if not np.any(cross_mask):  # 如果全部未交叉
                    cross_mask[np.random.randint(self.dim)] = True  # 强制选择一个维度
                trial = np.where(cross_mask, mutant, self.pop[i])

                trial_fitness = self.func(trial)

                # 贪婪选择
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    self.fitness[i] = trial_fitness
                    self.archive.append(self.pop[i].copy())
                    F_values.append(F)
                    CR_values.append(CR)
                    # 控制存档大小
                    if len(self.archive) > self.pop_size:
                        self.archive.pop(np.random.randint(0, len(self.archive)))
                else:
                    new_pop.append(self.pop[i])

            # 更新种群
            self.pop = np.array(new_pop)

            # 更新自适应参数
            if F_values:
                self.F_mean = (1 - 0.1) * self.F_mean + 0.1 * (np.sum(np.square(F_values))) / np.sum(F_values)
            if CR_values:
                self.CR_mean = (1 - 0.1) * self.CR_mean + 0.1 * np.mean(CR_values)

            print(f"Iteration {gen + 1}, Best fitness: {np.min(self.fitness):.6f}")

        # 返回最优解
        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log