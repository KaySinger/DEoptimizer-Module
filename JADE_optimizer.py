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
        self.archive_size = pop_size

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

    def resize_archive(self):
        if len(self.archive) > self.archive_size:
            np.random.shuffle(self.archive)
            self.archive = self.archive[:self.archive_size]

    def mutant(self, F, i):
        # 选择p_best
        p_best_size = max(int(self.pop_size * self.p), 2)  # 至少1个
        p_best_indices = np.argsort(self.fitness)[:p_best_size]
        p_best_idx = np.random.choice(p_best_indices)
        p_best = self.pop[p_best_idx]

        # 选择a（来自种群）
        a_candidates = [idx for idx in range(self.pop_size)
                        if idx != i and not np.array_equal(self.pop[idx], p_best)]

        if len(a_candidates) == 0:
            a = self.pop[np.random.choice([x for x in range(self.pop_size) if x != i])]
        else:
            a = self.pop[np.random.choice(a_candidates)]

        # 合并种群和存档
        combined_pop = np.vstack([self.pop, np.array(self.archive)]) if self.archive else self.pop

        # 选择b（来自合并种群）
        b_candidates = []
        for idx in range(len(combined_pop)):
            if not np.array_equal(combined_pop[idx], self.pop[i]) and \
                    not np.array_equal(combined_pop[idx], p_best) and \
                    not np.array_equal(combined_pop[idx], a):
                b_candidates.append(idx)

        if len(b_candidates) == 0:
            b = combined_pop[np.random.choice(len(combined_pop))]
        else:
            b = combined_pop[np.random.choice(b_candidates)]

        # 变异操作
        mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (a - b)
        mutant = self._repair(mutant, self.pop[i], self.bounds)

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

                mutant = self.mutant(F, i)

                # 交叉操作
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
                else:
                    new_pop.append(self.pop[i])

            # 更新种群
            self.pop = np.array(new_pop)
            self.resize_archive()

            # 更新自适应参数
            if F_values:
                self.F_mean = (1 - 0.1) * self.F_mean + 0.1 * (np.sum(np.square(F_values))) / np.sum(F_values)
            if CR_values:
                self.CR_mean = (1 - 0.1) * self.CR_mean + 0.1 * np.mean(CR_values)

            print(f"Iteration {gen + 1}, Best fitness: {np.min(self.fitness)}")

        # 返回最优解
        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log
