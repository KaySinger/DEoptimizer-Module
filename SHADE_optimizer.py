import numpy as np

class SHADE:
    def __init__(self, func, bounds, pop_size=100, max_gen=1000, H=100, tol=1e-6):
        """
        SHADE优化算法类

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
        self.archive_size = pop_size

        # 初始化历史记忆
        self.F_memory = [0.5] * H
        self.CR_memory = [0.5] * H
        self.hist_idx = 0

        # 初始化种群和存档
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.pop_size, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.iteration_log = []

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

    def resize_archive(self):
        if len(self.archive) > self.archive_size:
            np.random.shuffle(self.archive)
            self.archive = self.archive[:self.archive_size]

    def mutant(self, F, i):
        # current-to-pbest/1变异策略
        p_min = max(2 / self.pop_size, 0.05)  # 双重保护
        p_i = np.random.uniform(p_min, 0.2)
        p_best_size = max(2, int(self.pop_size * p_i))

        # 选择p_best
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

    @staticmethod
    def _crossover(parent, mutant, CR):
        """带强制交叉的二项交叉"""
        cross_mask = np.random.rand(parent.shape[0]) < CR
        if not np.any(cross_mask):  # 确保至少一个维度交叉
            cross_mask[np.random.randint(parent.shape[0])] = True
        return np.where(cross_mask, mutant, parent)

    def optimize(self):
        """执行优化过程"""
        for gen in range(self.max_gen):
            S_F, S_CR, S_weights = [], [], []
            new_pop = []

            # 记录当前最优值
            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            if self.tol is not None and best_val <= self.tol:
                print(f"Converged at generation {gen} with precision {best_val}")
                break
            elif gen >= self.max_gen - 1:
                print(f"Converged at generation {gen} with precision {best_val}")
                break

            for i in range(self.pop_size):
                # 从历史记忆随机选择索引
                r = np.random.randint(0, self.H)

                # 生成F和CR
                F = np.clip(np.random.standard_cauchy() * 0.1 + self.F_memory[r], 0, 1)
                CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0, 1)

                # current-to-pbest/1变异策略
                mutant = self.mutant(F, i)

                # 交叉操作
                trial = self._crossover(self.pop[i], mutant, CR)
                trial_fitness = self.func(trial)

                # 贪心选择
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    # 记录成功参数和适应度改进量
                    S_F.append(F)
                    S_CR.append(CR)
                    S_weights.append(self.fitness[i] - trial_fitness)  # 绝对改进量
                    # 更新适应度和存档
                    self.fitness[i] = trial_fitness
                    self.archive.append(self.pop[i].copy())
                else:
                    new_pop.append(self.pop[i])

            # 更新种群
            self.pop = np.array(new_pop)
            self.resize_archive()

            # 更新历史记忆
            if S_F:
                total_weight = np.sum(S_weights)
                if total_weight > 0:
                    # F使用Lehmer均值
                    F_lehmer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                    # CR使用加权算术均值
                    CR_mean = np.sum(np.array(S_CR) * S_weights) / total_weight

                    # 更新记忆
                    self.F_memory[self.hist_idx] = F_lehmer
                    self.CR_memory[self.hist_idx] = CR_mean
                    self.hist_idx = (self.hist_idx + 1) % self.H

            print(f"Iteration {gen + 1}, Best fitness: {np.min(self.fitness)}")

        # 返回最优解
        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log
