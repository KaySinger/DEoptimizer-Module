import numpy as np

class L_SHADE:
    def __init__(self, func, bounds, pop_size=100, max_gen=1000, H=100, tol=1e-6, N_min=18):
        """
        L-SHADE优化算法类

        参数：
        func: 目标函数（最小化）
        bounds: 变量边界的列表，例如 [(min1, max1), (min2, max2), ...]
        pop_size: 初始种群大小
        max_gen: 最大迭代次数
        H: 历史记忆大小
        tol: 收敛精度，达到时提前终止
        N_min: 最小种群大小（默认18）
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.N_init = pop_size
        self.N_min = N_min
        self.max_gen = max_gen
        self.H = H
        self.tol = tol

        # 初始化历史记忆
        self.F_memory = [0.5] * H
        self.CR_memory = [0.5] * H
        self.hist_idx = 0

        # 初始化种群和存档
        self.N_current = self.N_init  # 当前种群大小
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.iteration_log = []

    def _linear_pop_size_reduction(self, gen):
        """线性种群缩减策略"""
        return max(
            self.N_min,
            int(round(
                self.N_init - (self.N_init - self.N_min) * gen / self.max_gen
            ))
        )

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
        if len(self.archive) > self.N_current:
            np.random.shuffle(self.archive)
            self.archive = self.archive[:self.N_current]

    def mutant(self, F, i):
        # current-to-pbest/1变异策略
        p_min = max(2 / self.N_current, 0.05)  # 双重保护
        p_i = np.random.uniform(p_min, 0.2)
        p_best_size = max(2, int(self.N_current * p_i))

        # 选择p_best
        p_best_indices = np.argsort(self.fitness)[:p_best_size]
        p_best_idx = np.random.choice(p_best_indices)
        p_best = self.pop[p_best_idx]

        # 选择a（来自种群）
        a_candidates = [idx for idx in range(self.N_current)
                        if idx != i and not np.array_equal(self.pop[idx], p_best)]

        if len(a_candidates) == 0:
            a = self.pop[np.random.choice([x for x in range(self.N_current) if x != i])]
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
        if not np.any(cross_mask):
            cross_mask[np.random.randint(parent.shape[0])] = True
        return np.where(cross_mask, mutant, parent)

    def optimize(self):
        """执行优化过程"""
        for gen in range(self.max_gen):
            S_F, S_CR, S_weights = [], [], []
            new_pop = []
            new_fitness = []

            # 记录当前最优值
            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            # 收敛检查
            if self.tol is not None and best_val <= self.tol:
                print(f"Converged at generation {gen + 1}: {best_val:.6e}")
                break
            elif gen >= self.max_gen - 1:
                print(f"Converged at generation {gen + 1}: {best_val:.6e}")
                break

            for i in range(self.N_current):
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
                    new_fitness.append(trial_fitness)
                    # 记录成功参数和适应度改进量
                    S_F.append(F)
                    S_CR.append(CR)
                    S_weights.append(self.fitness[i] - trial_fitness)
                    # 更新存档
                    self.archive.append(self.pop[i].copy())
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])

            # --- LPSR关键步骤 ---
            # 更新种群大小
            new_N = self._linear_pop_size_reduction(gen)

            # 选择适应度最好的个体保留
            combined_fitness = np.array(new_fitness)
            survivor_indices = np.argsort(combined_fitness)[:new_N]

            # 更新种群和适应度
            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.N_current = new_N

            # 同步缩减存档大小
            self.resize_archive()

            # 更新历史记忆
            if S_F:
                total_weight = np.sum(S_weights)
                if total_weight > 0:
                    F_lemer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                    CR_mean = np.sum(np.array(S_CR) * S_weights) / total_weight
                    self.F_memory[self.hist_idx] = F_lemer
                    self.CR_memory[self.hist_idx] = CR_mean
                    self.hist_idx = (self.hist_idx + 1) % self.H

            print(f"Iteration {gen + 1}, Pop Size: {self.N_current}, Best: {np.min(self.fitness)}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log
