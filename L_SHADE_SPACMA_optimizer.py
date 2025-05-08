import numpy as np

class L_SHADE_SPACMA:
    def __init__(self, func, bounds, pop_size=100, max_gen=1000, H=100, tol=1e-6, N_min=18):
        """
        L-SHADE-SPACMA优化算法类

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

        # 混合算法参数
        self.FCP_memory = [0.5] * H  # First Class Probability (LSHADE分配概率)
        self.L_rate = 0.8  # 学习率
        self.Hybridization_flag = True

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

        # CMA-ES参数初始化
        self.sigma = 0.5
        self.xmean = np.mean(self.pop, axis=0)
        self.xmean_old = self.xmean.copy()
        self.mu = self.N_current // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights ** 2)

        # 协方差矩阵参数
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        # 协方差矩阵
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = self.B @ np.diag(self.D ** 2) @ self.B.T
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.eigeneval = 0
        self.chiN = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

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

    def generate_mutant(self, i, F, CR):
        """生成变异个体（包含CMA-ES混合逻辑）"""
        # 选择算法类别
        r = np.random.randint(0, self.H)
        FCP = self.FCP_memory[r]
        use_lshade = np.random.rand() < FCP

        if use_lshade:
            # LSHADE变异策略
            p_min = max(2 / self.N_current, 0.05)
            p_i = np.random.uniform(p_min, 0.2)
            p_best_size = max(2, int(self.N_current * p_i))

            p_best_indices = np.argsort(self.fitness)[:p_best_size]
            p_best_idx = np.random.choice(p_best_indices)
            p_best = self.pop[p_best_idx]

            # 选择a和b
            a = self.pop[np.random.choice([x for x in range(self.N_current) if x != i])]
            combined_pop = np.vstack([self.pop, self.archive]) if self.archive else self.pop
            b = combined_pop[np.random.choice(len(combined_pop))]

            mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (a - b)
        else:
            # CMA-ES变异策略
            z = self.B @ (self.D * np.random.randn(self.dim))
            mutant = self.xmean + self.sigma * z

        return self._repair(mutant, self.pop[i], self.bounds)

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
            delta_alg1, delta_alg2 = [], []
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
                # 参数生成阶段
                r = np.random.randint(0, self.H)

                # SPA机制：前半段固定F范围，后半段自适应
                if gen < self.max_gen / 2:
                    F = np.clip(0.45 + 0.1 * np.random.rand(), 0.4, 0.5)
                else:
                    F = np.clip(np.random.standard_cauchy() * 0.1 + self.F_memory[r], 0, 1)

                CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0, 1)

                # 生成变异个体
                mutant = self.generate_mutant(i, F, CR)
                trial = self._crossover(self.pop[i], mutant, CR)
                trial_fitness = self.func(trial)

                # 选择操作
                if trial_fitness < self.fitness[i]:
                    success = True
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                    S_F.append(F)
                    S_CR.append(CR)
                    S_weights.append(self.fitness[i] - trial_fitness)
                    self.archive.append(self.pop[i].copy())
                else:
                    success = False
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])

                # 记录算法表现
                r = np.random.randint(0, self.H)
                if success:
                    if gen < self.max_gen / 2:
                        delta_alg1.append(self.fitness[i] - trial_fitness)
                    else:
                        delta_alg2.append(self.fitness[i] - trial_fitness)

            # 更新历史记忆
            if S_F:
                total_weight = np.sum(S_weights)
                F_lemer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                CR_mean = np.sum(np.array(S_CR) * S_weights) / total_weight
                self.F_memory[self.hist_idx] = F_lemer
                self.CR_memory[self.hist_idx] = CR_mean

                # 更新混合概率
                if delta_alg1 and delta_alg2:
                    ratio = np.sum(delta_alg1) / (np.sum(delta_alg1) + np.sum(delta_alg2) + 1e-50)
                    self.FCP_memory[self.hist_idx] = self.L_rate * self.FCP_memory[self.hist_idx] + (
                                1 - self.L_rate) * ratio
                    self.FCP_memory[self.hist_idx] = np.clip(self.FCP_memory[self.hist_idx], 0.2, 0.8)

                self.hist_idx = (self.hist_idx + 1) % self.H

                # 更新CMA-ES参数
            if self.Hybridization_flag and len(new_pop) > 1:
                self.xmean_old = self.xmean.copy()  # 先保存旧值
                sorted_indices = np.argsort(new_fitness)
                selected = sorted_indices[:self.mu]

                # 更新均值
                # 检查维度一致性
                if len(selected) != len(self.weights):
                    raise ValueError(
                        f"权重维度({len(self.weights)})与选中个体数({len(selected)})不匹配"
                    )

                # 加权均值计算（确保矩阵乘法正确）
                selected_pop = np.array([new_pop[i] for i in selected])
                self.xmean = np.dot(self.weights, selected_pop)

                # 更新进化路径
                y = (self.xmean - self.xmean_old) / self.sigma
                self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.B @ y)

                # 更新协方差矩阵
                self.C = (1 - self.c1 - self.cmu) * self.C + \
                         self.c1 * np.outer(self.pc, self.pc) + \
                         self.cmu * np.dot((np.array(new_pop)[selected] - self.xmean_old).T,
                                           self.weights[:, None] * (np.array(new_pop)[selected] - self.xmean_old))

                # 更新步长
                self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

                # 特征分解（定期更新）
                if gen % 100 == 0:
                    self.C = np.triu(self.C) + np.triu(self.C, 1).T
                    D, B = np.linalg.eigh(self.C)
                    self.D = np.sqrt(np.abs(D))
                    self.B = B

            # 种群缩减
            new_N = self._linear_pop_size_reduction(gen)
            survivor_indices = np.argsort(new_fitness)[:new_N]
            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.N_current = new_N

            self.mu = max(1, self.N_current // 2)  # 防止mu为0
            self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
            self.weights /= np.sum(self.weights)  # 归一化权重
            self.mueff = 1 / np.sum(self.weights ** 2)  # 更新有效种群大小

            self.resize_archive()

            print(f"Iteration {gen + 1}, Pop Size: {self.N_current}, Best: {np.min(self.fitness):.4e}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log
