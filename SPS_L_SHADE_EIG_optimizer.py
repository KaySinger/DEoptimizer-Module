import numpy as np
from scipy.linalg import eigh


class SPS_L_SHADE_EIG:
    def __init__(self, func, bounds, pop_size=100, max_gen=1000, H=100,
                 tol=1e-6, N_min=18, Q=64, p=0.11, Ar=2.6, cw=0.3, erw=0.2):
        """
        SPS-L-SHADE-EIG 优化算法

        参数：
        func: 目标函数（最小化）
        bounds: 变量边界的列表，例如 [(min1, max1), (min2, max2), ...]
        pop_size: 初始种群大小
        max_gen: 最大迭代次数
        H: 历史记忆大小
        tol: 收敛精度
        N_min: 最小种群大小
        Q: 失败计数阈值
        p: pbest选择比例
        Ar: 存档比例因子
        cw: 协方差矩阵更新权重
        erw: EIG率扰动权重
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.N_init = pop_size
        self.N_min = N_min
        self.max_gen = max_gen
        self.H = H
        self.tol = tol
        self.Q = Q
        self.p = p
        self.Ar = Ar
        self.cw = cw
        self.erw = erw

        # 初始化历史记忆
        self.F_memory = [0.5] * H
        self.CR_memory = [0.5] * H
        self.hist_idx = 0

        # 初始化种群和存档
        self.N_current = self.N_init
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.SP = []  # 成功解存档
        self.FC = np.zeros(self.N_init)  # 失败计数器
        self.C = np.eye(self.dim)  # 协方差矩阵
        self.iteration_log = []

    def _linear_pop_size_reduction(self, gen):
        """线性种群缩减策略"""
        return max(
            self.N_min,
            int(round(
                self.N_init - (self.N_init - self.N_min) * gen / self.max_gen
            ))
        )

    def _repair(self, mutant, parent):
        """边界修复策略"""
        repaired = np.copy(mutant)
        for j in range(self.dim):
            low, high = self.bounds[j]
            if repaired[j] < low:
                repaired[j] = (parent[j] + low) / 2
            elif repaired[j] > high:
                repaired[j] = (parent[j] + high) / 2
        return repaired

    def _eig_crossover(self, parent, mutant, CR):
        """基于特征向量的EIG交叉"""
        # 计算协方差矩阵的特征分解
        try:
            eigvals, eigvecs = eigh(self.C)
        except:
            eigvecs = np.eye(self.dim)

        # 转换到特征空间
        parent_eig = eigvecs.T @ parent
        mutant_eig = eigvecs.T @ mutant

        # 执行交叉
        cross_mask = (np.random.rand(self.dim) < CR) | \
                     (np.arange(self.dim) == np.random.randint(self.dim))
        trial_eig = np.where(cross_mask, mutant_eig, parent_eig)

        # 转换回原始空间
        return eigvecs @ trial_eig

    def _mutation_SPS(self, i, F):
        """成功父代选择变异策略"""
        # 合并种群和存档
        combined = np.vstack([self.pop, self.archive]) if self.archive else self.pop

        # 选择pbest
        p_best_size = max(2, int(self.N_current * self.p))
        p_best_idx = np.random.choice(np.argsort(self.fitness)[:p_best_size])
        p_best = self.pop[p_best_idx]

        # 选择r1和r2
        r1 = np.random.randint(self.N_current)
        while r1 == i:
            r1 = np.random.randint(self.N_current)

        r2 = np.random.randint(len(combined))
        while r2 == i or r2 == r1:
            r2 = np.random.randint(len(combined))

        # 变异策略选择
        if self.FC[i] <= self.Q:
            base = self.pop[i]
            donor1 = p_best
            donor2 = self.pop[r1] - combined[r2]
        else:
            # 从成功存档SP选择
            sp_combined = np.vstack([self.SP, self.archive]) if self.SP else self.pop
            base = self.SP[i % len(self.SP)] if self.SP else self.pop[i]
            donor1 = self.SP[p_best_idx % len(self.SP)] if self.SP else p_best
            donor2 = sp_combined[r1 % len(sp_combined)] - sp_combined[r2 % len(sp_combined)]

        mutant = base + F * (donor1 - base) + F * donor2
        return self._repair(mutant, base)

    def _update_parameters(self, S_F, S_CR, S_weights):
        """更新历史记忆和协方差矩阵"""
        if S_F:
            total_weight = np.sum(S_weights)
            # 更新F记忆
            F_lemer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
            self.F_memory[self.hist_idx] = np.clip(F_lemer, 0.1, 1.0)

            # 更新CR记忆
            CR_mean = np.sum(np.array(S_CR) * S_weights) / total_weight
            self.CR_memory[self.hist_idx] = np.clip(CR_mean, 0.05, 0.95)

            # 更新协方差矩阵
            pop_mean = np.mean(self.pop, axis=0)
            diff = self.pop - pop_mean
            self.C = (1 - self.cw) * self.C + self.cw * (diff.T @ diff) / (self.N_current - 1)

            self.hist_idx = (self.hist_idx + 1) % self.H

    def optimize(self):
        """执行优化过程"""
        for gen in range(self.max_gen):
            S_F, S_CR, S_weights = [], [], []
            new_pop = []
            new_fitness = []
            ER = np.zeros(self.N_current)  # EIG应用概率

            # 生成EIG率
            for i in range(self.N_current):
                r = np.random.randint(self.H)
                ER[i] = np.clip(self.CR_memory[r] + self.erw * np.random.randn(), 0, 1)

            for i in range(self.N_current):
                # 选择历史参数
                r = np.random.randint(self.H)
                F = np.clip(np.random.standard_cauchy() * 0.1 + self.F_memory[r], 0.1, 1.0)
                CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0.05, 0.95)

                # SPS变异
                mutant = self._mutation_SPS(i, F)

                # 概率性应用EIG交叉
                if np.random.rand() < ER[i]:
                    trial = self._eig_crossover(self.pop[i], mutant, CR)
                else:
                    # 标准二项交叉
                    cross_mask = (np.random.rand(self.dim) < CR) | \
                                 (np.arange(self.dim) == np.random.randint(self.dim))
                    trial = np.where(cross_mask, mutant, self.pop[i])

                # 边界处理
                trial = self._repair(trial, self.pop[i])

                # 评估
                trial_fitness = self.func(trial)

                # 选择操作
                if trial_fitness < self.fitness[i]:
                    # 更新成功存档
                    if len(self.SP) < self.Ar * self.N_current:
                        self.SP.append(self.pop[i].copy())
                    else:
                        replace_idx = np.random.randint(len(self.SP))
                        self.SP[replace_idx] = self.pop[i].copy()

                    # 记录成功参数
                    S_F.append(F)
                    S_CR.append(CR)
                    S_weights.append(self.fitness[i] - trial_fitness)
                    self.FC[i] = 0
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])
                    self.FC[i] += 1

            # --- 种群缩减和更新 ---
            # 更新种群大小
            new_N = self._linear_pop_size_reduction(gen)

            # 选择存活个体
            sorted_indices = np.argsort(new_fitness)
            survivor_indices = sorted_indices[:new_N]

            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.N_current = new_N

            # 更新存档
            self.archive = [ind.copy() for idx, ind in enumerate(new_pop)
                            if idx not in survivor_indices][:self.N_current]

            # 更新参数
            self._update_parameters(S_F, S_CR, S_weights)

            # 记录最优值
            best_fitness = np.min(self.fitness)
            self.iteration_log.append(best_fitness)
            print(f"Iteration {gen + 1}, Best: {best_fitness:.6f}, Pop Size: {self.N_current}")

            # 收敛检查
            if self.tol is not None and best_fitness <= self.tol:
                print(f"Converged at generation {gen + 1}: {best_fitness:.6e}")
                break
            elif gen >= self.max_gen - 1:
                print(f"Converged at generation {gen + 1}: {best_fitness:.6e}")
                break

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log
