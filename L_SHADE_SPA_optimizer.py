import numpy as np

class L_SHADE_SPA:
    def __init__(self, func, bounds, pop_size=100, max_gen=1000, H=100, tol=1e-6, N_min=18):
        """
        L-SHADE-SPA优化算法（带半参数自适应机制）

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
        self.M_F = [0.5] * H  # F的历史记忆
        self.M_Cr = [0.5] * H  # Cr的历史记忆
        self.freeze_flags = [False] * H  # 冻结标记位
        self.hist_idx = 0

        # 初始化种群和存档
        self.N_current = self.N_init
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.array([self.func(ind) for ind in self.pop])
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

    @staticmethod
    def _crossover(parent, mutant, Cr):
        """带强制交叉的二项交叉"""
        cross_mask = np.random.rand(parent.shape[0]) < Cr
        if not np.any(cross_mask):
            cross_mask[np.random.randint(parent.shape[0])] = True
        return np.where(cross_mask, mutant, parent)

    def optimize(self):
        """执行优化过程"""
        for gen in range(self.max_gen):
            S_F, S_Cr, S_weights = [], [], []
            new_pop = []
            new_fitness = []

            # 记录当前最优值
            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            if best_val <= self.tol:
                print(f"Converged at generation {gen} with precision {best_val:.6e}")
                break

            # ======== SPA阶段判断 ========
            is_first_phase = (gen < self.max_gen / 2)

            for i in range(self.N_current):
                # 从历史记忆随机选择索引
                r = np.random.randint(0, self.H)

                # ======== 参数生成 ========
                # --- F生成规则 ---
                if is_first_phase:
                    F = 0.45 + 0.1 * np.random.rand()  # 式(7)
                else:
                    # 从Cauchy分布采样 式(9)
                    F = np.clip(np.random.standard_cauchy() * 0.1 + self.M_F[r], 0, 1)

                # --- Cr生成规则 ---
                if self.freeze_flags[r]:
                    Cr = self.M_Cr[r]  # 冻结时直接使用存储值
                else:
                    Cr = np.clip(np.random.normal(self.M_Cr[r], 0.1), 0, 1)  # 式(8)

                # ======== 变异操作 ========
                # current-to-pbest/1变异策略
                p_min = max(2 / self.N_current, 0.05)
                p_i = np.random.uniform(p_min, 0.2)
                p_best_size = max(2, int(self.N_current * p_i))

                # 选择p_best
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
                    selected = np.random.choice(self.N_current, 2, replace=False)
                    a, b = self.pop[selected[0]], self.pop[selected[1]]

                # 生成变异向量
                mutant = self._repair(
                    self.pop[i] + F * (p_best - self.pop[i]) + F * (a - b),
                    self.pop[i],
                    self.bounds
                )

                # ======== 交叉操作 ========
                trial = self._crossover(self.pop[i], mutant, Cr)
                trial_fitness = self.func(trial)

                # ======== 选择操作 ========
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)

                    # 记录成功参数（根据阶段）
                    if is_first_phase:
                        S_Cr.append(Cr)  # 前半阶段只存Cr
                    else:
                        S_F.append(F)  # 后半阶段存F和Cr
                        S_Cr.append(Cr)

                    S_weights.append(self.fitness[i] - trial_fitness)

                    # 更新存档
                    self.archive.append(self.pop[i].copy())
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])

            # ======== 种群缩减 ========
            new_N = self._linear_pop_size_reduction(gen)
            combined_fitness = np.array(new_fitness)
            survivor_indices = np.argsort(combined_fitness)[:new_N]

            # 更新种群和适应度
            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.N_current = new_N

            # 同步缩减存档大小
            if len(self.archive) > self.N_current:
                self.archive = self.archive[:self.N_current]

            # ======== 历史内存更新 ========
            if S_weights:
                total_weight = np.sum(S_weights)

                # --- 前半阶段：只更新Cr内存 ---
                if is_first_phase:
                    if S_Cr:
                        Cr_mean = np.sum(np.array(S_Cr) * S_weights) / total_weight
                        self.M_Cr[self.hist_idx] = Cr_mean
                        self.hist_idx = (self.hist_idx + 1) % self.H

                # --- 后半阶段：更新F内存，条件更新Cr ---
                else:
                    # 更新F内存（Lehmer均值）
                    if S_F:  # 确保S_F不为空
                        F_lemer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                        self.M_F[self.hist_idx] = F_lemer

                    # 条件更新Cr内存
                    if S_Cr:
                        Cr_mean = np.sum(np.array(S_Cr) * S_weights) / total_weight
                        self.M_Cr[self.hist_idx] = Cr_mean
                    else:
                        # 如果该代所有Cr失败，标记冻结
                        self.freeze_flags[self.hist_idx] = True

                    self.hist_idx = (self.hist_idx + 1) % self.H

            # ======== 后半阶段初始化M_F ========
            if gen == int(self.max_gen / 2) - 1:
                # 取前5代的F值初始化M_F（论文描述）
                valid_F = [F for F in self.M_F if F is not None][-5 * self.H // self.max_gen:]
                self.M_F = valid_F + [0.5] * (self.H - len(valid_F))

            print(f"Gen {gen + 1}/{self.max_gen}, Pop: {self.N_current}, Best: {np.min(self.fitness):.6f}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log