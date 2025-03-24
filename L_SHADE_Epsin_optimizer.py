import numpy as np
from scipy.stats import cauchy

class L_SHADE_Epsin:
    def __init__(self, func, bounds, pop_size=None, max_gen=None, H=None, tol=1e-6, N_min=4):
        """
        L-SHADE-Epsin优化算法类

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
        self.freq_memory = [0.5] * H  # 频率历史记忆
        self.local_search_triggered = False  # 局部搜索触发标志
        self.G_LS = 250  # 局部搜索代数
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

    @staticmethod
    def _crossover(parent, mutant, CR):
        """带强制交叉的二项交叉"""
        cross_mask = np.random.rand(parent.shape[0]) < CR
        if not np.any(cross_mask):
            cross_mask[np.random.randint(parent.shape[0])] = True
        return np.where(cross_mask, mutant, parent)

    def _gaussian_walk_search(self):
        """高斯游走局部搜索"""
        best_idx = np.argmin(self.fitness)
        x_best = self.pop[best_idx]

        # 初始化10个随机个体
        local_pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(10, self.dim)
        )
        local_fitness = np.apply_along_axis(self.func, 1, local_pop)

        # 进行G_LS代搜索
        for g in range(self.G_LS):
            for i in range(10):
                # 计算自适应标准差
                sigma = np.abs((np.log(g + 1) / (g + 1) * (local_pop[i] - x_best)))
                # 高斯扰动+随机偏移
                trial = x_best + np.random.normal(0, sigma) + (np.random.rand() * x_best - np.random.rand() * local_pop[i])
                trial = np.clip(trial, self.bounds[:, 0], self.bounds[:, 1])
                trial_fitness = self.func(trial)

                if trial_fitness < local_fitness[i]:
                    local_pop[i] = trial
                local_fitness[i] = trial_fitness

        # 替换原种群最差的10个个体
        worst_indices = np.argsort(self.fitness)[-10:]
        for i in range(10):
            if local_fitness[i] < self.fitness[worst_indices[i]]:
                self.pop[worst_indices[i]] = local_pop[i]
                self.fitness[worst_indices[i]] = local_fitness[i]

    def optimize(self):
        """执行优化过程"""
        for gen in range(self.max_gen):
            S_F, S_CR, S_freq, S_weights = [], [], [], []
            new_pop = []
            new_fitness = []

            # 记录当前最优值
            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            if best_val <= self.tol:
                print(f"Converged at generation {gen} with precision {best_val:.6e}")
                break
 
            for i in range(self.N_current):
                r = np.random.randint(0, self.H)

                # 前半段使用混合正弦策略
                if gen < self.max_gen // 2:
                    # 随机选择正弦策略（50%概率）
                    if np.random.rand() < 0.5:
                        # 非自适应递减策略
                        freq = 0.5
                        F = 0.5 * (np.sin(2 * np.pi * freq * gen + np.pi) * ((self.max_gen - gen) / self.max_gen) + 1)
                    else:
                        # 自适应递增策略
                        freq = cauchy.rvs(loc=self.freq_memory[r], scale=0.1)
                        F = 0.5 * (np.sin(2 * np.pi * freq * gen) * (gen / self.max_gen) + 1)

                    # CR沿用原策略
                    CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0, 1)
                else:
                    # 后半段使用原L-SHADE策略
                    F = np.clip(cauchy.rvs(loc=self.F_memory[r], scale=0.1), 0, 1)
                    CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0, 1)

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
                    if not (np.array_equal(combined_pop[idx], self.pop[i]) or np.array_equal(combined_pop[idx], p_best)):
                        candidates.append(idx)

                # 处理候选不足的情况
                if len(candidates) >= 2:
                    selected = np.random.choice(candidates, 2, replace=False)
                    a, b = combined_pop[selected[0]], combined_pop[selected[1]]
                else:
                    selected = np.random.choice(self.N_current, 2, replace=False)
                    a, b = self.pop[selected[0]], self.pop[selected[1]]

                # 变异操作
                mutant = self._repair(
                    self.pop[i] + F * (p_best - self.pop[i]) + F * (a - b),
                    self.pop[i],
                    self.bounds
                )

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
                    if gen < self.max_gen // 2 and np.random.rand() < 0.5:
                        S_freq.append(freq)  # 仅记录自适应策略的成功频率
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
            if len(self.archive) > self.N_current:
                self.archive = self.archive[:self.N_current]

            if not self.local_search_triggered and self.N_current <= 20:
                self._gaussian_walk_search()
                self.local_search_triggered = True
                print(f"Trigger Gaussian Walk at gen {gen}")

            # 更新历史记忆
            if len(S_F) > 0 and len(S_CR) > 0:
                total_weight = np.sum(S_weights)
                if total_weight > 0:
                    F_lemer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                    CR_mean = np.sum(np.array(S_CR) * S_weights) / total_weight
                    self.F_memory[self.hist_idx] = F_lemer
                    self.CR_memory[self.hist_idx] = CR_mean
                    # 更新频率记忆（仅前半段）
                    if S_freq:
                        freq_mean = np.sum(np.array(S_freq) * S_weights[:len(S_freq)]) / total_weight
                        self.freq_memory[self.hist_idx] = freq_mean

                    # 更新索引
                    self.hist_idx += 1
                    if self.hist_idx >= self.H:
                        self.hist_idx = 1  # 对齐伪代码k > H时设为1

            print(f"Iteration {gen + 1}, Pop Size: {self.N_current}, Best fitness: {np.min(self.fitness)}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log