import numpy as np
from scipy.stats import cauchy, norm

class JSO:
    def optimize(self):
        """
        执行优化（基于 JADE，增加选择性反向学习）
        """
        for gen in range(self.max_gen):
            # 初始化新一代种群
            new_pop = np.zeros_like(self.pop)
            new_fitness = np.zeros(self.pop_size)

            # 用于更新参数的成功参数
            S_F = []
            S_CR = []
            S_fitness = []

            for i in range(self.pop_size):
                # 生成 F 和 CR
                F = cauchy.rvs(loc=self.mu_F, scale=0.1)
                while F <= 0:
                    F = cauchy.rvs(loc=self.mu_F, scale=0.1)
                if F > 1:
                    F = 1

                CR = norm.rvs(loc=self.mu_CR, scale=0.1)
                CR = np.clip(CR, 0, 1)

                # 选择性反向学习
                if np.random.rand() < 0.5:
                    trial = self.bounds[:, 0] + self.bounds[:, 1] - self.pop[i]
                else:
                    # 选择 p_best
                    p_best_size = max(2, int(self.pop_size * self.p))
                    p_best_indices = np.argsort(self.fitness)[:p_best_size]
                    p_best_idx = np.random.choice(p_best_indices)
                    p_best = self.pop[p_best_idx]

                    # 选择两个不同的个体
                    candidates = [idx for idx in range(self.pop_size) if idx != i]
                    a, b = self.pop[np.random.choice(candidates, 2, replace=False)]

                    # 变异
                    mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (a - b)
                    mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

                    # 交叉
                    cross_points = np.random.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, self.pop[i])

                # 评估
                trial_fitness = self.func(trial)
                if trial_fitness < self.fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = trial_fitness
                    self.archive.append(self.pop[i].copy())
                    S_F.append(F)
                    S_CR.append(CR)
                    S_fitness.append(abs(self.fitness[i] - trial_fitness))

                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial
                        self.best_fitness = trial_fitness
                else:
                    new_pop[i] = self.pop[i]
                    new_fitness[i] = self.fitness[i]

            # 更新种群
            self.pop = new_pop
            self.fitness = new_fitness

            # 更新参数
            if S_F:
                # 计算加权平均值
                weights = np.array(S_fitness) / np.sum(S_fitness)
                self.mu_F = (1 - self.c) * self.mu_F + self.c * np.sum(weights * np.array(S_F))
                self.mu_CR = (1 - self.c) * self.mu_CR + self.c * np.sum(weights * np.array(S_CR))

            # 检查收敛条件
            if self.best_fitness <= self.tol:
                print(f"Converged at generation {gen + 1} with fitness {self.best_fitness:.6e}")
                break

            # 打印当前最优值
            if (gen + 1) % 100 == 0:
                print(f"Generation {gen + 1}, Best Fitness: {self.best_fitness:.6e}")

        return self.best_solution, self.best_fitness