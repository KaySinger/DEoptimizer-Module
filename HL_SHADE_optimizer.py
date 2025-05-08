import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import cma


class HLSHADE:
    def __init__(self, func, bounds, pop_size=100, max_gen=1000, H=100,
                 tol=1e-6, N_min=18, Q=64, p=0.11, Ar=2.6, cw=0.3, erw=0.2):
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

        self.F_memory = [0.5] * H
        self.CR_memory = [0.5] * H
        self.hist_idx = 0

        self.N_current = self.N_init
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.SP = []
        self.FC = np.zeros(self.N_init)
        self.C = np.eye(self.dim)
        self.iteration_log = []

        # 控制参数
        self.rho1 = 0
        self.rho1_max = 20
        self.rho2 = 0
        self.rho2_max = 10
        self.ls_eval = 0.01
        self.Bound_init = 0.5
        self.Bound_min = 0.1

    def _linear_pop_size_reduction(self, gen):
        return max(
            self.N_min,
            int(round(
                self.N_init - (self.N_init - self.N_min) * gen / self.max_gen
            ))
        )

    def _repair(self, mutant, parent):
        repaired = np.copy(mutant)
        for j in range(self.dim):
            low, high = self.bounds[j]
            if repaired[j] < low:
                repaired[j] = (parent[j] + low) / 2
            elif repaired[j] > high:
                repaired[j] = (parent[j] + high) / 2
        return repaired

    def _eig_crossover(self, parent, mutant, CR):
        try:
            eigvals, eigvecs = eigh(self.C)
        except:
            eigvecs = np.eye(self.dim)

        parent_eig = eigvecs.T @ parent
        mutant_eig = eigvecs.T @ mutant
        mask = (np.random.rand(self.dim) < CR) | (np.arange(self.dim) == np.random.randint(self.dim))
        trial_eig = np.where(mask, mutant_eig, parent_eig)
        return eigvecs @ trial_eig

    def _mutation_SPS(self, i, F):
        combined = np.vstack([self.pop, self.archive]) if self.archive else self.pop
        p_best_size = max(2, int(self.N_current * self.p))
        p_best_idx = np.random.choice(np.argsort(self.fitness)[:p_best_size])
        p_best = self.pop[p_best_idx]

        r1 = np.random.randint(self.N_current)
        while r1 == i:
            r1 = np.random.randint(self.N_current)

        r2 = np.random.randint(len(combined))
        while r2 == i or r2 == r1:
            r2 = np.random.randint(len(combined))

        if self.FC[i] <= self.Q:
            base = self.pop[i]
            donor1 = p_best
            donor2 = self.pop[r1] - combined[r2]
        else:
            sp_combined = np.vstack([self.SP, self.archive]) if self.SP else self.pop
            base = self.SP[i % len(self.SP)] if self.SP else self.pop[i]
            donor1 = self.SP[p_best_idx % len(self.SP)] if self.SP else p_best
            donor2 = sp_combined[r1 % len(sp_combined)] - sp_combined[r2 % len(sp_combined)]

        mutant = base + F * (donor1 - base) + F * donor2
        return self._repair(mutant, base)

    def _update_parameters(self, S_F, S_CR, S_weights):
        if S_F:
            total_weight = np.sum(S_weights)
            F_lemer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
            self.F_memory[self.hist_idx] = np.clip(F_lemer, 0.1, 1.0)
            CR_mean = np.sum(np.array(S_CR) * S_weights) / total_weight
            self.CR_memory[self.hist_idx] = np.clip(CR_mean, 0.05, 0.95)
            pop_mean = np.mean(self.pop, axis=0)
            diff = self.pop - pop_mean
            self.C = (1 - self.cw) * self.C + self.cw * (diff.T @ diff) / (self.N_current - 1)
            self.hist_idx = (self.hist_idx + 1) % self.H

    def _cma_es_search(self, x_best, gen):
        """CMA-ES局部搜索模块（修正版）"""
        # 1. 动态计算搜索范围
        Bound = (self.Bound_min - self.Bound_init) * gen / self.max_gen + self.Bound_init
        x_lb = np.maximum(self.bounds[:, 0], x_best - Bound * (self.bounds[:, 1] - self.bounds[:, 0]))
        x_ub = np.minimum(self.bounds[:, 1], x_best + Bound * (self.bounds[:, 1] - self.bounds[:, 0]))

        # 2. 边界有效性强制修正
        invalid = x_lb >= x_ub
        x_lb[invalid] = (self.bounds[invalid, 0] + self.bounds[invalid, 1]) / 2 - 1e-10
        x_ub[invalid] = (self.bounds[invalid, 0] + self.bounds[invalid, 1]) / 2 + 1e-10

        # 3. CMA-ES参数配置
        opts = {
            'bounds': [x_lb.tolist(), x_ub.tolist()],
            'verbose': -9,
            'popsize': max(5, 4 + int(3 * np.log(self.dim))),  # 保证最小种群数
            'maxfevals': int(self.ls_eval * self.max_gen),
            'CMA_stds': (x_ub - x_lb) / 6  # 自适应步长
        }

        # 4. 正确传递目标函数（关键修正点）
        try:
            es = cma.CMAEvolutionStrategy(
                (x_lb + x_ub).tolist(),  # 初始均值
                np.mean(x_ub - x_lb) / 4,  # 初始步长
                opts
            )

            # 优化时显式传递目标函数
            while not es.stop():
                solutions = es.ask()
                es.tell(solutions, [self.func(x) for x in solutions])  # 正确填充目标函数值

            return es.result.xbest, es.result.fbest

        except Exception as e:
            print(f"CMA-ES优化失败: {str(e)}")
            return x_best, self.func(x_best)

    def _interior_point_search(self, x0):
        # 调用内点法
        res = minimize(
            self.func,
            x0=np.clip(x0, self.bounds[:, 0], self.bounds[:, 1]),
            method='trust-constr',
            bounds=self.bounds,
            options={'maxiter': 50, 'disp': False}  # 限制迭代次数
        )

        # 返回优化结果
        return res.x, res.fun

    def optimize(self):
        for gen in range(self.max_gen):
            S_F, S_CR, S_weights = [], [], []
            new_pop, new_fitness = [], []
            ER = np.zeros(self.N_current)

            for i in range(self.N_current):
                r = np.random.randint(self.H)
                ER[i] = np.clip(self.CR_memory[r] + self.erw * np.random.randn(), 0, 1)

            for i in range(self.N_current):
                r = np.random.randint(self.H)
                F = np.clip(np.random.standard_cauchy() * 0.1 + self.F_memory[r], 0.1, 1.0)
                CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0.05, 0.95)
                mutant = self._mutation_SPS(i, F)

                if np.random.rand() < ER[i]:
                    trial = self._eig_crossover(self.pop[i], mutant, CR)
                else:
                    mask = (np.random.rand(self.dim) < CR) | (np.arange(self.dim) == np.random.randint(self.dim))
                    trial = np.where(mask, mutant, self.pop[i])

                trial = self._repair(trial, self.pop[i])
                trial_fitness = self.func(trial)

                if trial_fitness < self.fitness[i]:
                    if len(self.SP) < self.Ar * self.N_current:
                        self.SP.append(self.pop[i].copy())
                    else:
                        self.SP[np.random.randint(len(self.SP))] = self.pop[i].copy()
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

            new_N = self._linear_pop_size_reduction(gen)
            sorted_idx = np.argsort(new_fitness)
            survivor_idx = sorted_idx[:new_N]

            self.pop = np.array([new_pop[i] for i in survivor_idx])
            self.fitness = np.array([new_fitness[i] for i in survivor_idx])
            self.N_current = new_N
            self.archive = [new_pop[i].copy() for i in sorted_idx[new_N:]]
            self._update_parameters(S_F, S_CR, S_weights)

            best_idx = np.argmin(self.fitness)
            current_best = self.pop[best_idx]
            current_fit = self.fitness[best_idx]

            if gen == 0:
                self.rho1 = 0
                self.rho2 = 0
            elif current_fit < self.iteration_log[-1] if self.iteration_log else np.inf:
                self.rho1 = 0
                self.rho1_max = min(self.rho1_max + 5, 30)
            else:
                self.rho1 += 1
                self.rho1_max = max(self.rho1_max - 5, 5)

            if self.rho1 >= self.rho1_max:
                x_cma, f_cma = self._cma_es_search(current_best, gen)
                if f_cma < current_fit:
                    current_best = x_cma.copy()
                    current_fit = f_cma
                    self.rho2 = 0
                    self.rho1_max = min(self.rho1_max + 5, 30)
                else:
                    self.rho2 += 1
                    self.rho1_max = max(self.rho1_max - 5, 5)

            if gen > int(self.max_gen * 0.75):
                x_ip, f_ip = self._interior_point_search(current_best)
                if f_ip < current_fit:
                    current_best = x_ip.copy()
                    current_fit = f_ip
                    self.ls_eval = min(self.ls_eval + 0.005, 0.02)
                else:
                    self.ls_eval = max(self.ls_eval - 0.005, 0.005)

            # 记录最优值
            best_fitness = np.min(self.fitness)
            self.iteration_log.append(best_fitness)
            print(f"Iteration {gen + 1}, Best: {best_fitness:.6f},ρ1={self.rho1}/{self.rho1_max}, ρ2={self.rho2}/{self.rho2_max}, Pop Size: {self.N_current}")

            # 收敛检查
            if self.tol is not None and best_fitness <= self.tol:
                print(f"Converged at generation {gen + 1}: {best_fitness:.6e}")
                break
            elif gen >= self.max_gen - 1:
                print(f"Converged at generation {gen + 1}: {best_fitness:.6e}")
                break

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log
