import numpy as np
from scipy.linalg import sqrtm

class L_SHADE_SPACMA:
    def __init__(self, func, bounds, pop_size=100, max_gen=1000, H=5, tol=1e-8, N_min=4):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.N_init = pop_size
        self.N_min = N_min
        self.max_gen = max_gen
        self.H = H
        self.tol = tol
        self.gen = 0

        # LSHADE-SPA parameters
        self.M_F = [0.5] * H
        self.M_Cr = [0.5] * H
        self.M_FCP = [0.5] * H
        self.freeze_flags = [False] * H
        self.hist_idx = 0
        self.archive = []

        # CMA-ES parameters
        self.cma_mean = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])
        self.cma_sigma = 0.3 * (bounds[:, 1] - bounds[:, 0]).mean()
        self.cma_C = np.eye(self.dim)
        self.cma_pc = np.zeros(self.dim)
        self.cma_ps = np.zeros(self.dim)

        # Dynamic parameters
        self.FCP = 0.5
        self.c = 0.8

        # Initialize population
        self.N_current = self.N_init
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim))
        self.fitness = np.array([self.func(ind) for ind in self.pop])
        self.iteration_log = []
        self.bsf_fit_var = np.inf
        self.bsf_solution = None

    def _update_archive(self, new_pop, new_fitness):
        self.archive.extend(list(zip(new_pop, new_fitness)))
        self.archive.sort(key=lambda x: x[1])
        self.archive = self.archive[:self.N_current]

    def _bound_constraint(self, vi, pop):
        for i in range(vi.shape[0]):
            for j in range(self.dim):
                if vi[i, j] < self.bounds[j, 0]:
                    vi[i, j] = (pop[i, j] + self.bounds[j, 0]) / 2
                elif vi[i, j] > self.bounds[j, 1]:
                    vi[i, j] = (pop[i, j] + self.bounds[j, 1]) / 2
        return vi

    def _cma_es_generate(self, count):
        candidates = []
        sqrtC = sqrtm(self.cma_C)
        for _ in range(count):
            z = np.random.randn(self.dim)
            x = self.cma_mean + self.cma_sigma * (sqrtC @ z)
            j_rand = np.random.randint(self.dim)
            mask = np.random.rand(self.dim) < 0.5
            mask[j_rand] = True
            base = self.pop[np.random.randint(self.N_current)]
            x = np.where(mask, x, base)
            candidates.append(np.clip(x, self.bounds[:, 0], self.bounds[:, 1]))
        return np.array(candidates)

    def _lshade_spa_generate(self, count):
        solutions = []
        S_F, S_Cr = [], []
        is_first_phase = (self.gen < self.max_gen / 2)

        for i in range(count):
            r = np.random.randint(0, self.H)

            if is_first_phase:
                F = 0.45 + 0.1 * np.random.rand()
            else:
                F = np.clip(np.random.standard_cauchy() * 0.1 + self.M_F[r], 0, 1)

            if self.freeze_flags[r]:
                Cr = self.M_Cr[r]
            else:
                Cr = np.clip(np.random.normal(self.M_Cr[r], 0.1), 0, 1)

            p_best_size = max(2, int(count * np.random.uniform(0.05, 0.2)))
            p_best_idx = np.random.choice(np.argsort(self.fitness)[:p_best_size])
            p_best = self.pop[p_best_idx]

            candidates = [idx for idx in range(len(self.pop))
                          if idx != i and not np.array_equal(self.pop[idx], p_best)]
            if len(candidates) >= 2:
                a, b = self.pop[np.random.choice(candidates, 2, replace=False)]
            else:
                a, b = self.pop[np.random.choice(len(self.pop), 2, replace=False)]

            mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (a - b)
            mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

            trial = np.copy(self.pop[i])
            cross_points = np.random.rand(self.dim) < Cr
            if not np.any(cross_points):
                cross_points[np.random.randint(self.dim)] = True
            trial[cross_points] = mutant[cross_points]

            solutions.append(trial)

            if is_first_phase:
                S_Cr.append(Cr)
            else:
                S_F.append(F)
                S_Cr.append(Cr)

        if S_Cr:
            weights = np.ones(len(S_Cr))

            if is_first_phase:
                Cr_mean = np.sum(np.array(S_Cr) * weights) / np.sum(weights)
                self.M_Cr[self.hist_idx] = Cr_mean
            else:
                if S_F:
                    F_lemer = np.sum(np.array(S_F) ** 2 * weights) / np.sum(np.array(S_F) * weights)
                    self.M_F[self.hist_idx] = F_lemer

                Cr_mean = np.sum(np.array(S_Cr) * weights) / np.sum(weights)
                self.M_Cr[self.hist_idx] = Cr_mean

                if len(S_Cr) < count * 0.1:
                    self.freeze_flags[self.hist_idx] = True

            self.hist_idx = (self.hist_idx + 1) % self.H

        return np.array(solutions)

    def _update_parameters(self, solutions, fitness):
        # Calculate improvements with fixed size arrays
        N_lshade = int(self.FCP * self.N_current)
        N_cma = self.N_current - N_lshade

        # Ensure we don't exceed array bounds
        N_lshade = min(N_lshade, len(self.fitness), len(fitness))
        N_cma = min(N_cma, len(self.fitness) - N_lshade, len(fitness) - N_lshade)

        lshade_improve = max(0, np.sum(self.fitness[:N_lshade] - fitness[:N_lshade]))
        cma_improve = max(0, np.sum(self.fitness[N_lshade:N_lshade + N_cma] - fitness[N_lshade:N_lshade + N_cma]))

        # Update FCP memory
        if lshade_improve + cma_improve > 0:
            new_FCP = max(0.2, min(0.8, lshade_improve / (lshade_improve + cma_improve)))
            self.M_FCP[self.hist_idx] = (1 - self.c) * self.M_FCP[self.hist_idx] + self.c * new_FCP
            self.FCP = self.M_FCP[self.hist_idx]

        # Update CMA-ES parameters if used
        if self.FCP < 1.0 and N_cma > 0:
            cma_start = N_lshade
            cma_end = min(N_lshade + N_cma, len(fitness))
            best_indices = np.argsort(fitness[cma_start:cma_end])[:5] + cma_start

            if len(best_indices) > 0:
                weights = np.array([0.4, 0.3, 0.2, 0.05, 0.05][:len(best_indices)])
                weights = weights / np.sum(weights)  # Normalize

                # CMA-ES parameter updates
                old_mean = self.cma_mean.copy()
                self.cma_mean = np.sum(weights[:, None] * solutions[best_indices], axis=0)

                y = (self.cma_mean - old_mean) / self.cma_sigma
                self.cma_ps = 0.8 * self.cma_ps + np.sqrt(0.2 * 2) * (sqrtm(np.linalg.inv(self.cma_C)) @ y)

                rank_mu_update = np.zeros((self.dim, self.dim))
                for i, idx in enumerate(best_indices):
                    z = (solutions[idx] - old_mean) / self.cma_sigma
                    rank_mu_update += weights[i] * np.outer(z, z)

                self.cma_C = 0.9 * self.cma_C + 0.1 * rank_mu_update
                self.cma_sigma *= np.exp(0.1 * (np.linalg.norm(self.cma_ps) / np.sqrt(self.dim) - 1))

    def optimize(self):
        for gen in range(self.max_gen):
            self.gen = gen
            N_lshade = int(round(self.FCP * self.N_current))
            N_cma = max(0, self.N_current - N_lshade)  # Ensure non-negative

            # Generate solutions
            lshade_solutions = self._lshade_spa_generate(N_lshade)
            cma_solutions = self._cma_es_generate(N_cma) if N_cma > 0 else np.zeros((0, self.dim))
            all_solutions = np.vstack([lshade_solutions, cma_solutions])

            # Evaluate
            all_fitness = np.array([self.func(ind) for ind in all_solutions])

            # Update best
            min_idx = np.argmin(all_fitness)
            current_best = all_fitness[min_idx]
            if current_best < self.bsf_fit_var:
                self.bsf_fit_var = current_best
                self.bsf_solution = all_solutions[min_idx]

            # Print progress
            print(f"Iter {gen + 1}/{self.max_gen} | Pop: {self.N_current} | Best: {self.bsf_fit_var:.6e}")

            # Check convergence
            if self.bsf_fit_var <= self.tol:
                print(f"Converged at generation {gen + 1} with precision {self.bsf_fit_var:.6e}")
                break

            # Update parameters
            self._update_archive(all_solutions, all_fitness)
            self._update_parameters(all_solutions, all_fitness)

            # Population reduction
            self.N_current = max(self.N_min,
                                 int(self.N_init - (self.N_init - self.N_min) * gen / self.max_gen))

            # Update population
            self.pop = all_solutions
            self.fitness = all_fitness
            self.iteration_log.append(self.bsf_fit_var)

        return self.bsf_solution, self.bsf_fit_var, self.iteration_log