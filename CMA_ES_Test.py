import copy
import numpy as np

class CMAES_algorithm:
    def __init__(self, func, max_iter_num, pop_size, n_dim, lb, ub, sigma=1, xmean=None, mu=None, B=None, D=None, norm_ord=2):
        self.fitness_func = func
        self.max_iter_num = max_iter_num
        self.lambda_pop = pop_size
        self.mu = int(self.lambda_pop / 2) if (not mu) else mu
        self.n_dim = n_dim
        self.N = n_dim
        self.lb, self.ub = lb, ub
        self.sigma = sigma
        self.xmean = (ub - lb) * (np.random.rand(n_dim, 1) - 0.5) if not xmean else xmean
        self.weights = (np.log(self.mu + 1 / 2) + np.log(range(1, self.mu + 1, 1))) / \
                       np.sum((np.log(self.mu + 1 / 2) + np.log(range(1, self.mu + 1, 1))))
        self.mueff = (np.sum(self.weights) ** 2) / (np.sum(self.weights ** 2))
        self.cc, self.cs, self.c1, self.cmu, self.damps = self.set_adapt_para()
        self.norm_ord = norm_ord
        self.pc, self.ps = np.zeros((n_dim, 1)), np.zeros((n_dim, 1))
        self.B = np.eye(n_dim) if not B else B
        self.D = np.eye(n_dim) if not D else D
        self.BD = np.matmul(self.B, self.D)
        self.C = np.matmul(self.BD, (self.BD).T)
        # E||N(0,I)|| = norm(randn(N,1))
        self.chiN = (n_dim ** 0.5) * (1 - (1 / (4 * n_dim)) + (1 / (21 * n_dim ** 2)))
        self.arfitness_opt_globa = []
        self.locat_opt_global = []
        self.arx_output = []
        self.countevel = 0
        self.eigeneval = 0

    def set_adapt_para(self):
        cc = (4 + self.mueff / self.N) / (self.N + 4 + 2 * self.mueff / self.N)  # 时间常数
        cs = (self.mueff + 2) / (self.N + self.mueff + 5)  # 用于控制sigma
        c1 = 2 / ((self.N + 1.3) ** 2 + self.mueff)  # 协方差矩阵秩一更新的学习率
        cmu = 2 * (self.mueff - 2 + (1 / self.mueff)) / (((self.N + 2) ** 2) + 2 * self.mueff / 2)  # 协方差矩阵秩mu更新的学习率
        damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.N + 1)) - 1)  # sigma的阻尼系数
        return cc, cs, c1, cmu, damps

    def clip_fun(self, X, lb, ub):
        # 限制区间
        XX = copy.deepcopy(X)
        for i in range(len(X)):
            X[i, :] = np.clip(XX[i, :], lb, ub)
        return X

    def CMA_ES_sloveFun(self):
        self.BD = np.matmul(self.B, self.D)
        arz = np.zeros((self.N, self.lambda_pop))
        arx = np.zeros((self.N, self.lambda_pop))
        arfitness = np.zeros(self.lambda_pop)

        # 从分布中进行采样
        for i in range(self.lambda_pop):
            arz[:, i] = np.random.randn(self.N)  # 生成符合正态分布的随机向量
            arx[:, i] = (self.xmean + self.sigma * np.matmul(self.BD, arz[:, i].reshape(-1, 1))).T[0]  # [0]是用于从 [[x1,x2,...,xn]] 中取值
            arx = self.clip_fun(arx, self.lb, self.ub)
            arfitness[i] = self.fitness_func(arx[:, i])
        self.countevel += self.lambda_pop

        # 对适应度函数进行排序并计算加权均值
        arindex = np.argsort(arfitness)
        arfitness = np.sort(arfitness)  # 选择最优的mu个点
        print(f"arindex[:10] = {arindex[:10]}")
        print(f"arfitness[:10] = {arfitness[:10]}")

        self.arfitness_opt_globa.append(arfitness[0])
        self.locat_opt_global.append(arx[:, arindex[0]])
        arx_select = np.zeros((self.N, self.mu))
        arz_select = np.zeros((self.N, self.mu))
        for i in range(self.mu):
            arx_select[:, i] = arx[:, arindex[i]]
            arz_select[:, i] = arz[:, arindex[i]]
        self.xmean = np.matmul(arx_select, self.weights.T)  # 重组，计算均值
        zmean = np.matmul(arz_select, self.weights.T)

        # 累积并计算进化路径
        norm_ps = np.linalg.norm(x=self.ps, ord=self.norm_ord)
        self.ps = (1 - self.cs) * self.ps + (np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.matmul(self.B, zmean)).reshape(-1, 1)
        hsig_cal1 = norm_ps / (np.sqrt(1 - (1 - self.cs) ** (self.countevel / self.lambda_pop)))
        hsig_cal2 = (1.4 + 2 / (self.N + 1)) * self.chiN
        hsig = 1 if hsig_cal1 < hsig_cal2 else 0
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * np.matmul(self.BD, zmean)

        # 自适应更新协方差矩阵
        BD_arz = np.matmul(self.BD, arz_select)
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.matmul(self.pc, (self.pc).T) + (1 - hsig) * (2 - self.cc) * self.C) + self.cmu * np.matmul(BD_arz, (np.matmul(np.diag(self.weights), BD_arz.T)))

        # 自适应更新步长sigma
        self.sigma = self.sigma * np.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1))
        # 由协方差矩阵C更新B和D
        if ((self.countevel - self.eigeneval) > (self.lambda_pop / (4 + self.cmu) / self.N / 10)):
            self.eigeneval = self.countevel
            self.C = np.triu(self.C, k=0) + np.triu(self.C, k=1).T  # 强制对称
            eigenvalues, eigenvectors = np.linalg.eig(self.C)
            self.B, self.D = eigenvectors, np.diag(np.sqrt(eigenvalues))

        # 离开平稳点
        if arfitness[0] == arfitness[int(np.ceil(0.7 * self.lambda_pop))]:
            # if np.abs(arfitness[0] - arfitness[int(np.ceil(0.7*lambda_pop))])<10:
            self.sigma = self.sigma * np.exp(0.2 + self.cs / self.damps)
            print(f"warning ,sigma = {self.sigma}")
            print(f"warning: flat fitness, consider reformulating the objective!")

        # 对本轮演化得到的分布进行采样，并返回arx_temp
        BD_temp = np.matmul(self.B, self.D)
        arz_temp = np.zeros((self.N, self.lambda_pop))
        arx_temp = np.zeros((self.N, self.lambda_pop))
        for i in range(self.lambda_pop):
            arz_temp[:, i] = np.random.randn(self.N)  # 生成符合正态分布的随机向量
            arx_temp[:, i] = (
                        self.xmean.reshape(1, -1)[0] + self.sigma * np.matmul(BD_temp, arz_temp[:, i]).reshape(-1, 1)[0])
            arx_temp = self.clip_fun(arx_temp, self.lb, self.ub)
        self.arx_output.append(arx_temp)

    def Slove(self, ):
        for i in range(self.max_iter_num):
            self.CMA_ES_sloveFun()
        return self.arfitness_opt_globa