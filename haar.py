import numpy as np
import torch
import math
import matplotlib.pyplot as plt


def psi(x, j, k):
    return 2 ** (j / 2) * ((2 ** j * x - k >= 0) & (2 ** j * x - k < 0.5))*1 - \
           ((2 ** j * x - k >= 0.5) & (2 ** j * x - k < 1))


class Haar:
    def __init__(self, n_basis):
        self.level = int(math.log2(n_basis))
        self.n_basis = n_basis
        self.linear = False
        self.mat = None
        self.length = None
        self.pen2 = np.zeros([self.n_basis, self.n_basis])
        # self.pen2[0, 0] = 1
        self.pen2[1, 1] = 4
        for level1 in range(1, self.level):
            base1 = 2**level1
            for k in range(1, base1):
                self.pen2[base1 + k - 1, base1 + k] = base1
                self.pen2[base1 + k, base1 + k] = 6 * base1
            self.pen2[base1, base1] = 5 * base1
            self.pen2[2 * base1 - 1, 2 * base1 - 1] = 5 * base1
            # self.pen2[base1//2, base1] = -2 ** (level1+0.5)
            # self.pen2[base1 - 1, base1*2-1] = -2 ** (level1+0.5)
            for level2 in range(level1):
                base2 = 2 ** level2
                # interval = 2 ** (level1-level2)
                interval = 2 ** (level1 - level2 - 1)
                val = 2 ** (level2/2 + level1/2)
                for m in range(base2):
                    row = base2 + m
                    if m > 0:
                        self.pen2[row, base1 + 2 * m * interval - 1] += val
                        self.pen2[row, base1 + 2 * m * interval] += val
                    self.pen2[row, base1 + (2 * m + 1) * interval - 1] -= 2*val
                    self.pen2[row, base1 + (2 * m + 1) * interval] -= 2*val
                    if m < base2 - 1:
                        self.pen2[row, base1 + 2*(m + 1) * interval - 1] += val
                        self.pen2[row, base1 + 2*(m + 1) * interval] += val

        self.pen2 = self.pen2 + self.pen2.T - np.diag(np.diag(self.pen2))
        # self.pen2 *= 2**self.level
        self.pen2 = torch.from_numpy(self.pen2)

    def evaluate(self, point):
        self.mat = np.zeros([self.n_basis, len(point)])
        self.mat[0, :] = 1 * ((point >= 0) & (point < 1))
        i = 1
        for j in range(self.level):
            for k in range(2**j):
                self.mat[i, :] = psi(point, j, k)
                i += 1
        self.mat = torch.from_numpy(self.mat.T)

    def to(self, device):
        self.mat = self.mat.to(device)
        self.pen2 = self.pen2.to(device)

    def pen_1d(self, param):
        pen = param @ self.pen2 @ param
        # if pen > 0:
        #     var = torch.sum(param[1-self.n_basis:] ** 2)
        #     pen = pen/var
        return pen

    def pen_2d(self, param, basis):
        pen = torch.trace((param.t() @ self.pen2) @ param) +\
              torch.trace((param @ basis.pen2) @ param.t())
        # if pen > 0:
        #     var = torch.sum(param ** 2) - param[0, 0]**2
        #     pen = pen / var
        return pen * 0.5

    def plot(self, param):
        param = torch.reshape(param, (-1,)).tolist()
        x = np.arange(1/128, 1, 1/64)
        y = param[0]
        i = 1
        for j in range(self.level):
            for k in range(2**j):
                y += psi(x, j, k) * param[i]
                i += 1
        plt.plot(x, y, '_')
        plt.show()
