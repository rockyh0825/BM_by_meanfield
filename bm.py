import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

class BoltzmannMachine:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.node_num = height * width
        self.h = np.random.normal(0, 1, self.node_num)
        self.J = np.random.normal(0, 1, (self.node_num, self.node_num))
        self.expected_value= np.random.uniform(-1, 1, self.node_num)

    def calc_expected_value(self, steps):
        # 自己無撞着方程式を計算して期待値を計算する
        for _ in range(steps):
            expected_value_old = self.expected_value.copy()
            for i in range(self.node_num):
                self.expected_value[i] = np.tanh(np.sum(self.J[i, :] * self.expected_value) + np.sum(self.J[:, i] * self.expected_value) - 2 * self.J[i, i] * self.expected_value[i] + self.h[i])

            if np.max(np.abs(expected_value_old - self.expected_value)) < 1e-3: # 収束判定
                break

    def gradient(self, data):
        self.calc_expected_value(1000)
        grad_h = np.mean(data, axis=0) - self.expected_value
        grad_J = np.zeros((self.node_num, self.node_num))
        for d in data:
            grad_J += np.outer(d, d)
        grad_J /= len(data)
        grad_J -= np.outer(self.expected_value, self.expected_value)
        return grad_h, grad_J
    
    def update(self, data, learning_rate, steps):
        for step in range(steps):
            old_h, old_J = self.h.copy(), self.J.copy()
            grad_h, grad_J = self.gradient(data)
            self.h += learning_rate * grad_h
            self.J += learning_rate * grad_J

            if np.max(np.abs(old_h - self.h)) < 1e-3 and np.max(np.abs(old_J - self.J)) < 1e-3: # 収束判定
                print("Converged at step", step)
                break

# MNISTデータセットをロード
digits = datasets.load_digits()

# 数字6のインデックスを取得
digit_6_indices = digits.target == 9

# 数字6のデータのみを抽出
digit_6_data = digits.data[digit_6_indices]

# データを0-1の範囲に正規化
normalized_data = digit_6_data / 16.0

# 0.5を閾値として+1,-1に変換
binary_data = np.where(normalized_data >= 0.5, 1, -1)

bm = BoltzmannMachine(8, 8)
bm.update(binary_data, 0.1, 1000)

plt.imshow(bm.expected_value.reshape(8, 8), cmap='gray')
plt.show()