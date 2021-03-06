# coding: utf-8

import sys
import json
import numpy as np
import random
import matplotlib.pyplot as plt

import mnist_load

trained_res = []

class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        # 可能存在nan和inf值 需要转化为0和有限数
        return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)


# 定义神经网络结构
class Network(object):
    def __init__(self, sizes, cost=CrossEntropyCost):
        '''
        sizes (list): [input_layer_num, hidden_layer_num, output_layer_num]
        '''
        # 网络层数
        self.num_layers = len(sizes)
        # 每层神经元个数
        self.sizes = sizes
        # 初始化每层权重和偏置
        self.default_weight_initializer()
        # 损失函数
        self.cost = cost


    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        
    def large_weight_initializer(self):
        # 初始化每层的偏置
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # 初始化每层的权重
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # def GD(self, training_data, epochs, eta):
    #     # 开始训练 循环每一个epochs
    #     for j in range(epochs):
    #         # 洗牌 打乱训练数据
    #         random.shuffle(training_data)

    #         # 保存每一层的偏导
    #         nabla_b = [np.zeros(b.shape) for b in self.biases]
    #         nabla_w = [np.zeros(w.shape) for w in self.weights]

    #         # 训练每一个数据
    #         for x, y in training_data:
    #             delta_nabla_b, delta_nabla_w = self.update(x, y)

    #             # 保存一次训练网络中每层的偏导
    #             nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
    #             nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    #         # 更新权重和偏置 Wn+1 = Wn - eta * nw
    #         self.weights = [w - eta * nw for w, nw in zip(self.weights, nabla_w)]
    #         self.biases = [b - eta * nb for b, nb in zip(self.biases, nabla_b)]
    #         print("Epoch {0} complete".format(j))

    def SGD(self, training_data, epochs,mini_batch_size, eta, lmbda=0.0, test_data=None):

        if test_data:
            n_test = len(test_data)

        # 训练数据总个数
        n = len(training_data)

        # 开始训练 循环每一个epochs
        for j in range(epochs):
            # 洗牌 打乱训练数据
            random.shuffle(training_data)

            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            # 训练mini_batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)

            print("Epoch {0} complete".format(j))

            cost = self.total_cost(training_data, lmbda)
            print("Cost on training data: {}".format(cost))
            accuracy = self.accuracy(training_data, convert=True)
            print("Accuracy on training data: {} / {}".format(accuracy, n))

            if test_data:
                cost = self.total_cost(test_data, lmbda, convert=True)
                print("Cost on test data: {}".format(cost))
                accuracy = self.accuracy(test_data)
                print("Accuracy on test data: {} / {}".format(accuracy, n_test))

                # print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        # 保存每一层的偏导
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]

            # 训练每一个mini_batch
            for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.update(x, y)

                # 保存一次训练网络中每层的偏导
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            # 更新权重和偏置 Wn+1 = Wn - eta * nw w: 添加正则化项 -eta*(lmbda/n)*w 
            self.weights = [(1 - eta*(lmbda/n)) * w - eta/len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - eta/len(mini_batch) * nb for b, nb in zip(self.biases, nabla_b)]   

    # 前向传播
    def update(self, x, y):
        # 保存每一层的偏导
        nabla_b = [np.zeros(b.shape) for b in self.biases]      
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x

        # 保存每一层的激励值a=sigmoid(z)
        activations = [x]

        # 保存每一层的z=wx+b
        zs = []
        for b, w in zip(self.biases, self.weights):
            # 计算每一层z
            z = np.dot(w, activation) + b
            zs.append(z)

            # 计算每一层a
            activation = sigmoid(z)
            activations.append(activation)

        # 反向更新
        # 计算最后一层误差

        # delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        delta = self.cost.delta(zs[-1], activations[-1], y) # 交叉熵


        # 最后一层权重和偏置的导数
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 倒数第二层到第一层 权重和偏置的导数
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)

            # 当前层误差
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())


        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
            
            global trained_res
            trained_res.append(sum((int(x==y)) for (x, y) in results))

        return sum((int(x==y)) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * lmbda/len(data)*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def cost_derivative(self, output_activation, y):
        return output_activation - y

    # 保存模型
    def save(self, filename):
        data = {"sizes" : self.sizes,
                "weights" : [w.tolist() for w in self.weights],
                "biases" : [w.tolist() for b in self.biases],
                "cost" : str(self.cost.__name__)
        }
        with open(filename, 'w') as f:
            json.dump(data, f)


# 加载模型
def load(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    cost = getattr(sys.modules[__name__], data["cost"])
    ney = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]

    return net

def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e

# sigmoid 激励函数
def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def train_net():
    training_data, validation_data, test_data = mnist_load.load_data_wrapper()
    net = Network([28*28, 60, 10])
    net.SGD(training_data, 30, 10, 0.5, 5.0, test_data=test_data)

if __name__ == '__main__':
    train_net()
    plt.plot(list(range(30)), trained_res)
    plt.show()

