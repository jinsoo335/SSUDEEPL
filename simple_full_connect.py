from collections import OrderedDict

import numpy as np

import layer
from layer import *

class SimpleFullConnect:

    def __init__(self, input_size, hidden_size, output_size):

        '''
        :param input_size: 입력층 뉴런 수
        :param hidden_sizes: 은닉층의 뉴런 수
        :param output_size: 출력층의 뉴런 수
        '''

        # 가중치 초기화
        # 초기에는 정규 분포 * 0.01로 수행 -> 값이 Nan으로 발산
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) / (np.sqrt(input_size) / 2)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, hidden_size) / (np.sqrt(hidden_size) / 2)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = np.random.randn(hidden_size, output_size) / (np.sqrt(hidden_size) / 2)
        self.params['b3'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = MeanSquaredLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x


    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)


    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db

        return grads


    def evaluate(self, x, t):
        y = self.predict(x)

        return mean_squared_error(y, t)


    def calculate_batch_loss(self, data, labels, batch_size):
        total_loss = 0.0
        data_size = len(data)

        for i in range(0, data_size, batch_size):
            x_batch = np.array(data[i: i+batch_size])
            y_batch = np.array(labels[i: i+batch_size])
            total_loss += self.loss(x_batch, y_batch)

        return total_loss / (data_size // batch_size)