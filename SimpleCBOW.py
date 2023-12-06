import numpy as np
from layer import *
from tqdm import tqdm
class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size, context_size):
        V, H = vocab_size, hidden_size

        self.vocab_size = V

        # 입력층, 출력층 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layers = []
        for _ in range(context_size):  # 문맥 크기에 따라 입력 계층 생성
            layer = MatMul(W_in)
            self.in_layers.append(layer)

        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = self.in_layers + [self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i in range(contexts.shape[1]):

            x = np.zeros((contexts.shape[0], self.vocab_size))
            for row, word_idx in enumerate(contexts[:, i]):
                x[row, word_idx] = 1

            h += self.in_layers[i].forward(x)

        h *= 1.0 / contexts.shape[1]
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)  # 손실 계층의 역전파
        dout = self.out_layer.backward(dout)  # 출력 계층의 역전파
        dout *= 1 / len(self.in_layers)  # 평균 계층의 역전파
        for layer in self.in_layers:
            layer.backward(dout)
        return None