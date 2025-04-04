import numpy as np
class MLP:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        self.params = {}
        self.params['W1'] = 0.001 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros((1, hidden_size))
        self.params['W2'] = 0.001 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 第一层
        h1 = np.dot(X, W1) + b1
        if self.activation == 'relu':
            h1 = np.maximum(0, h1)
        elif self.activation == 'sigmoid':
            h1 = 1 / (1 + np.exp(-h1))

        # 第二层
        scores = np.dot(h1, W2) + b2
        return scores, h1

    def loss(self, X, y, reg):
        num_train = X.shape[0]
        scores, h1 = self.forward(X)

        # 计算交叉熵损失
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(num_train), y])
        data_loss = np.sum(correct_logprobs) / num_train

        # L2 正则化损失
        reg_loss = 0.5 * reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2))
        loss = data_loss + reg_loss

        # 反向传播
        dscores = probs
        dscores[range(num_train), y] -= 1
        dscores /= num_train

        dW2 = np.dot(h1.T, dscores) + reg * self.params['W2']
        db2 = np.sum(dscores, axis=0, keepdims=True)

        if self.activation == 'relu':
            dh1 = np.dot(dscores, self.params['W2'].T)
            dh1[h1 <= 0] = 0
        elif self.activation == 'sigmoid':
            dh1 = np.dot(dscores, self.params['W2'].T) * (h1 * (1 - h1))

        dW1 = np.dot(X.T, dh1) + reg * self.params['W1']
        db1 = np.sum(dh1, axis=0, keepdims=True)

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return loss, grads

