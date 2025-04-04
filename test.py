import numpy as np
import pickle

from model import MLP
from dataset import load_cifar10, load_cifar10_batch

def test(model, X_test, y_test, model_path=None):
    if model_path:
        with open(model_path, 'rb') as f:
            best_params = pickle.load(f)
        model.params = best_params
    test_acc = (predict(model, X_test) == y_test).mean()
    print('\nTest accuracy: %.2f%%' % (test_acc * 100))

def predict(model, X):
    scores, _ = model.forward(X)
    y_pred = np.argmax(scores, axis=1)
    return y_pred

if __name__ == "__main__":
    # 加载数据集
    X_train, y_train, X_test, y_test = load_cifar10('cifar-10-batches-py')

    # 数据预处理
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    model = MLP(input_size=32 * 32 * 3, hidden_size=100, output_size=10)

    # 测试模型
    model_path = 'ckpts/best_params.pkl'
    test(model, X_test, y_test, model_path)
