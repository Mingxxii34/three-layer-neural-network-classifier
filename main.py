import numpy as np
import pickle
import os

from model import MLP
from dataset import load_cifar10, load_cifar10_batch
import matplotlib.pyplot as plt

# SGD 优化器
def sgd_update(params, grads, learning_rate):
    for param_name in params:
        params[param_name] -= learning_rate * grads[param_name]
    return params

# 预测类别
def predict(model, X):
    scores, _ = model.forward(X)
    y_pred = np.argmax(scores, axis=1)
    return y_pred

def train(model, X_train, y_train, X_val, y_val, 
          learning_rate=1e-3, 
          learning_rate_decay=0.95, 
          reg=1e-5, 
          num_iters=3000,
          batch_size=200,
          save_dir='ckpts'):
    print(f'\nTrain with learning_rate {learning_rate}, learning_rate_decay {learning_rate_decay}, reg {reg}, batch_size {batch_size}, num_iters {num_iters}')
    raw_lr= learning_rate
    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    best_val_acc = 0
    best_params = {}

    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    for it in range(num_iters):
        indices = np.random.choice(num_train, batch_size)
        X_batch = X_train[indices]
        y_batch = y_train[indices]

        _, grads = model.loss(X_batch, y_batch, reg)
        
        

        model.params = sgd_update(model.params, grads, learning_rate)

        if it % 100 == 0:
            train_loss, grads = model.loss(X_train, y_train, reg)
            val_loss, grads = model.loss(X_val, y_val, reg)
            val_acc = (predict(model, X_val) == y_val).mean()
            
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            train_loss_history.append(train_loss)
            print('iteration %d / %d: train loss %.4f, val loss = %.4f, val acc = %.2f%%' % (it, num_iters, train_loss, val_loss, val_acc * 100))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = model.params.copy()

        if it % iterations_per_epoch == 0:
            learning_rate *= learning_rate_decay
    # Plot the curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([i * 100 for i in range(len(train_loss_history))], train_loss_history, label='Train Loss')
    plt.plot([i * 100 for i in range(len(val_loss_history))], val_loss_history, label='Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([i * 100 for i in range(len(val_acc_history))], val_acc_history, label='Validation Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()

    # 添加超参数标注图例
    hyperparams_text = f"Learning Rate: {raw_lr}\n" \
                   f"Hidden Size: {model.params['W1'].shape[1]}\n" \
                   f"Regularization: {reg}"
    plt.figtext(0.01, 0.95, hyperparams_text, ha="left", va="top", fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    plt.suptitle("Training and Validation Metrics of Three Layer Net", fontsize=14)

    # 保存图像
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = f'{save_dir}/lr_{raw_lr}_hs_{model.params["W1"].shape[1]}_reg_{reg}.png'
    plt.savefig(filename)



    model.params = best_params
    return model

def hyperparameter_search(X_train, y_train, X_val, y_val, 
                          learning_rates = [1e-3, 1e-4], 
                          hidden_sizes = [50, 100], 
                          reg_strengths = [1e-3, 1e-4]):
    print('Search hyperparameter learning_rates in {learning_rates}, hidden_sizes in {hidden_sizes}, reg_strengths in {reg_strengths}')

    best_val_acc = 0
    best_hparams = {}

    for lr in learning_rates:
        for hs in hidden_sizes:
            for reg in reg_strengths:
                model = MLP(input_size=32 * 32 * 3, hidden_size=hs, output_size=10)
                model = train(model, X_train, y_train, X_val, y_val, learning_rate=lr, reg=reg)
                val_acc = (predict(model, X_val) == y_val).mean()
                print('lr %e, hidden_size %d, reg %e, val acc: %.2f%%' % (lr, hs, reg, val_acc * 100))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_hparams = {'learning_rate': lr, 'hidden_size': hs, 'reg': reg}

    print('\nBest validation accuracy: %.2f%%' % (best_val_acc * 100))
    print('Best hyperparameters:', best_hparams)
    return best_hparams

def test(model, X_test, y_test, model_path=None):
    if model_path:
        with open(model_path, 'rb') as f:
            best_params = pickle.load(f)
        model.params = best_params
    test_acc = (predict(model, X_test) == y_test).mean()
    print('\nTest accuracy: %.2f%%' % (test_acc * 100))

if __name__ == "__main__":
    # 超参数
    num_training = 40000
    num_validation = 10000
    root_dir = 'ckpts'
    learning_rates = [1e-3, 5e-4] 
    hidden_sizes = [50, 100]
    reg_strengths = [1e-3, 1e-4]


    # 加载数据集
    X_train, y_train, X_test, y_test = load_cifar10('cifar-10-batches-py')

    # 数据预处理
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    # 划分训练集和验证集
    X_val = X_train[num_training:]
    y_val = y_train[num_training:]
    X_train = X_train[:num_training]
    y_train = y_train[:num_training]

    # 超参数查找
    best_hparams = hyperparameter_search(X_train, y_train, X_val, y_val, 
                                         learning_rates=learning_rates,
                                         hidden_sizes=hidden_sizes,
                                         reg_strengths=reg_strengths
                                         )

    # 使用最佳超参数训练模型
    best_model = MLP(input_size=32 * 32 * 3, hidden_size=best_hparams['hidden_size'], output_size=10)
    best_model = train(best_model, X_train, y_train, X_val, y_val, 
                       learning_rate=best_hparams['learning_rate'],
                       reg=best_hparams['reg'],
                       num_iters=3000)

    with open(os.path.join(root_dir, 'best_params.pkl'), 'wb') as f:
        pickle.dump(best_model.params, f)

    # 测试模型
    model_path = None
    test(best_model, X_test, y_test, model_path)
    