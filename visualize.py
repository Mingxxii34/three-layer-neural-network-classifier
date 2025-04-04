import numpy as np
import matplotlib.pyplot as plt
import pickle

def visualize_weights(params, save_dir='ckpts'):
    # 可视化第一层权重 W1
    W1 = params['W1']
    num_filters = W1.shape[1]
    filter_size = int(np.sqrt(W1.shape[0] / 3))  # 假设输入是 3 通道图像

    # 创建一个子图网格
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i in range(grid_size):
        for j in range(grid_size):
            if i * grid_size + j < num_filters:
                # 提取一个滤波器的权重
                filter_weights = W1[:, i * grid_size + j].reshape(3, filter_size, filter_size).transpose(1, 2, 0)
                # 归一化权重到 [0, 1] 范围
                filter_weights = (filter_weights - np.min(filter_weights)) / (
                        np.max(filter_weights) - np.min(filter_weights))
                axes[i, j].imshow(filter_weights)
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    plt.suptitle('Visualization of First Layer Weights (W1)')
    plt.savefig(f'{save_dir}/first_layer_weights.png')  # 保存图像
    plt.close()  # 关闭图像


    # 可视化第二层权重 W2
    W2 = params['W2']
    num_classes = W2.shape[1]
    hidden_size = W2.shape[0]

    fig, axes = plt.subplots(num_classes, 1, figsize=(5, num_classes))
    for i in range(num_classes):
        # 提取一个类别的权重
        class_weights = W2[:, i].reshape(-1)
        # 归一化权重到 [0, 1] 范围
        class_weights = (class_weights - np.min(class_weights)) / (
                np.max(class_weights) - np.min(class_weights))
        axes[i].bar(range(hidden_size), class_weights)
        axes[i].set_title(f'Weights for Class {i}')
        axes[i].set_xlabel('Hidden Units')
        axes[i].set_ylabel('Weight')

    plt.suptitle('Visualization of Second Layer Weights (W2)')
    plt.title('Weights Connecting Hidden Layer to Output Classes')  # 添加图像标题
    plt.tight_layout()
    plt.savefig(f'{save_dir}/second_layer_weights.png')  # 保存图像
    plt.close()  # 关闭图像

if __name__ == "__main__":
    model_path = 'ckpts/best_params.pkl'
    with open(model_path, 'rb') as f:
        best_params = pickle.load(f)
    visualize_weights(best_params)