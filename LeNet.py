import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import json
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feature_hook_img = {}
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10))

    def forward(self, X):
        X = self.features(X)
        X = self.classify(X)
        return X

    def add_hooks(self):
        def create_hook_fn(idx):
            def hook_fn(model, input, output):
                self.feature_hook_img[idx] = output.cpu()

            return hook_fn

        for _idx, _layer in enumerate(self.features):
            _layer.register_forward_hook(create_hook_fn(_idx))

    def add_image_summary(self, writer, step, prefix=None):
        if len(self.feature_hook_img) == 0:
            return
        if prefix is None:
            prefix = 'layer'
        else:
            prefix = f"{prefix}_layer"
        for _k in self.feature_hook_img:  # 包含原始图像
            _v = self.feature_hook_img[_k][0:1, ...]  # 只获取第一张图像
            _v = torch.permute(_v, (1, 0, 2, 3))  # (1,c,h,w)->(c,1,h,w)# 交换通道，展示每个维度的提取的图像特征
            writer.add_images(f"{prefix}_{_k}", _v, step)


trans = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor()
])
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=trans, download=True)
train_loader = DataLoader(mnist_train, shuffle=True, batch_size=256)
test_loader = DataLoader(mnist_test, shuffle=False, batch_size=256)

model = LeNet()
model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

losses = []
train_acc = []
test_acc = []
mean = []
# writer = SummaryWriter(log_dir='./output/log')
# writer.add_graph(model, torch.empty(10, 1, 28, 28))


def test(data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            image, target = data
            image = image.to('cuda')
            target = target.to('cuda')
            output = model(image)
            probability, predict = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (predict == target).sum().item()
            acc = correct / total
        print("准确率为：%.2f" % acc)
        # model.add_image_summary(writer, epoch, 'test')
    return acc


def train(data_loader):
    model.train()
    for index, data in enumerate(data_loader):
        image, target = data
        image = image.to('cuda')
        target = target.to('cuda')
        optimizer.zero_grad()
        y_predict = model(image)
        loss = criterion(y_predict, target)
        # print(loss)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def visualize_kernels(weights, num_rows, num_cols, save_path):
    fig, axes = plt.subplots(num_rows, num_cols)
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            kernel = weights[idx, :, :, :]
            kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))
            axes[i, j].imshow(np.transpose(kernel, (1, 2, 0)), cmap='viridis')  # 仅展示第一个输入通道的卷积核
            axes[i, j].axis('off')
            idx += 1
    plt.savefig(save_path)  # 保存图像到指定路径
    plt.close(fig)  # 关闭图像窗口，释放资源（可选）


if __name__ == '__main__':
    max_epoch = 10
    model.apply(init_weights)
    for epoch in range(max_epoch):
        train(train_loader)
        test(test_loader)
    filename = 'LeNet_MaxPool'
    json.dump(losses, open('./{}/losses.json'.format(filename), 'w'))
    json.dump(train_acc, open('./{}/train_acc.json'.format(filename), 'w'))
    json.dump(test_acc, open('./{}/test_acc.json'.format(filename), 'w'))