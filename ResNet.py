import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import json
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

model = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

trans = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor()
])
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=trans, download=True)
train_loader = DataLoader(mnist_train, shuffle=True, batch_size=256)
test_loader = DataLoader(mnist_test, shuffle=False, batch_size=256)


model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

losses = []
train_acc = []
test_acc = []
mean = []


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
        if index % 10 == 0:
            train_acc.append(test(train_loader))
            test_acc.append(test(test_loader))
            # torch.save(model.state_dict(),"./model/model.pkl")
            # torch.save(optimizer.state_dict(),"./model/optimizer.pkl")
            print("损失值为：%.2f" % loss.item())
            print("[{}|{}]".format(index, len(data_loader)))


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    max_epoch = 5
    model.apply(init_weights)
    for epoch in range(max_epoch):
        train(train_loader)
    filename = 'ResNet'
    json.dump(losses, open('./{}/losses.json'.format(filename), 'w'))
    json.dump(train_acc, open('./{}/train_acc.json'.format(filename), 'w'))
    json.dump(test_acc, open('./{}/test_acc.json'.format(filename), 'w'))