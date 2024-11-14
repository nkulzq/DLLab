import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import json
from torch.nn import functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1, 1)),
                   nn.Flatten())

model = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))


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
    filename = 'GoogLeNet'
    json.dump(losses, open('./{}/losses.json'.format(filename), 'w'))
    json.dump(train_acc, open('./{}/train_acc.json'.format(filename), 'w'))
    json.dump(test_acc, open('./{}/test_acc.json'.format(filename), 'w'))