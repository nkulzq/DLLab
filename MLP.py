import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import json
import torch.nn.functional as F
from modules import maxout

losses = []
train_acc = []
test_acc = []
mean = []

transform = transforms.Compose({
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), 0.3081)
})


train_data = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)


test_data = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, shuffle=False, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784, 256)
        self.bn = nn.BatchNorm1d(256)
        self.activation = nn.ReLU()
        # self.activation = nn.GELU()
        # self.activation = maxout(2, 256, 256)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear1(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


model = Model()
criterion = nn.CrossEntropyLoss()
# criterion = nn.KLDivLoss('batchmean')
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)


def test(data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            image, target = data
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
        optimizer.zero_grad()
        y_predict = model(image)
        # print(y_predict)
        # log_probs = F.log_softmax(y_predict, dim=1)
        # target_dis = F.one_hot(target, num_classes=10).float()
        # print(log_probs)
        loss = criterion(y_predict, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if index % 20 == 0:
            train_acc.append(test(train_loader))
            test_acc.append(test(test_loader))
            # torch.save(model.state_dict(),"./model/model.pkl")
            # torch.save(optimizer.state_dict(),"./model/optimizer.pkl")
            print("损失值为：%.2f" % loss.item())
            print("[{}|{}]".format(index, len(data_loader)))


if __name__ == '__main__':
    max_epoch = 1
    for epoch in range(max_epoch):
        train(train_loader)
    json.dump(losses, open('./results/losses.json', 'w'))
    json.dump(train_acc, open('./results/train_acc.json', 'w'))
    json.dump(test_acc, open('./results/test_acc.json', 'w'))