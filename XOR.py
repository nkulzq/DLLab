from dataset import XORDataset
import torch
from torch import nn
from torch.utils.data import DataLoader
import json


train_dataset = XORDataset(6000)
test_dataset = XORDataset(1000)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
losses = []
train_acc = []
test_acc = []


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(2,5)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(5,1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


model = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)


def test(data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            input = data['input']
            target = data['output']
            output = model(input)
            predicted = output > 0.5
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc = correct / total
        print("准确率为：%.2f" % acc)
    return acc


def train(data_loader):
    model.train()
    for index, data in enumerate(data_loader):
        input = data['input']
        target = data['output']
        optimizer.zero_grad()
        y_predict = model(input)
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
    max_epoch = 5
    for epoch in range(max_epoch):
        train(train_loader)
    json.dump(losses, open('./results/losses.json', 'w'))
    json.dump(train_acc, open('./results/train_acc.json', 'w'))
    json.dump(test_acc, open('./results/test_acc.json', 'w'))