import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import json


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))


trans = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=trans, download=True)
train_loader = DataLoader(mnist_train, shuffle=True, batch_size=256)
test_loader = DataLoader(mnist_test, shuffle=False, batch_size=256)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
model = vgg(small_conv_arch)
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
    filename = 'VGG'
    json.dump(losses, open('./{}/losses.json'.format(filename), 'w'))
    json.dump(train_acc, open('./{}/train_acc.json'.format(filename), 'w'))
    json.dump(test_acc, open('./{}/test_acc.json'.format(filename), 'w'))