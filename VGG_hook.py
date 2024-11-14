import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import json

class FeatureExtractor(object):
    def __init__(self, model):
        self.model = model
        self.feature_hook_img = {}

    def add_hooks(self):
        def create_hook_fn(idx):
            def hook_fn(module, input, output):
                self.feature_hook_img[idx] = output.detach().cpu()
            return hook_fn

        for idx, layer in enumerate(self.model.features):
            if isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(create_hook_fn(idx))

    def add_image_summary(self, writer, step, prefix=None):
        if not self.feature_hook_img:
            return
        if prefix is None:
            prefix = 'layer'
        else:
            prefix = f"{prefix}_layer"
        
        for k in self.feature_hook_img:
            v = self.feature_hook_img[k][0:1, ...]
            v = torch.permute(v, (1, 0, 2, 3))
            writer.add_images(f"{prefix}_{k}", v, step)


vgg = models.vgg16(pretrained=True)
vgg.eval()


feature_extractor = FeatureExtractor(vgg)
feature_extractor.add_hooks()


writer = SummaryWriter('./output')


trans = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=trans, download=True)
train_loader = DataLoader(mnist_train, shuffle=True, batch_size=256)
test_loader = DataLoader(mnist_test, shuffle=False, batch_size=256)
data_iter = iter(train_loader)
images, labels = next(data_iter)


with torch.no_grad():
    vgg(images)


feature_extractor.add_image_summary(writer, step=0)


writer.close()