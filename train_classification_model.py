import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import cv2
import numpy as np
from torchvision.models import alexnet
from torchvision.models import vgg16
from torch import optim
from torchvision.datasets import mnist
weight_path1 = "weights/alexnet-owt-4df8aa71.pth"
im_path = "data/car.jpg"

class GKothyari_net(nn.Module):

    def __init__(self, num_classes=1000):
        super(GKothyari_net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


tree  = open("imagenet_classes.txt")
list1 = list()
for i in tree.readlines():
    l = i.strip()
    list1.append(l)

from PIL import Image
ima = Image.open(im_path)

batch_size_train = 1
batch_size_test = 1

train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('data/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('data/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])),batch_size=batch_size_test, shuffle=True)


transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize
(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print("data file size",example_data.shape)

Device = torch.device('cpu')
Network = GKothyari_net(1000)

loss_function = torch.nn.CrossEntropyLoss()
state_dict = torch.load(weight_path1)
Network.load_state_dict(state_dict)


custom_layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
        )

Network.classifier.add_module("abc",custom_layer)
#print(Network.classifier)
vls = list(Network.parameters())
for i in Network.parameters():
    i.requires_grad = False
    #print(i.requires_grad)
#print(len(vls))

optimizer = optim.SGD(Network.parameters(), lr=0.001,
                      momentum=0.1)
n_epochs = 1000
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(100 + 1)]
#print(test_counter)

def train(epoch):
  Network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = Network(data)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 2 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(Network.state_dict(), '/results/model.pth')
      torch.save(optimizer.state_dict(), '/results/optimizer.pth')

def test():
  Network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = Network(data)
      test_loss += torch.nn.functional.cross_entropy(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()