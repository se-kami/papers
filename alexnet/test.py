import torch
import torch.nn as nn
import model
from torchvision import models

torch.manual_seed(0)


t = torch.rand(16, 3, 224, 224)
a1 = models.AlexNet(10, pretrained=False)
a2 = model.AlexNet(10)

v1 = a1(t)
v2 = a2(t)

print(v1 - v2)
print(v1.shape)
