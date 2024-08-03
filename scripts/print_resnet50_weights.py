
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

if __name__ == "__main__":
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    for name, param in resnet50.named_parameters():
        print(name)