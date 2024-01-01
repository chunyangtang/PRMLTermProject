import torch
import torchvision

# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN


def get_model(configs, num_classes):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT', num_classes=num_classes+1,
                                    trainable_backbone_layers=3)
    return model
