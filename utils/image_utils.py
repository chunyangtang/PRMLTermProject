
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A


def image_transform(transform_config: dict, image: torch.Tensor, bboxes: torch.Tensor = None) -> (torch.Tensor,
                                                                                                  torch.Tensor):
    """
    Transform the image and bounding boxes according to the given configurations.
    :param transform_config: the configurations of the image transformation
    :param image: the image to be transformed
    :param bboxes: the bounding boxes of the objects in the image, can be None for model inference
    :return image: the transformed image
    :return bboxes: the transformed bounding boxes
    """

    h_orig, w_orig = image.shape[1], image.shape[2]

    # image resizing
    should_resize = "resize" in transform_config
    h_new, w_new = h_orig, w_orig
    if should_resize:
        h_new, w_new = transform_config["resize"]
        transform = transforms.Compose([transforms.Resize((h_new, w_new))])
        image = transform(image)
        # bounding box transformation
        if bboxes is not None:
            bboxes[:, 0] = bboxes[:, 0] / w_orig * w_new
            bboxes[:, 1] = bboxes[:, 1] / h_orig * h_new
            bboxes[:, 2] = bboxes[:, 2] / w_orig * w_new
            bboxes[:, 3] = bboxes[:, 3] / h_orig * h_new

    # image normalization
    if "normalize" in transform_config and transform_config["normalize"]:
        image = image / 255.0

    # image transformation
    if "augment" in transform_config and transform_config["augment"]:
        # albumentations augmentation
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomResizedCrop(h_new, w_new, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc'))
        transformed = transform(image=image.permute(1, 2, 0).numpy(), bboxes=bboxes.numpy())
        image = torch.FloatTensor(transformed["image"]).permute(2, 0, 1)
        bboxes = torch.FloatTensor(transformed["bboxes"])

    return image, bboxes


def image_inference(model: torch.nn.Module, image: torch.Tensor, transform_config: dict = None) -> (torch.Tensor,
                                                                                                    torch.Tensor):
    """
    Perform inference on the given image with the given model and return the predicted bounding boxes and labels.
    :param model: the model to perform inference
    :param image: the image to perform inference
    :param transform_config: the configurations of the image transformation, None for no transformation
    :return bboxes: the predicted bounding boxes
    :return labels: the predicted labels
    """

    # transform the image
    if transform_config is not None:
        image, _ = image_transform(transform_config, image)

    # perform inference
    model.eval()
    with torch.no_grad():
        pred = model([image])

    # get the predicted bounding boxes and labels
    bboxes = pred[0]["boxes"]
    labels = pred[0]["labels"]
    scores = pred[0]["scores"]
    # filter out the predictions with score less than 0.5
    mask = scores > 0.5
    bboxes = bboxes[mask]
    labels = labels[mask]

    return bboxes, labels


def bbox_visualizer(image: torch.Tensor, targets: dict, label_strings: list, save_path: str = None):
    """
    Visualize the bounding box on the image.
    :param image: the image to be visualized. Tensor of shape (C, H, W)
    :param targets: the targets of the image, dict: {"boxes": torch.FloatTensor, "labels": torch.LongTensor}
    :param label_strings: all the labels in the dataset, order consistent with the labels in the dataset
    :param save_path: the path to save the image, None for not saving the image
    """

    num_labels = len(label_strings)
    bboxes = targets["boxes"]
    labels = targets["labels"]
    cmap = matplotlib.colormaps.get_cmap('hsv')
    colors = cmap(np.linspace(0, 1, num_labels))

    image = image.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox, label in zip(bboxes, labels):
        bbox = bbox.cpu().numpy()
        label = label.cpu().numpy()
        color = colors[label]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        plt.text(bbox[0], bbox[1], label_strings[label], color=color, bbox=dict(facecolor='white', alpha=0.5))

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


# For testing
if __name__ == "__main__":
    import os
    if os.getcwd().endswith("PRMLTermProject"):
        os.chdir("utils")
    import sys
    sys.path.append("..")
    # Test bbox_visualizer
    import torchvision
    image = torchvision.io.read_image("../dataset/JPEGImages/2008_007666.jpg")  # 2010_005654.jpg")
    from utils.dataset_loader import xml_label_parser
    bboxes, label_strings = xml_label_parser("../dataset/Annotations/2008_007666.xml")
    labels = [label_strings.index(label) for label in label_strings]
    targets = {"boxes": torch.FloatTensor(bboxes), "labels": torch.LongTensor(labels)}
    bbox_visualizer(image, targets, label_strings)

    # Test inference
    label_strings = ['background', 'bicycle', 'cat', 'tvmonitor', 'pottedplant', 'car', 'bird', 'sheep', 'sofa',
                     'chair', 'dog', 'boat', 'train', 'aeroplane', 'diningtable', 'bottle', 'horse', 'cow',
                     'motorbike', 'bus', 'person']
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=91,
                                                                 trainable_backbone_layers=3)
    model.load_state_dict(torch.load('../model_weights/fasterrcnn_resnet50_fpn_weights_best.pth',
                                     map_location=torch.device('cpu')))

    bboxes, labels = image_inference(model, image, {"normalize": True})
    targets = {"boxes": bboxes, "labels": labels}
    bbox_visualizer(image, targets, label_strings)




