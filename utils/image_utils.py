
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A


def image_transform(transform_config: dict, image: torch.Tensor, bboxes: torch.Tensor = None,
                    labels: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
    """
    Transform the image and bounding boxes according to the given configurations.
    :param transform_config: the configurations of the image transformation
    :param image: the image to be transformed
    :param bboxes: the bounding boxes of the objects in the image, can be None for model inference
    :param labels: the labels of the objects in the image, can be None for model inference
    :return image: the transformed image
    :return bboxes: the transformed bounding boxes
    :return labels: the transformed labels (nearly the same as the original labels, could be less)
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
        # Restore to uint8 image
        image = image * 255.0
        image = image.type(torch.uint8)

        # albumentations augmentation
        # # transform 1
        # transform = A.Compose([
        #     A.HorizontalFlip(p=0.5),
        #     A.RandomBrightnessContrast(p=0.2),
        #     A.RandomGamma(p=0.2),
        #     A.RandomRotate90(p=0.5),
        #     A.ShiftScaleRotate(p=0.5),
        #     A.RandomResizedCrop(h_new, w_new, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), p=0.5),
        # ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        # transform 2
        transform = A.Compose([
            A.RandomResizedCrop(height=h_new, width=w_new, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)  # HSV color-space augment
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        transformed = transform(image=image.permute(1, 2, 0).numpy(), bboxes=bboxes.numpy(),
                                labels=labels.numpy())

        # Transform image back to float tensor
        image = torch.FloatTensor(transformed["image"] / 255.0).permute(2, 0, 1)

        bboxes, labels = [], []
        for bbox, label in zip(transformed["bboxes"], transformed["labels"]):
            if len(bbox) == 4:
                bboxes.append(bbox)
                labels.append(label)

        bboxes = torch.FloatTensor(bboxes)
        labels = torch.LongTensor(labels)

    return image, bboxes, labels


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
        image, _, _ = image_transform(transform_config, image)

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
        if label >= num_labels:
            continue
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


def transform_dataset_visualization(dataset_path: str, export_path: str):
    """
    Visualize how the training dataset might look like after apply the transformations for augmentation.
    :param dataset_path: the path to the dataset
    :param export_path: the path to export the visualization images
    """
    import os
    # Get the dataset
    config = {"img_path": os.path.join(dataset_path, "JPEGImages"),
              "label_path": os.path.join(dataset_path, "Annotations"), "train_start": "2007_000559",
              "train_end": "2012_001051", "test_start": "2012_001055", "test_end": "2012_004308",
              "img_format": ".jpg", "label_format": ".xml",
              "image_transform": {
            "transform": True,
            "normalize": True
        }, "augmentation": {
            "augment": True
        }, "val_proportion": 0.2
              }

    from utils.dataset_loader import load_dataset
    train_dataset, val_dataset, test_dataset, label_strs = load_dataset(config)
    # Augment the training dataset and visualize
    from rich.progress import track
    os.makedirs(os.path.join(export_path, "augment_imgs"), exist_ok=True)
    for image, target in track(train_dataset):
        bbox_visualizer(image, target, label_strs,
                        save_path=os.path.join(export_path, "augment_imgs", f"{target['image_id'].item()}_ori.png"))
        image, boxes, labels = image_transform({"augment": True}, image,
                                               bboxes=target["boxes"], labels=target["labels"])
        bbox_visualizer(image, {"boxes": boxes, "labels": labels}, label_strs,
                        save_path=os.path.join(export_path, "augment_imgs", f"{target['image_id'].item()}_aug.png"))


def rebuild_inference_images(log_path: str, dataset_path: str):
    """
    Recompute the images and prediction visualization for the test dataset in case folder `inference_img/` is lost.
    :param log_path: the path to the log folder of a model run
    :param dataset_path: the path to the dataset
    """

    # Get the label strings for log.txt
    import os
    log_txt = os.path.join(log_path, "log.txt")
    # Read the log file
    with open(log_txt, "r") as f:
        logs = f.read()
    # Splitting the log by lines
    import re
    log_lines = logs.strip().split("\n")
    # Variable to store labels
    label_strings = []
    # Search for the line with labels
    for i, line in enumerate(log_lines):
        if "All labels are: " in line:
            # Extracting the next line which contains the labels
            if i + 1 < len(log_lines):
                labels_line = log_lines[i + 1]  # Next line contains the labels
                label_strings = re.findall(r"'(.*?)'", labels_line)
            break

    # Get the dataset
    config = {"img_path": os.path.join(dataset_path, "JPEGImages"),
              "label_path": os.path.join(dataset_path, "Annotations"), "train_start": "2007_000559",
              "train_end": "2012_001051", "test_start": "2012_001055", "test_end": "2012_004308",
              "img_format": ".jpg", "label_format": ".xml",
              "image_transform": {
                  "transform": True,
                  "normalize": True
              }, "augmentation": {
            "augment": True
        }, "val_proportion": 0.2
              }
    from utils.dataset_loader import load_dataset
    train_dataset, val_dataset, test_dataset, label_strs = load_dataset(config)

    # Get the model and load the weights
    import torchvision
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=91,
                                                                 trainable_backbone_layers=3)
    model.load_state_dict(
        torch.load(os.path.join(log_path, "model_weights", "fasterrcnn_resnet50_fpn_weights_last.pth"),
                   map_location=torch.device('cpu')))

    # Visualize the inference
    from rich.progress import track
    for image, target in track(test_dataset):
        bboxes, labels = image_inference(model, image)
        targets = {"boxes": bboxes, "labels": labels}
        bbox_visualizer(image, targets, label_strings,
                        save_path=os.path.join(log_path, "inference_img", f"{target['image_id']}_inference.jpg"))


# For testing
if __name__ == "__main__":

    # rebuild_inference_images("../logs/fasterrcnn_240105-215555-311129", "../dataset")

    transform_dataset_visualization("../dataset", "../")
