import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, random_split
from torchvision.io import read_image
from rich.console import Console
from rich.progress import Progress

from utils.image_utils import image_transform


class MyImageDataset(Dataset):
    def __init__(self, images: list, targets: list):
        """
        :param images: image dataset in the form of an n-length list, each element is a tensor shaped (c, h, w)
        :param targets: bboxes & labels of the images, an n-length list with each element being a dict like
                        {"boxes": torch.FloatTensor, "labels": torch.IntTensor}
        """
        super(MyImageDataset, self).__init__()
        self.images = images
        self.targets = targets
        assert len(self.images) == len(self.targets), "The number of images and labels are not equal."
        self.data_len = len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]

    def __len__(self):
        return self.data_len


def xml_label_parser(xml_path: str) -> (list, list):
    """
    Parsing the xml label file of an image to derive the bounding boxes of the objects in the image.
    NOTE THAT THIS FUNCTION IS FOR SINGLE IMAGE & LABEL FILE ONLY.
    :param xml_path: the path of the xml label file of an image
    :return bboxes: the bounding boxes of the objects in the image, int list with the shape of (n, 4)
    :return labels: the labels of the objects in the image, str list with the length of n
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # image size for label validation
    height = int(root.find("size").find("height").text)
    width = int(root.find("size").find("width").text)

    # object bounding boxes in the image
    objects = root.findall("object")
    bboxes = []
    for obj in objects:
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        assert xmin <= width and ymin <= height and xmax <= width and ymax <= height, \
            f"The bounding box is out of the image at {xml_path}."
        bboxes.append([xmin, ymin, xmax, ymax])

    # object labels in the image
    labels = [obj.find("name").text for obj in objects]

    return bboxes, labels

def load_dataset(configs: dict):
    """
    Load the dataset from the given configurations in file path and return the dataset.
    :param configs: the data session of the project config file `config.json`
    :return train_images: the training images, list of tensors shaped (c, h, w)
    :return train_labels: the training labels, list of dicts: {"boxes": torch.FloatTensor, "labels": torch.IntTensor}
    :return test_images: the testing images, list of tensors shaped (c, h, w)
    :return test_labels: the testing labels, list of dicts: {"boxes": torch.FloatTensor, "labels": torch.LongTensor}
    :return label_strings: all the labels in the dataset, order consistent with the labels in the dataset, list of str
    """

    # loading the configurations
    img_path, label_path = configs["img_path"], configs["label_path"]
    train_start, train_end = configs["train_start"], configs["train_end"]
    test_start, test_end = configs["test_start"], configs["test_end"]
    img_format, label_format = configs["img_format"], configs["label_format"]

    # loading the image and label filenames
    img_list = os.listdir(img_path)
    img_list.sort()
    label_list = os.listdir(label_path)
    label_list.sort()
    # getting the start and end index of the train and test dataset
    train_start_index = img_list.index(train_start + img_format)
    train_end_index = img_list.index(train_end + img_format)
    test_start_index = img_list.index(test_start + img_format)
    test_end_index = img_list.index(test_end + img_format)
    # separating the train and test dataset filenames
    train_images_filename = img_list[train_start_index:train_end_index + 1]
    train_labels_filename = label_list[train_start_index:train_end_index + 1]
    test_images_filename = img_list[test_start_index:test_end_index + 1]
    test_labels_filename = label_list[test_start_index:test_end_index + 1]

    # loading the train and test dataset
    label_strings = set()  # all the labels in the dataset
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    with Console() as console:
        console.log(f"[bold green]Loading {len(train_images_filename)} training data and "
                    f"{len(test_images_filename)} testing data...[/bold green]")

    with Progress() as progress:  # progress bar
        task1 = progress.add_task("Loading Training Data", total=len(train_images_filename))
        task2 = progress.add_task("Loading Testing Data", total=len(test_images_filename))
        for img_train, label_train in zip(train_images_filename, train_labels_filename):
            # updating the progress bar
            progress.update(task1, advance=1, description=f"[cyan]Loading {img_train}")
            # reading the image
            try:
                img = read_image(img_path + img_train)
            except RuntimeError as e:
                print(img_path + img_train)
                raise RuntimeError(e)

            # reading the label
            bboxes, labels = xml_label_parser(label_path + label_train)
            bboxes = torch.FloatTensor(bboxes)
            # adding the labels to the set
            label_strings.update(labels)

            # image transformation
            if configs["image_transform"]["transform"]:
                img, bboxes = image_transform(configs["image_transform"], img, bboxes)

            # Creating image id
            img_id = int(os.path.splitext(img_train)[0])
            img_id = torch.tensor([img_id])

            # adding the image and label to the dataset
            train_images.append(img)
            train_labels.append({"boxes": bboxes, "labels": labels, "image_id": img_id})

        for img_test, label_test in zip(test_images_filename, test_labels_filename):
            # updating the progress bar
            progress.update(task2, advance=1, description=f"[cyan]Loading {img_test}")
            # reading the image
            try:
                img = read_image(img_path + img_test)
            except RuntimeError as e:
                print(img_path + img_test)
                raise RuntimeError(e)

            # reading the label
            bboxes, labels = xml_label_parser(label_path + label_test)
            bboxes = torch.FloatTensor(bboxes)
            # adding the labels to the set
            label_strings.update(labels)

            # image transformation
            if configs["image_transform"]["transform"]:
                img, bboxes = image_transform(configs["image_transform"], img, bboxes)

            # adding the image and label to the dataset
            test_images.append(img)
            test_labels.append({"boxes": bboxes, "labels": labels, "image_id": os.path.splitext(img_test)[0]})

    # Duplicated with the one in main.py
    # with Console() as console:
    #     console.log(f"[bold green]All labels are: \n{label_strings}[/bold green]")

    with Console() as console, console.status("[bold green]Working on creating datasets...") as status:
        # converting str labels to torch.IntTensor labels
        label_strings = list(label_strings)
        label_strings = ["background"] + label_strings  # adding background label
        for i in range(len(train_labels)):
            train_labels[i]["labels"] = torch.LongTensor(
                [label_strings.index(label) for label in train_labels[i]["labels"]])
        for i in range(len(test_labels)):
            test_labels[i]["labels"] = torch.LongTensor(
                [label_strings.index(label) for label in test_labels[i]["labels"]])

        train_dataset = MyImageDataset(train_images, train_labels)
        test_dataset = MyImageDataset(test_images, test_labels)

        # splitting the train dataset into train and validation dataset
        val_len = int(len(train_dataset) * configs["val_proportion"])
        train_len = len(train_dataset) - val_len
        train_dataset, val_dataset = random_split(train_dataset, [train_len, val_len])

        console.log("Datasets created.")

    return train_dataset, val_dataset, test_dataset, label_strings


# For testing
if __name__ == "__main__":
    config = {
        "img_path": "./dataset/JPEGImages/",
        "label_path": "./dataset/Annotations/",

        "train_start": "2007_000559",
        "train_end": "2012_001051",
        "test_start": "2012_001055",
        "test_end": "2012_004308",

        "img_format": ".jpg",
        "label_format": ".xml",

        "comment": "The paths and critical labels of the dataset."
    }

    dataset_train, dataset_val, dataset_test, label_strs = load_dataset(config)
