import json
import os
import shutil
from datetime import datetime
import argparse

import torch
from torch.utils.data import DataLoader, ConcatDataset

# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from rich.traceback import install
from rich.progress import track

from utils.dataset_loader import load_dataset, MyImageDataset
from utils.accuracy_evaluator import calculate_accuracy
from utils.image_utils import bbox_visualizer
from utils.image_utils import image_transform
from utils.logging_utils import MyLogger


# Rich traceback initialization
install()

logger = MyLogger()

device_str = "cpu"
if torch.cuda.is_available():
    device_str = "cuda"
    logger.console_log(f"[bold green]Using CUDA as {torch.cuda.get_device_name('cuda')} is available.[/bold green]")
elif torch.backends.mps.is_available():
    device_str = "mps"
    logger.console_log("[bold green]Using Apple MPS as it's available.[/bold green]")
else:
    logger.console_log("[bold red]CUDA is not available, using CPU.[/bold red]")

DEVICE = torch.device(device_str)


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="The path to the config file.")
    parser.add_argument("--train_one_epoch", action="store_true",
                        help="Train the model for only one epoch, overwriting the 'num_epochs' parameter "
                             "in the configs for debugging.")
    args = parser.parse_args()

    config_path = args.config
    # Loading the config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # Print the config file
    logger.console.log("[bold green]Configurations Overview[/bold green]")
    logger.console.log(config)

    # Loading the dataset
    train_dataset, val_dataset, test_dataset, label_strings = load_dataset(config["data"])

    logger.console_log(f"[bold green]All labels are: \n{label_strings}[/bold green]")

    # Create data loaders
    def collate_fn(batch):
        return tuple(zip(*batch))
    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["model"]["batch_size"],
                             shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["model"]["batch_size"],
                            shuffle=False, collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(weights='DEFAULT', trainable_backbone_layers=3).to(DEVICE)
    # logger.console.log("[bold green]Model Overview[/bold green]")
    # logger.console.log(model)

    # Define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config["model"]["lr"], momentum=config["model"]["SGD_momentum"],
                                weight_decay=config["model"]["SGD_weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=config["model"]["lr_factor"],
                                                           patience=config["model"]["lr_patience"], verbose=True)

    # Training
    logger.console.log("[bold cyan]Begin training...[/bold cyan]")

    num_epochs = config["model"]["num_epochs"] if not args.train_one_epoch else 1

    # for model validation
    best_val_accuracy = float(0)  # Best validation accuracy
    no_improve_epochs = 0   # Current epochs without validation improvement
    es_epochs = config["model"]["early_stop"]  # Early stopping epochs

    for epoch in range(num_epochs):
        model.train()

        # Creating augmented dataset
        if config["data"]["augmentation"]["augment"]:
            with logger.console.status("[bold green]Augmenting the training dataset...[/bold green]"):
                # Augment the training dataset
                image_augmented, target_augmented = [], []
                for image, target in train_dataset:
                    image, boxes, labels = image_transform({"augment": True}, image,
                                                           bboxes=target["boxes"], labels=target["labels"])
                    image_augmented.append(image)
                    target_augmented.append({"boxes": boxes, "labels": labels})

                # Concatenate the augmented dataset with the original dataset
                train_dataset_augmented = MyImageDataset(image_augmented, target_augmented)
                train_dataset_concat = ConcatDataset([train_dataset, train_dataset_augmented])
                train_loader = DataLoader(train_dataset_concat, batch_size=config["model"]["batch_size"],
                                          shuffle=True, collate_fn=collate_fn)

        for images, targets in track(train_loader, description=f"[cyan]Epoch  {epoch+1} / {num_epochs}     [/cyan]"):
            # images = list(image.to(DEVICE) for image in images)
            # targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            images_filtered, targets_filtered = [], []
            for image, target in zip(images, targets):
                image = image.to(DEVICE)
                target = {k: v.to(DEVICE) for k, v in target.items()}
                if target["boxes"].shape[0] == 0:
                    continue
                for i, box in enumerate(target["boxes"]):
                    if (box[2] - box[0]) <= 1 or (box[3] - box[1]) <= 1:
                        target["boxes"] = torch.cat((target["boxes"][:i], target["boxes"][i+1:]))  # remove the box
                images_filtered.append(image)
                targets_filtered.append(target)
            if len(images) == 0:
                continue
            loss_dict = model(images_filtered, targets_filtered)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for images, targets in track(val_loader, description=f"[cyan]Model Evaluating...[/cyan]"):
                images = list(image.to(DEVICE) for image in images)
                outputs = model(images)

                all_predictions.extend(outputs)
                all_targets.extend(targets)

        # Calculate the validation accuracy
        with logger.console.status("[bold green]Calculating Validation Accuracy...[/bold green]"):
            valid_accuracy = calculate_accuracy(all_predictions, all_targets)

        # Logging the results
        logger.log_epoch(losses.item(), valid_accuracy)

        logger.console_log(f"[bold green]Epoch {epoch+1}/{num_epochs}, Training Loss: {losses.item()}, "
                           f"Validation Accuracy: {valid_accuracy}[/bold green]")

        scheduler.step(valid_accuracy)

        if valid_accuracy > best_val_accuracy:  # Update the best validation accuracy
            best_val_accuracy = valid_accuracy
            no_improve_epochs = 0
            # Save the model
            if not os.path.exists("model_weights"):
                os.mkdir("model_weights")
            torch.save(model.state_dict(), "model_weights/" + "{}_best{}".format(*os.path.splitext(config["model"]
                                                                                                   ["save_path"])))
        else:  # No improvement
            no_improve_epochs += 1

        if no_improve_epochs >= es_epochs:
            logger.console_log(f"[bold red]Early stopping at epoch {epoch+1}[/bold red]")
            break

    logger.console_log(f"[bold green]Training finished. Best Validation Accuracy: {best_val_accuracy}[/bold green]")
    torch.save(model.state_dict(), "model_weights/" + "{}_last{}".format(*os.path.splitext(config["model"]
                                                                                           ["save_path"])))

    # Testing
    model.eval()

    all_predictions = []
    all_targets = []

    # Save the annotated images
    if not os.path.exists("annotated_images"):
        os.mkdir("annotated_images")

    with torch.no_grad():
        for images, targets in track(test_loader, description=f"[cyan]Model Testing...[/cyan]"):
            images = list(image.to(DEVICE) for image in images)
            outputs = model(images)

            all_predictions.extend(outputs)
            all_targets.extend(targets)

            # Visualize the images
            for image, target in zip(images, targets):
                bbox_visualizer(image, target, label_strings,
                                save_path=f"annotated_images/{target['image_id']}_annotated.jpg")

    # Calculate the accuracy
    with logger.console.status("[bold green]Calculating Test Accuracy...[/bold green]"):
        accuracy = calculate_accuracy(all_predictions, all_targets)
    logger.console_log(f"[bold green]Test Accuracy: {accuracy}[/bold green]")

    # Logging the results
    with logger.console.status("[bold green]Logging the results...[/bold green]"):
        # Create the results directory
        if not os.path.exists("logs"):
            os.mkdir("logs")
        log_dir = os.path.join("logs", datetime.now().strftime("{}_%y%m%d-%H%M%S-%f"
                                                               .format(config["logger"]["log_folder"])))
        os.mkdir(log_dir)

        # Save the config file
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # Backup the main.py file
        shutil.copy("main.py", os.path.join(log_dir, "main.py.bak"))

        # Save the model
        shutil.copytree("model_weights", os.path.join(log_dir, "model_weights"))

        # Save the annotated images
        shutil.copytree("annotated_images", os.path.join(log_dir, "annotated_images"))

        # Save the logs
        logger.set_log_path(log_dir)
        logger.export_log()

    logger.console_log(f"[bold green]Results saved at {log_dir}[/bold green]")

