import json
import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader

# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from rich.console import Console
from rich.traceback import install
from rich.progress import track

from utils.dataset_loader import load_dataset
from utils.accuracy_evaluator import calculate_accuracy


# Rich console initialization
console = Console()
# Rich traceback initialization
install()

log_str = ""

device_str = "cpu"
if torch.cuda.is_available():
    device_str = "cuda"
    console.log(f"[bold green]Using CUDA as {torch.cuda.get_device_name('cuda')} is available.[/bold green]")
elif torch.backends.mps.is_available():
    device_str = "mps"
    console.log("[bold green]Using Apple MPS as it's available.[/bold green]")
else:
    console.log("[bold red]CUDA is not available, using CPU.[/bold red]")

DEVICE = torch.device(device_str)


if __name__ == "__main__":

    # Loading the config file
    with open("config.json", "r") as f:
        config = json.load(f)

    # Print the config file
    console.log("[bold green]Configurations Overview[/bold green]")
    console.log(config)

    # Loading the dataset
    train_dataset, val_dataset, test_dataset, label_strings = load_dataset(config["data"])

    # Create data loaders
    def collate_fn(batch):
        return tuple(zip(*batch))
    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["model"]["batch_size"],
                             shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["model"]["batch_size"],
                            shuffle=False, collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(weights='DEFAULT', num_classes=len(label_strings)+1,
                                    trainable_backbone_layers=3).to(DEVICE)
    # console.log("[bold green]Model Overview[/bold green]")
    # console.log(model)

    # Define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config["model"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=config["model"]["lr_factor"],
                                                           patience=config["model"]["lr_patience"], verbose=True)

    # Training
    console.log("[bold cyan]Begin training...[/bold cyan]")

    num_epochs = config["model"]["num_epochs"]

    # for model validation
    best_val_accuracy = float(0)  # Best validation accuracy
    no_improve_epochs = 0   # Current epochs without validation improvement
    es_epochs = config["model"]["early_stop"]  # Early stopping epochs

    for epoch in range(num_epochs):
        model.train()
        for images, targets in track(train_loader, description=f"[cyan]Epoch  {epoch+1} / {num_epochs}     [/cyan]"):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
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
        with console.status("[bold green]Calculating Validation Accuracy...[/bold green]"):
            valid_accuracy = calculate_accuracy(all_predictions, all_targets)

        console.log(f"[bold green]Epoch {epoch+1}/{num_epochs}, Training Loss: {losses.item()}, "
                    f"Validation Accuracy: {valid_accuracy}[/bold green]")
        log_str += f"Epoch {epoch+1}/{num_epochs}, Training Loss: {losses.item()}, " \
                   f"Validation Accuracy: {valid_accuracy}\n"

        scheduler.step(valid_accuracy)

        if valid_accuracy > best_val_accuracy:
            best_val_accuracy = valid_accuracy
            no_improve_epochs = 0
            torch.save(model.state_dict(), "model_weights/" + config["model"]["save_path"])
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= es_epochs:
            console.log(f"[bold red]Early stopping at epoch {epoch+1}[/bold red]")
            log_str += f"Early stopping at epoch {epoch+1}\n"
            break

    console.log(f"[bold green]Training finished. Best Validation Accuracy: {best_val_accuracy}[/bold green]")
    log_str += f"Training finished. Best Validation Accuracy: {best_val_accuracy}\n"

    # Testing
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in track(test_loader, description=f"[cyan]Model Testing...[/cyan]"):
            images = list(image.to(DEVICE) for image in images)
            outputs = model(images)

            all_predictions.extend(outputs)
            all_targets.extend(targets)

    # Calculate the accuracy
    with console.status("[bold green]Calculating Test Accuracy...[/bold green]"):
        accuracy = calculate_accuracy(all_predictions, all_targets)
    console.log(f"[bold green]Test Accuracy: {accuracy}[/bold green]")
    log_str += f"Test Accuracy: {accuracy}\n"

    # Logging the results
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
    os.system("cp main.py {}".format(os.path.join(log_dir, "main.py.bak")))

    # Save the model
    os.system("cp model_weights/{} {}".format(config["model"]["save_path"], log_dir))

    # Save the results
    with open(os.path.join(log_dir, "results.txt"), "w") as f:
        f.write(log_str)



