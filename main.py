import json
import torch
from torch.utils.data import DataLoader

# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from rich.console import Console
from rich.traceback import install
from rich.progress import Progress
from rich.progress import track

from dataset_loader import load_dataset
from MobileNetV2 import MobileNetV2


# Rich console initialization
console = Console()
# Rich traceback initialization
install()

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
    train_dataset, test_dataset, label_strings = load_dataset(config["data"])

    # Create data loaders
    def collate_fn(batch):
        return tuple(zip(*batch))
    train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=len(label_strings)+1,
                                    trainable_backbone_layers=5).to(DEVICE)
    # console.log("[bold green]Model Overview[/bold green]")
    # console.log(model)

    # Training
    console.log("[bold cyan]Begin training...[/bold cyan]")

    model.train()
    num_epochs = config["model"]["num_epochs"]

    for epoch in range(num_epochs):
        for images, targets in track(train_loader, description=f"[cyan]Epoch {epoch+1}/{num_epochs}[/cyan]"):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            output = model(images, targets)
        console.log(output)

    # Model parameters saving
    torch.save(model.state_dict(), config["model"]["save_path"])

    # Testing
    model.eval()






