import os

import matplotlib.pyplot as plt
from rich.console import Console


class MyLogger:
    def __init__(self, log_path: str = None):
        self.losses_per_epoch = []
        self.val_accuracy_per_epoch = []
        self.log_path = log_path
        self.console = Console()
        self.log_str = ""

    def console_log(self, log_str: str):
        self.console.log(log_str)
        self.log_str += log_str + "\n"

    def log_epoch(self, losses: float, val_accuracy: float):
        self.losses_per_epoch.append(losses)
        self.val_accuracy_per_epoch.append(val_accuracy)

    def set_log_path(self, log_path: str):
        self.log_path = log_path

    def export_log(self, log_path: str = None):
        if self.log_path is None and log_path is None:
            raise ValueError("log_path is not specified")
        elif log_path is None:
            log_path = self.log_path

        # Create a figure and a set of subplots
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch")
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax1.set_ylabel("Training Loss")
        ax1.plot(self.losses_per_epoch, color="tab:blue", label="Training Loss")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Validation Accuracy")
        ax2.plot(self.val_accuracy_per_epoch, color="tab:orange", label="Validation Accuracy")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc="upper right")

        fig.tight_layout()
        plt.savefig(os.path.join(log_path, "training_losses_val_accuracy.png"))

        # Export log string
        with open(os.path.join(log_path, "log.txt"), "w") as f:
            f.write(self.log_str)


if __name__ == "__main__":
    logger = MyLogger()
    logger.log_epoch(0.1, 0.5)
    logger.log_epoch(0.02, 0.6)
    logger.log_epoch(0.003, 0.7)
    logger.log_epoch(0.0004, 0.8)
    logger.log_epoch(0.00005, 0.9)
    logger.log_epoch(0.000006, 0.95)
    logger.log_epoch(0.0000007, 0.99)

    logger.export_log(".")
