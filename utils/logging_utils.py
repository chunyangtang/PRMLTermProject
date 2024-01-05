import os
import re

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console


class MyLogger:
    """
    A logger for logging the training process.
    It provides methods for logging the training loss and validation accuracy per epoch and exporting them as a figure.
    It also provides a method for logging the training process in the console and exporting the log as a text file.
    """
    def __init__(self, log_path: str = None):
        """
        :param log_path: the path to save the log file, can be specified later before exporting.
        """
        self.losses_per_epoch = []
        self.val_accuracy_per_epoch = []
        self.log_path = log_path
        self.console = Console()
        self.log_str = ""

    def console_log(self, log_str: str):
        """
        Log the outputs in the console and store them into a built-in str.
        :param log_str: the string to be logged and outputted in the console.
        """
        self.console.log(log_str)
        self.log_str += log_str + "\n"

    def log_epoch(self, losses: float, val_accuracy: float):
        """
        Log the training loss and validation accuracy per epoch.
        Should be called after each epoch.
        :param losses: The training loss of the last batch in the epoch.
        :param val_accuracy: The validation accuracy of the epoch.
        """
        self.losses_per_epoch.append(losses)
        self.val_accuracy_per_epoch.append(val_accuracy)

    def set_log_path(self, log_path: str):
        """
        Set the path to save the log file.
        :param log_path: the path to save the log file.
        """
        self.log_path = log_path

    def export_log(self, log_path: str = None):
        """
        Export the log file as a text file and the training losses and validation accuracy as a figure.
        :param log_path: The path for saving, this will override the log_path specified in __init__ and set_log_path.
        """
        if self.log_path is None and log_path is None:
            raise ValueError("log_path is not specified")
        elif log_path is None:
            log_path = self.log_path

        # Create a figure and a set of subplots for training losses and validation accuracy
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
        plt.legend(lines + lines2, labels + labels2, loc="upper left")

        fig.tight_layout()
        plt.savefig(os.path.join(log_path, "training_losses_val_accuracy.png"))

        plt.clf()

        # Simple function for moving average
        def moving_average(data, window_size):
            # padding
            data = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        # Create a figure and a set of subplots for training losses and validation accuracy
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch")
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax1.set_ylabel("Training Loss")
        # ax1.plot(self.losses_per_epoch, color="tab:blue", label="Training Loss")
        ax1.plot(self.losses_per_epoch, label='Raw Training Loss', color='tab:blue', alpha=0.3)
        ax1.plot(moving_average(self.losses_per_epoch, 5), label='Smoothed Training Loss', color='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel("Validation Accuracy")
        ax2.plot(self.val_accuracy_per_epoch, label='Raw Accuracy', color='tab:orange', alpha=0.3)
        ax2.plot(moving_average(self.val_accuracy_per_epoch, 5), label='Smoothed Accuracy', color='tab:orange')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc="upper left")

        fig.tight_layout()
        plt.savefig(os.path.join(log_path, "training_losses_val_accuracy_smoothed.png"))

        # Export log string
        with open(os.path.join(log_path, "log.txt"), "w") as f:
            f.write(self.log_str)

    @staticmethod
    def plot_from_log(log_path: str, export_path: str = None):
        """
        Plot the training losses and validation accuracy from the log string.
        :param log_path: The path to the log file.
        :param export_path: The path to export the figure, None for not exporting.
        """
        log_path = os.path.abspath(log_path)
        if export_path is None:
            export_path = os.path.dirname(log_path)
        # Read the log file
        with open(log_path, "r") as f:
            logs = f.read()

        # Splitting the log by lines
        log_lines = logs.strip().split("\n")
        losses = []
        accuracies = []

        # Regex pattern to match floating point numbers
        pattern = r"[-+]?\d*\.\d+|\d+"

        # Extracting loss and accuracy from each line
        for line in log_lines:
            parts = line.split(", ")
            # Make sure the line is as expected
            if len(parts) == 3:
                # Extracting numeric values using regex
                loss_match = re.findall(pattern, parts[1])
                accuracy_match = re.findall(pattern, parts[2])

                # Ensure that there's a match and append the numeric part to the respective lists
                if loss_match:
                    losses.append(float(loss_match[0]))
                if accuracy_match:
                    accuracies.append(float(accuracy_match[0]))

        # Plotting the loss and accuracy
        # Create a figure and a set of subplots for training losses and validation accuracy
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch")
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax1.set_ylabel("Training Loss")
        ax1.plot(losses, color="tab:blue", label="Training Loss")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Validation Accuracy")
        ax2.plot(accuracies, color="tab:orange", label="Validation Accuracy")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc="upper left")

        fig.tight_layout()
        plt.savefig(os.path.join(export_path, "training_losses_val_accuracy.png"))

        plt.clf()

        # Simple function for moving average
        def moving_average(data, window_size):
            # padding
            data = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        # Create a figure and a set of subplots for training losses and validation accuracy
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch")
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax1.set_ylabel("Training Loss")
        # ax1.plot(self.losses_per_epoch, color="tab:blue", label="Training Loss")
        ax1.plot(losses, label='Raw Training Loss', color='tab:blue', alpha=0.3)
        ax1.plot(moving_average(losses, 5), label='Smoothed Training Loss', color='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel("Validation Accuracy")
        ax2.plot(accuracies, label='Raw Accuracy', color='tab:orange', alpha=0.3)
        ax2.plot(moving_average(accuracies, 5), label='Smoothed Accuracy', color='tab:orange')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc="upper left")

        fig.tight_layout()
        plt.savefig(os.path.join(export_path, "training_losses_val_accuracy_smoothed.png"))


# For testing
if __name__ == "__main__":
    logger = MyLogger()
    logger.log_epoch(0.1, 0.5)
    logger.log_epoch(0.02, 0.6)
    logger.log_epoch(0.003, 0.7)
    logger.log_epoch(0.0004, 0.8)
    logger.log_epoch(0.00005, 0.9)
    logger.log_epoch(0.000006, 0.95)
    logger.log_epoch(0.0000007, 0.99)

    # logger.export_log(".")

    MyLogger.plot_from_log("log.txt", ".")
