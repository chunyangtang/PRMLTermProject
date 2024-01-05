# PRMLTermProject

Term Project for Pattern Recognition and Machine Learning (2023 Fall)

## Task
Use the given dataset and train an ML model for object detection.

## File Structure

The project initially contains the following files and folders:

- `main.py`: main program for training and testing.
- `config.json`: configuration file for the program, including training, logging and dataset parameters.
- `dataset/`: dataset folder containing the provided subset of PASCAL VOC2012 dataset (1713 photos), with annotations 
in `Annotations/` and images in `JPEGImages/`, each .jpg image have a corresponding .xml annotation.
- `utils/`: folder for the implementation of the project utilities.
  - `accuracy_evaluator.py`: Evaluating the model accuracy by calculating mAP (Mean Average Precision), call `calculate_accuracy` for evaluation.
  - `dataset_loader.py`: Loading, parsing and splitting the dataset, call `load_dataset` to get the train, validation, test sets and strings for each label.
  - `image_utils.py`: Image processing utilities, including image transformation, visualization, etc.
  - `logging_utils.py`: Logging utilities, including logging to console and exporting training logs to file.

On training the model, the following files and folders will be created:

- `model_weights/fasterrcnn_resnet50_fpn_weights_best.pth`: The current model weights with highest validation accuracy.

On testing the model, the following files and folders will be created:

- `annotated_images/`: folder for saving the annotated test images.

After training and testing the model, the following files and folders will be created:

- `model_weights/fasterrcnn_resnet50_fpn_weights_last.pth`: The model weights from the last epoch.
- `logs/`: folder for saving the training logs, figure of loss & accuracy, configuration file and main program file, and `annotated_images/` & `model_weights/`.

## Usage

As PyTorch compatability varies on different platforms, we do not provide an `requirements.txt` file. 
Please manually install the compatible version of PyTorch and other dependencies as follows:
```
torch torchvision numpy matplotlib rich albumentations
```
As for PyTorch version, we develop the project under `2.0.1`, but it should work on other versions as well.

To run the model training and testing, use:
```bash
python main.py
```

Notice that:
- The program will automatically download the pretrained models from the Internet if using `fasterrcnn_resnet50_fpn`.
- The program will automatically create `model_weights` and `logs` folder for logging.

To change the configurations, refers to `config.json` and modify the parameters.


## TODO
- [x] Implement a better way for data augmentation as it's currently implemented in main.py.
- [x] Create a more reproducible way for label strings storing as current labels string orders are different on different runs.
- [x] Implement an unified way for logging as some of them are currently directly put in main.py.
- [x] Make the config.json file more readable in the data part.


