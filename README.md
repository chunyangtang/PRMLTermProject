# PRMLTermProject

Term Project for Pattern Recognition and Machine Learning (2023 Fall)

## Task
Use the given dataset and train an ML model for object detection.

## File Structure & Usage

- `main.py`: main program for training and testing.
- `config.json`: configuration file for the program.
- `dataset/`: dataset folder containing the provided subset of PASCAL VOC2012 dataset (1713 photos), with annotations 
in `Annotations/` and images in `JPEGImages/`, each .jpg image have a corresponding .xml annotation.
- `models/`: folder for the implementation of the models.
- `utils/`: folder for the implementation of the project utilities.
  - `accuracy_evaluator.py`: Evaluating the model accuracy by calculating mAP (Mean Average Precision), call `calculate_accuracy` for evaluation.
  - `dataset_loader.py`: Loading, parsing and splitting the dataset, call `load_dataset` to get the train, validation, test sets and strings for each label.
  - `image_utils.py`: Image processing utilities, including image transformation, visualization, etc.

As PyTorch compatability varies on different platforms, we do not provide an `requirements.txt` file. 
Please manually install the compatible version of PyTorch and other dependencies as follows:
```
torch torchvision numpy matplotlib rich
```
As for PyTorch version, we develop the project under `2.0.1`, but it should work on other versions as well.

To run the model training and testing, use:
```bash
python main.py
```

Notice that:
- There's `os.system('cp ...')` commands in `main.py` so it might need some modification to run on non-Unix-like systems.
- The program will automatically download the pretrained models from the Internet if using `fasterrcnn_resnet50_fpn`.
- The program will automatically create `model_weights` and `logs` folder for logging.

To change the configurations, refers to `config.json` and modify the parameters.



