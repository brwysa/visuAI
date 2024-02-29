# Image Classifier - Command Line Application

This project is a Python-based command-line application for image classification using deep learning models. It includes two main scripts: `train.py` for training a new network on a dataset and `predict.py` for predicting the class of an input image.

## Features

- Train a new network on a dataset with `train.py`.
- Print training loss, validation loss, and accuracy during training.
- Choose from multiple model architectures available in `torchvision.models`.
- Set hyperparameters such as learning rate, number of hidden units, and epochs.
- Option to train the model on GPU.
- Predict the class of an input image with `predict.py`.
- Return top K most likely classes along with probabilities.
- Utilize a mapping of categories to real names for interpretation.
- Use GPU for inference if available.

## Usage

### Train
Optional arguments:
- `--save_dir`: Directory to save checkpoints.
- `--arch`: Choose architecture (default: "vgg13").
- `--learning_rate`: Set learning rate (default: 0.01).
- `--hidden_units`: Set number of hidden units (default: 512).
- `--epochs`: Set number of epochs (default: 20).
- `--gpu`: Use GPU for training.

### Predict
Optional arguments:
- `--top_k`: Return top K most likely classes (default: 1).
- `--category_names`: Path to JSON file mapping categories to real names.
- `--gpu`: Use GPU for inference.

## Dependencies
- Python 3
- PyTorch
- torchvision
- NumPy
- JSON (for category mapping)
