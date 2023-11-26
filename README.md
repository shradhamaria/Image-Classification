# Flower Image Classifier

This project is an image classifier that can recognize different species of flowers. The model is trained on a dataset of 102 flower categories, and it can be used as part of an application that identifies flowers from images.

## Project Overview

The project is broken down into the following steps:

1. **Load and preprocess the image dataset:** Use torchvision to load the data and apply necessary transformations such as random scaling, cropping, and flipping for training. Resize and normalize the data for validation and testing.

2. **Train the image classifier:** Utilize a pre-trained neural network (VGG16 in this case) and define a new feed-forward classifier. Train the classifier on the flower dataset, track loss, and accuracy on the validation set.

3. **Save the trained model:** Save the trained model along with other necessary information such as the mapping of classes to indices and optimizer state. This allows for easy loading and inference later.

4. **Inference:** Test the trained network on the test set to measure its accuracy on new, unseen images.

5. **Prediction:** Implement a function to make predictions using the trained model. This function takes an image path and returns the top-k most likely classes along with their probabilities.

6. **Sanity Checking:** Perform sanity checks by displaying an image along with the top predicted classes to ensure the model's predictions make sense.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python
- PyTorch
- Matplotlib
- NumPy
- PIL

### Installation

Clone the repository:

```bash
https://github.com/SmartHacks25/Image-Classification.git
```

## Usage
To train the model:

```bash
python train.py
```

To make predictions:

```bash
python predict.py --image_path /path/to/image.jpg --model_path /path/to/checkpoint.pth
```

For more options and parameters, refer to the documentation in each script.

## Acknowledgments
This project is part of the Udacity Nanodegree program.