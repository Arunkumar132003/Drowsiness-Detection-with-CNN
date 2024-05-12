## Drowsy Detection model using CNN

### Overview
Drowsy Detection is a project aimed at detecting the state of a person's eyes, whether they are open or closed, using machine learning techniques. This project has applications in various fields, especially in the domain of driver safety systems where detecting drowsiness can prevent accidents.

### Dataset
The dataset used in this project consists of images of both closed and open eyes. The images are divided into two directories: `Closed_Eyes` and `Open_Eyes`. Each directory contains images corresponding to the respective eye state.

#### Dataset link : https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset

### Requirements
- Python 3.x
- TensorFlow
- OpenCV
- Matplotlib
- NumPy
- Seaborn

### Model Architecture
The CNN model used in this project consists of convolutional layers followed by max-pooling, batch normalization, dropout, and dense layers. Here's a summary of the model architecture:
- Input: 64x64 grayscale image
- Convolutional Layers:
  - 32 filters, kernel size 5x5, ReLU activation
  - 32 filters, kernel size 5x5, ReLU activation, batch normalization
  - MaxPooling, Dropout (30%)
  - 64 filters, kernel size 3x3, ReLU activation
  - 64 filters, kernel size 3x3, ReLU activation, batch normalization
  - MaxPooling, Dropout (30%)
- Fully Connected Layers:
  - Dense layer with 256 units, ReLU activation, batch normalization
  - Dense layer with 128 units, ReLU activation
  - Dense layer with 84 units, ReLU activation, batch normalization, dropout (30%)
- Output: Sigmoid activation (Binary classification)

### Evaluation
The model is evaluated on a test set to measure its performance. The evaluation includes metrics such as accuracy and a confusion matrix, which shows the model's performance in classifying closed and open eyes.


