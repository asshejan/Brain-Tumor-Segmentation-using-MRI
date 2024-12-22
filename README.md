# Medical-Image-Segmentation-Brats-MRI-Dataset-(With Knowledge Distillation) 
# Teacher-Student Model Architecture

This project implements a Teacher-Student model architecture, where the **Teacher Model** is a larger, more complex model that provides soft labels to train the **Student Model**. The Student Model is a smaller, more efficient model designed to mimic the Teacher Model's behavior. This approach leverages model distillation to reduce the computational complexity while retaining performance.

## Table of Contents
1. [Teacher Model Architecture](#teacher-model-architecture)
2. [Student Model Architecture](#student-model-architecture)
3. [Layer Descriptions](#layer-descriptions)
4. [Training Procedure](#training-procedure)
5. [Installation](#installation)
6. [Usage](#usage)

## Teacher Model Architecture

The Teacher Model is a deep Convolutional Neural Network (CNN) that uses separable convolutions, batch normalization, Leaky ReLU activation functions, max-pooling layers, and upsampling layers. This model is used for feature extraction and generating soft labels to teach the Student Model.

### Layers of the Teacher Model:
1. **Input Layer**:
   - Shape: `(128, 128, 3)`
   - Description: The model takes in images of size 128x128 pixels with 3 RGB channels.

2. **Separable Convolutional Layers**:
   - Uses depthwise separable convolutions to reduce parameters while preserving feature extraction ability.
   - Example: `Conv2D(16, (3, 3), padding='same', activation='relu')`

3. **Batch Normalization**:
   - Applied after each convolution to stabilize the learning process.
   - Example: `BatchNormalization()`

4. **Leaky ReLU Activation**:
   - Prevents dead neurons by allowing small negative values.
   - Example: `LeakyReLU(alpha=0.01)`

5. **MaxPooling2D**:
   - Reduces the spatial dimensions of the feature map by taking the maximum value from each patch.
   - Example: `MaxPooling2D(pool_size=(2, 2))`

6. **Dropout**:
   - Regularizes the model by randomly setting a fraction of the input units to zero during training to prevent overfitting.
   - Example: `Dropout(rate=0.5)`

7. **Conv2DTranspose (Upsampling)**:
   - Increases the spatial resolution of the feature map for output generation.
   - Example: `Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')`

8. **Concatenate Layer**:
   - Merges two or more feature maps to provide richer feature representations for the next layer.
   - Example: `Concatenate(axis=-1)`
9. **Params**:
    - Total params: 8,639,778
    - Trainable params: 8,639,778
    - Non-trainable params: 0
## Student Model Architecture

The Student Model is a smaller, more efficient CNN designed to learn from the Teacher Model. It follows a similar structure but uses fewer parameters to ensure fast inference without sacrificing too much performance.

### Layers of the Student Model:
1. **Input Layer**:
   - Shape: `(128, 128, 3)`
   - Description: The model takes in images of size 128x128 pixels with 3 RGB channels.

2. **Separable Convolutional Layers**:
   - Applies separable convolutions to reduce model complexity.
   - Example: `Conv2D(32, (3, 3), padding='same', activation='relu')`

3. **Batch Normalization**:
   - Normalizes the activations to stabilize training.
   - Example: `BatchNormalization()`

4. **Leaky ReLU Activation**:
   - Ensures the model does not suffer from dead neurons.
   - Example: `LeakyReLU(alpha=0.01)`

5. **MaxPooling2D**:
   - Down-samples the feature map to reduce dimensionality.
   - Example: `MaxPooling2D(pool_size=(2, 2))`

6. **Dropout**:
   - Regularization to reduce overfitting.
   - Example: `Dropout(rate=0.3)`

7. **Conv2DTranspose (Upsampling)**:
   - Increases the spatial resolution of the feature map.
   - Example: `Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')`

8. **Concatenate Layer**:
   - Combines feature maps from different layers to enhance feature learning.
   - Example: `Concatenate(axis=-1)`
     
9. **Params**:
    - Total params: 608,589
    - Trainable params: 605,645
    - Non-trainable params: 2,944

## Layer Descriptions

### 1. **Separable Convolutions**
   - Separable convolutions are used to reduce the computational cost. They split the convolution into depthwise and pointwise operations, which are more efficient than standard convolutions.

### 2. **Batch Normalization**
   - Batch normalization normalizes the activations of a given layer by adjusting and scaling them, which speeds up training and improves performance.

### 3. **Leaky ReLU Activation**
   - Leaky ReLU allows a small gradient for negative values, helping to prevent dead neurons, especially during training in deep networks.

### 4. **MaxPooling2D**
   - MaxPooling is a technique to reduce the size of the feature maps while retaining the most important features. This is done by taking the maximum value in each pool of values.

### 5. **Dropout**
   - Dropout is a regularization technique that randomly sets some neurons' weights to zero during training to prevent overfitting.

### 6. **Conv2DTranspose (Upsampling)**
   - Conv2DTranspose is used to increase the spatial dimensions of the feature map, usually in tasks like image segmentation or image generation.

### 7. **Concatenate Layer**
   - Concatenate layers merge two or more feature maps along a specified axis, typically to combine low-level and high-level features.

## Training Procedure

1. **Model Distillation**: The Teacher Model trains first and generates soft labels. These labels are used to train the smaller Student Model, which learns to mimic the Teacher's predictions.
   
2. **Loss Function**: The Student Model uses a combination of Cross-Entropy loss and Kullback-Leibler (KL) Divergence loss to minimize the difference between the Teacher and Student predictions.

3. **Optimizer**: The Adam optimizer is typically used for both models due to its efficiency and adaptability.

4. **Learning Rate Scheduling**: The learning rate is adjusted during training to improve convergence and avoid overshooting during optimization.

## Installation

To install the necessary dependencies, you can use the following commands:

```bash
pip install -r requirements.txt
