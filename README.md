[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NishqShah/mnist-from-scratch/blob/main/mnist_from_scratch.ipynb)

# MNIST Digit Classifier — Built from Scratch with NumPy

This project demonstrates how a simple neural network can be implemented **from the ground up** using just NumPy. It’s trained to recognize handwritten digits from the MNIST dataset and achieves a test accuracy of around **94.77%** — all without using any machine learning libraries like TensorFlow or PyTorch.

---

## About the Project

The goal was to understand the inner workings of neural networks by implementing every component manually — including the forward pass, backpropagation, weight updates, and model evaluation.

This kind of hands-on approach is especially useful for developing an intuitive understanding of how deep learning models actually work behind the scenes.

---

## What's Inside

- `mnist_from_scratch.ipynb`: The full notebook with all steps — from data loading and EDA to training and evaluation.
- `eda_images/`: A folder with saved plots and charts from the EDA and evaluation sections (digit counts, sample digits, confusion matrix, etc.).

---

## Key Features

- Neural network built entirely with NumPy  
- One hidden layer with ReLU activation  
- Softmax output for multi-class classification  
- Manual backpropagation and weight updates  
- Accuracy tracking across epochs  
- Confusion matrix to visualize model performance  

---

## EDA Highlights

Before training the model, we explored the dataset to understand the input better. This included checking the distribution of digit classes, visualizing random sample images, and verifying the shape and structure of the input data.

Key visualizations like label distributions and digit samples were saved in the `eda_images/` folder for reference.

---

## Model Architecture

- **Input Layer**: 784 neurons (flattened 28x28 pixel images)  
- **Hidden Layer**: 64 neurons with ReLU  
- **Output Layer**: 10 neurons with Softmax (one for each digit)  

---

## Training Setup

- **Loss Function**: Cross-Entropy  
- **Optimizer**: Basic Gradient Descent  
- **Epochs**: 20  
- **Batch Size**: 64  
- **Learning Rate**: 0.01  
- **Bias Terms**: Manually added to both layers  

---

## Performance

- **Final Test Accuracy**: **94.77%**

Confusion matrix:

![confusion-matrix](eda_images/confusion_matrix.png)

The model performed well overall, though it occasionally confused visually similar digits like 5 and 3, or 9 and 4 — which is expected in a basic model.

---

## What I Learned

- How forward and backward propagation work in practice  
- How to compute gradients and update weights manually  
- The importance of activation functions and bias terms  
- How to evaluate a classification model using accuracy and confusion matrices  

---

## 📂 Files in This Repository

### 📝 mnist_from_scratch.ipynb
The main notebook containing the full implementation of a neural network from scratch using NumPy. It includes EDA, training, and evaluation.

### 📁 images/
EDA plots used for data understanding:
- EDA 1: Sample MNIST Digits(`eda_plot_1.png`)
- EDA 2: Distribution of Digits in Training Data (`eda_plot_2.png`)
- EDA 3: Pixel Intensity Heatmap (`eda_plot_3.png`)
- EDA 4: Pixel Intensity Distribution (`eda_plot_4.png`)
- EDA 5: Confusion Matrix Test Set(`eda_plot_5.png`)
- EDA 6: Sample Predictions on Test Set(`eda_plot_6.png`)

