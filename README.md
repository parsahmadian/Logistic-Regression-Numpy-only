# Logistic Regression from Scratch (Python & NumPy)

This project is a **from-scratch implementation of logistic regression**  
using only **Python and NumPy**, without any machine learning libraries.

## 🎯 Goal
To deeply understand:
- How logistic regression works internally
- How the Sigmoid function maps predictions to probabilities
- How Binary Cross-Entropy (BCE) loss affects learning
- How gradient descent updates weights and bias
- The impact of learning rate and feature scaling on convergence

## 🧠 Model
The model predicts probabilities using:

\[
\hat{y} = \sigma(X \cdot w + b)
\]

Where:
- **X**: input features
- **w**: weights
- **b**: bias
- **σ**: sigmoid activation function

Loss function:
- Binary Cross-Entropy (BCE)

Optimization:
- Gradient Descent (manual implementation with NumPy)

## 📊 Results
- Training loss decreases steadily and converges
- Demonstrates how weights evolve during optimization
- Shows the effect of feature scaling on training stability
- Includes decision boundary visualization for 2D projection of multi-feature datasets

## 🔍 Key Highlights
- **From Scratch Implementation**: No scikit-learn, TensorFlow, or PyTorch used
- **Multi-feature Support**: Works with any number of features
- **Visual Insights**: Loss curves and 2D decision boundary plots (soon)
- **Understanding Overfitting & Convergence**: Shows how learning rate and data distribution affect training

## ⚠️ Notes
- Only NumPy is used; Matplotlib and Pandas is used for read file and visualization
- Features were optionally standardized to observe stability effects
- Decision boundary visualizations are 2D projections when dataset has more than 2 features

## 🛠 Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib

## 🚀 Why this project?
This project focuses on **conceptual mastery**, not just using libraries.  
It demonstrates my ability to:
- Implement core ML algorithms from scratch  
- Debug and reason about numerical issues  
- Visualize model behavior for multi-feature datasets  
- Explain model decisions clearly