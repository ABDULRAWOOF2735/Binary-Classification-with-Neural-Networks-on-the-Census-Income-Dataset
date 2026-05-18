# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset
# Binary Classification with Neural Networks on the Census Income Dataset

This project builds a Deep Learning Tabular Model using PyTorch to predict whether a person’s income is high or low based on demographic and work-related details.

The model uses:
- Categorical embeddings
- Continuous features
- Neural Networks
- CrossEntropy Loss
- Adam Optimizer

---

# Project Overview

The notebook trains a neural network on an income dataset using:
- Age
- Gender
- Education
- Marital Status
- Workclass
- Occupation
- Hours worked per week

The goal is to classify income into two categories:
- Low Income
- High Income

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# Dataset

Dataset file used:
```bash
income.csv
```

The dataset contains:
- Categorical columns
- Continuous numerical columns
- Label column for income prediction

### Features Used

#### Categorical Features
- sex
- education
- marital-status
- workclass
- occupation

#### Continuous Features
- age
- hours-per-week

#### Target
- label

---

# Project Workflow

## 1. Import Libraries

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

---

## 2. Load Dataset

```python
df = pd.read_csv('income.csv')
```

---

## 3. Data Preprocessing

- Shuffle dataset
- Convert categorical columns into category datatype
- Create category codes
- Convert data into tensors

---

## 4. Embedding Layers

Categorical features are converted into embeddings:

```python
nn.Embedding()
```

This helps the neural network learn relationships between categories.

---

## 5. Build Neural Network Model

The custom `TabularModel` contains:
- Embedding layers
- Batch normalization
- Dropout layers
- Fully connected dense layers
- ReLU activation

---

## 6. Model Training

Loss Function:

```python
nn.CrossEntropyLoss()
```

Optimizer:

```python
torch.optim.Adam()
```

The model is trained for:

```python
epochs = 300
```

---

## 7. Evaluation

The model evaluates:
- Cross Entropy Loss
- Prediction Accuracy

---

## 8. Prediction System

The notebook also includes a user input prediction function:

```python
predict_income()
```

Users can enter:
- Age
- Gender
- Education
- Occupation
- Work hours

The model predicts the income category.

---

# Setup Instructions

## Step 1: Clone Repository

```bash
git clone <repository-link>
```

---

## Step 2: Install Required Libraries

```bash
pip install torch pandas numpy matplotlib scikit-learn
```

---

## Step 3: Add Dataset

Place:

```bash
income.csv
```

inside the project folder.

---

## Step 4: Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```bash
WORKSHOP.f.ipynb
```

---

# Model Architecture

The neural network includes:
- Embedding Layers
- Linear Layers
- Batch Normalization
- Dropout
- ReLU Activation

Architecture:

```python
layers=[50]
dropout=0.4
```

---

# Results

The model achieves:
- Good classification accuracy
- Reduced loss over epochs

Loss graph is plotted using Matplotlib.

---

# Future Improvements

Possible enhancements:
- Hyperparameter tuning
- More hidden layers
- Better feature engineering
- Use larger datasets
- Deploy using Flask or Streamlit

---

# Learning Outcomes

From this project you can learn:
- PyTorch basics
- Deep Learning for tabular data
- Embedding layers
- Neural networks
- Tensor operations
- Model evaluation
- Data preprocessing

---

# Author

Developed as a Deep Learning workshop/project using PyTorch for tabular income prediction.
