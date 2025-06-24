# Workshop Data Directory

This directory contains sample datasets for the Week 1 workshop.

## Built-in Datasets

The workshop primarily uses scikit-learn's built-in datasets:

### Wine Dataset
- **Size**: 178 samples, 13 features
- **Classes**: 3 wine types
- **Use**: Main workshop classification example
- **Access**: `from sklearn.datasets import load_wine`

### Iris Dataset  
- **Size**: 150 samples, 4 features
- **Classes**: 3 iris species
- **Use**: Basic ML examples and verification
- **Access**: `from sklearn.datasets import load_iris`

### Breast Cancer Dataset
- **Size**: 569 samples, 30 features
- **Classes**: 2 (malignant, benign)
- **Use**: Practice exercises
- **Access**: `from sklearn.datasets import load_breast_cancer`

## Custom Sample Data

### sample_data.csv
A synthetic dataset created for advanced exercises:
- **Size**: 1000 samples, 10 features
- **Type**: Mixed numeric and categorical
- **Use**: Data preprocessing practice

```python
# Loading sample data
import pandas as pd
df = pd.read_csv('data/sample_data.csv')
```

## Creating Your Own Data

### Synthetic Classification Data
```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_classes=3,
    random_state=42
)
```

### Synthetic Regression Data
```python
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=0.1,
    random_state=42
)
```

## Data Ethics Note

All datasets used in this workshop are:
- ✅ Publicly available
- ✅ Appropriate for educational use
- ✅ Free from sensitive personal information
- ✅ Well-documented and researched

Always consider data ethics and privacy when working with real-world datasets!