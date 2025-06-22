# Week 3 Resource Guides

## sklearn_algorithms_guide.md

# Scikit-learn Algorithms Comprehensive Reference

## 🎯 Classification Algorithms

### Logistic Regression
**Best for:** Linear relationships, baseline models, interpretability

**Strengths:**
- Fast training and prediction
- Probabilistic outputs
- No hyperparameter tuning required
- Highly interpretable coefficients
- Works well with scaled features

**Weaknesses:**
- Assumes linear relationship
- Sensitive to outliers
- Requires feature scaling
- May underfit complex data

**Key Parameters:**
```python
LogisticRegression(
    C=1.0,              # Regularization strength (smaller = more regularization)
    penalty='l2',       # 'l1', 'l2', 'elasticnet', 'none'
    solver='lbfgs',     # 'liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'
    max_iter=100,       # Maximum iterations
    multi_class='auto', # 'ovr', 'multinomial', 'auto'
    random_state=42
)
```

**When to Use:**
- Binary or multi-class classification
- Need probabilistic predictions
- Interpretability is important
- Linear decision boundary is appropriate
- Baseline model establishment

---

### Random Forest Classifier
**Best for:** Complex patterns, feature importance, robust predictions

**Strengths:**
- Handles non-linear relationships
- Built-in feature importance
- Resistant to overfitting
- Works with mixed data types
- No need for feature scaling

**Weaknesses:**
- Can be slow on large datasets
- Memory intensive
- Less interpretable than linear models
- May overfit with very noisy data

**Key Parameters:**
```python
RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=None,          # Maximum depth of trees
    min_samples_split=2,     # Min samples to split internal node
    min_samples_leaf=1,      # Min samples in leaf node
    max_features='sqrt',     # Features to consider for best split
    bootstrap=True,          # Bootstrap sampling
    random_state=42,
    n_jobs=-1               # Parallel processing
)
```

**Hyperparameter Tuning Ranges:**
- `n_estimators`: [50, 100, 200, 500]
- `max_depth`: [None, 10, 20, 30]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ['sqrt', 'log2', None]

---

### Gradient Boosting Classifier
**Best for:** High accuracy, complex patterns, tabular data competitions

**Strengths:**
- Often achieves highest accuracy
- Handles complex interactions
- Built-in feature importance
- Good with imbalanced data
- Sequential learning

**Weaknesses:**
- Prone to overfitting
- Sensitive to hyperparameters
- Computationally expensive
- Requires careful tuning

**Key Parameters:**
```python
GradientBoostingClassifier(
    n_estimators=100,        # Number of boosting stages
    learning_rate=0.1,       # Learning rate shrinks contribution
    max_depth=3,             # Maximum depth of trees
    min_samples_split=2,     # Min samples to split
    min_samples_leaf=1,      # Min samples in leaf
    subsample=1.0,           # Fraction of samples for fitting
    random_state=42
)
```

**Hyperparameter Tuning Ranges:**
- `n_estimators`: [50, 100, 200, 500]
- `learning_rate`: [0.01, 0.1, 0.2, 0.3]
- `max_depth`: [3, 5, 7, 9]
- `subsample`: [0.8, 0.9, 1.0]

---

### Support Vector Machine (SVM)
**Best for:** High-dimensional data, non-linear patterns, robust boundaries

**Strengths:**
- Effective in high dimensions
- Memory efficient
- Versatile (different kernels)
- Works well with clear margins

**Weaknesses:**
- Slow on large datasets
- Requires feature scaling
- No probabilistic output (without calibration)
- Sensitive to feature scaling

**Key Parameters:**
```python
SVC(
    C=1.0,                  # Regularization parameter
    kernel='rbf',           # 'linear', 'poly', 'rbf', 'sigmoid'
    degree=3,               # Degree for poly kernel
    gamma='scale',          # Kernel coefficient
    probability=False,      # Enable probability estimates
    random_state=42
)
```

**Hyperparameter Tuning Ranges:**
- `C`: [0.1, 1, 10, 100]
- `gamma`: ['scale', 'auto', 0.001, 0.01, 0.1, 1]
- `kernel`: ['rbf', 'poly', 'linear']

---

### Decision Tree Classifier
**Best for:** Interpretability, rule extraction, feature selection

**Strengths:**
- Highly interpretable
- No assumptions about data distribution
- Handles both numerical and categorical data
- Can capture non-linear relationships
- Built-in feature selection

**Weaknesses:**
- Prone to overfitting
- Unstable (small changes can create different tree)
- Biased toward features with more levels
- Not optimal for numerical prediction

**Key Parameters:**
```python
DecisionTreeClassifier(
    criterion='gini',        # 'gini' or 'entropy'
    max_depth=None,          # Maximum depth
    min_samples_split=2,     # Min samples to split
    min_samples_leaf=1,      # Min samples in leaf
    max_features=None,       # Max features to consider
    random_state=42
)
```

---

### Naive Bayes
**Best for:** Text classification, categorical features, baseline models

**Strengths:**
- Fast training and prediction
- Works well with small datasets
- Good performance with irrelevant features
- Handles multi-class naturally
- Simple and interpretable

**Weaknesses:**
- Strong independence assumption
- Can be outperformed by more sophisticated methods
- Requires smoothing for zero probabilities
- Sensitive to skewed data

**Implementation:**
```python
# For continuous features
GaussianNB()

# For discrete features
MultinomialNB(alpha=1.0)

# For binary features
BernoulliNB(alpha=1.0)
```

---

## 🎯 Regression Algorithms

### Linear Regression
**Best for:** Baseline models, interpretability, linear relationships

**Implementation:**
```python
LinearRegression(
    fit_intercept=True,      # Calculate intercept
    normalize=False,         # Deprecated, use StandardScaler
    copy_X=True,            # Copy X or overwrite
    n_jobs=None             # Parallel processing
)
```

---

### Ridge Regression
**Best for:** Multicollinearity, regularization, many features

**Key Parameters:**
```python
Ridge(
    alpha=1.0,              # Regularization strength
    fit_intercept=True,     # Calculate intercept
    normalize=False,        # Deprecated
    solver='auto',          # 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
    random_state=42
)
```

**Hyperparameter Tuning:**
- `alpha`: [0.01, 0.1, 1.0, 10.0, 100.0]

---

### Lasso Regression
**Best for:** Feature selection, sparse solutions, interpretability

**Key Parameters:**
```python
Lasso(
    alpha=1.0,              # Regularization strength
    fit_intercept=True,     # Calculate intercept
    normalize=False,        # Deprecated
    max_iter=1000,          # Maximum iterations
    selection='cyclic',     # 'cyclic' or 'random'
    random_state=42
)
```

**Hyperparameter Tuning:**
- `alpha`: [0.001, 0.01, 0.1, 1.0, 10.0]

---

### Random Forest Regressor
**Best for:** Non-linear relationships, feature importance, robust predictions

**Key Parameters:**
```python
RandomForestRegressor(
    n_estimators=100,        # Number of trees
    criterion='squared_error', # 'squared_error', 'absolute_error', 'poisson'
    max_depth=None,          # Maximum depth
    min_samples_split=2,     # Min samples to split
    min_samples_leaf=1,      # Min samples in leaf
    bootstrap=True,          # Bootstrap sampling
    random_state=42,
    n_jobs=-1
)
```

---

## 🎯 Algorithm Selection Guide

### Problem Type Decision Tree
```
1. Classification vs Regression?
   ├── Classification
   │   ├── Linear separable? → Logistic Regression
   │   ├── High accuracy needed? → Gradient Boosting
   │   ├── Interpretability important? → Decision Tree
   │   ├── Many features? → Random Forest
   │   └── Text data? → Naive Bayes
   └── Regression
       ├── Linear relationship? → Linear Regression
       ├── Many features? → Ridge/Lasso
       ├── Feature selection needed? → Lasso
       ├── Non-linear patterns? → Random Forest
       └── High accuracy needed? → Gradient Boosting
```

### Dataset Size Guidelines
- **Small (< 1K samples)**: Naive Bayes, Linear models, kNN
- **Medium (1K - 100K)**: Random Forest, SVM, Gradient Boosting
- **Large (> 100K)**: Linear models, SGD, ensemble methods

### Feature Count Guidelines
- **Few features (< 10)**: Any algorithm
- **Medium features (10-1000)**: Tree-based, SVM
- **Many features (> 1000)**: Linear models with regularization

---

## 🎯 Performance Optimization Tips

### Memory Usage
1. **Use sparse matrices** for high-dimensional data
2. **Reduce n_estimators** for ensemble methods
3. **Use partial_fit** for incremental learning
4. **Consider feature selection** to reduce dimensionality

### Speed Optimization
1. **Use n_jobs=-1** for parallel processing
2. **Set random_state** for reproducibility
3. **Use warm_start** for iterative training
4. **Consider simpler models** for real-time predictions

### Hyperparameter Tuning Strategy
1. **Start with default parameters**
2. **Use RandomizedSearchCV** for initial exploration
3. **Use GridSearchCV** for fine-tuning
4. **Implement early stopping** for iterative algorithms
5. **Use nested cross-validation** for unbiased estimates

---

## 🎯 Common Pitfalls and Solutions

### Data Leakage
- **Problem**: Using future information to predict past
- **Solution**: Proper train/validation/test splits, time-aware splits

### Overfitting
- **Problem**: Model performs well on training but poorly on test
- **Solution**: Cross-validation, regularization, simpler models

### Underfitting
- **Problem**: Model performs poorly on both training and test
- **Solution**: More complex models, feature engineering, more data

### Imbalanced Classes
- **Problem**: Unequal class distributions
- **Solution**: class_weight='balanced', SMOTE, stratified sampling

### Feature Scaling
- **Problem**: Features with different scales
- **Solution**: StandardScaler, MinMaxScaler, RobustScaler

---

---

## metaflow_ml_patterns.md

# Metaflow ML Patterns for Supervised Learning

## 🌊 Core ML Pipeline Patterns

### Pattern 1: Parallel Algorithm Training
**Use Case**: Train multiple algorithms simultaneously to compare performance

```python
from metaflow import FlowSpec, step, foreach, resources

class MultiAlgorithmFlow(FlowSpec):
    @step
    def start(self):
        # Define algorithms to train in parallel
        self.algorithms = [
            'logistic_regression',
            'random_forest',
            'gradient_boosting',
            'svm'
        ]
        self.next(self.train_algorithm, foreach='algorithms')
    
    @resources(memory=4000, cpu=2)
    @step
    def train_algorithm(self):
        algorithm_name = self.input
        # Train specific algorithm
        # Store results in self.model_results
        self.next(self.compare_models)
    
    @step
    def compare_models(self, inputs):
        # Aggregate results from all parallel branches
        all_results = {}
        for input_flow in inputs:
            all_results[input_flow.input] = input_flow.model_results
        self.aggregated_results = all_results
        self.next(self.end)
```

**Benefits:**
- Parallel execution reduces total training time
- Easy comparison of multiple algorithms
- Resource isolation per algorithm
- Automatic result aggregation

---

### Pattern 2: Nested Cross-Validation
**Use Case**: Unbiased model selection and hyperparameter tuning

```python
class NestedCVFlow(FlowSpec):
    @step
    def start(self):
        # Outer CV for model selection
        self.outer_folds = list(range(5))
        self.next(self.outer_cv_fold, foreach='outer_folds')
    
    @step
    def outer_cv_fold(self):
        self.current_outer_fold = self.input
        # Create train/val split for current fold
        # Define hyperparameter combinations
        self.param_combinations = self._get_param_grid()
        self.next(self.inner_cv, foreach='param_combinations')
    
    @resources(memory=2000, cpu=1)
    @step
    def inner_cv(self):
        # Inner CV for hyperparameter tuning
        params = self.input
        # Perform 5-fold CV with current parameters
        # Store CV score in self.cv_score
        self.next(self.select_best_params)
    
    @step
    def select_best_params(self, inputs):
        # Select best parameters from inner CV
        best_params = max(inputs, key=lambda x: x.cv_score)
        self.best_params = best_params.input
        self.best_score = best_params.cv_score
        self.next(self.train_final_model)
    
    @step
    def train_final_model(self):
        # Train model with best parameters on full outer fold
        self.next(self.aggregate_outer_results)
    
    @step
    def aggregate_outer_results(self, inputs):
        # Aggregate results from all outer folds
        self.final_cv_score = np.mean([inp.outer_score for inp in inputs])
        self.next(self.end)
```

---

### Pattern 3: Feature Engineering Pipeline
**Use Case**: Systematic feature transformation and selection

```python
class FeatureEngineeringFlow(FlowSpec):
    @step
    def start(self):
        self.feature_strategies = [
            'polynomial_features',
            'feature_selection',
            'dimensionality_reduction',
            'interaction_features'
        ]
        self.next(self.engineer_features, foreach='feature_strategies')
    
    @step
    def engineer_features(self):
        strategy = self.input
        if strategy == 'polynomial_features':
            self._create_polynomial_features()
        elif strategy == 'feature_selection':
            self._select_best_features()
        # etc.
        self.next(self.train_with_features)
    
    @resources(memory=6000, cpu=2)
    @step
    def train_with_features(self):
        # Train model with engineered features
        # Compare performance
        self.next(self.compare_feature_strategies)
    
    @step
    def compare_feature_strategies(self, inputs):
        # Select best feature engineering strategy
        best_strategy = max(inputs, key=lambda x: x.performance_score)
        self.best_features = best_strategy.engineered_features
        self.next(self.end)
```

---

### Pattern 4: Incremental Learning
**Use Case**: Handle large datasets or streaming data

```python
class IncrementalLearningFlow(FlowSpec):
    batch_size = Parameter('batch_size', default=1000)
    
    @step
    def start(self):
        # Split data into batches
        self.data_batches = self._create_batches()
        self.model_state = None
        self.next(self.process_batch, foreach='data_batches')
    
    @step
    def process_batch(self):
        batch_data = self.input
        if self.model_state is None:
            # Initialize model on first batch
            self.model = self._initialize_model()
        else:
            # Load previous model state
            self.model = self._load_model_state(self.model_state)
        
        # Train incrementally on current batch
        self.model.partial_fit(batch_data.X, batch_data.y)
        self.model_state = self._save_model_state(self.model)
        self.next(self.aggregate_model)
    
    @step
    def aggregate_model(self, inputs):
        # Use final model state
        final_input = inputs[-1]  # Last batch
        self.final_model = final_input.model
        self.next(self.end)
```

---

### Pattern 5: Model Ensemble
**Use Case**: Combine multiple models for better performance

```python
class EnsembleFlow(FlowSpec):
    @step
    def start(self):
        self.base_models = [
            'random_forest',
            'gradient_boosting',
            'svm',
            'logistic_regression'
        ]
        self.next(self.train_base_models, foreach='base_models')
    
    @resources(memory=4000, cpu=2)
    @step
    def train_base_models(self):
        model_type = self.input
        # Train base model
        # Generate out-of-fold predictions
        self.base_predictions = self._get_oof_predictions()
        self.trained_model = self._train_model(model_type)
        self.next(self.train_meta_model)
    
    @step
    def train_meta_model(self, inputs):
        # Collect all base model predictions
        base_predictions = np.column_stack([
            inp.base_predictions for inp in inputs
        ])
        
        # Train meta-model on base predictions
        self.meta_model = LogisticRegression()
        self.meta_model.fit(base_predictions, self.y_train)
        
        # Store base models for final ensemble
        self.base_models_trained = [inp.trained_model for inp in inputs]
        self.next(self.end)
```

---

## 🎯 Resource Management Patterns

### Memory Optimization
```python
@resources(memory=8000)  # 8GB RAM
@step
def memory_intensive_step(self):
    # Large dataset processing
    # Feature engineering
    # Model training with large datasets
    pass

@resources(memory=2000)  # 2GB RAM
@step  
def lightweight_step(self):
    # Simple computations
    # Model evaluation
    # Result aggregation
    pass
```

### CPU Optimization
```python
@resources(cpu=8)  # 8 CPU cores
@step
def parallel_computation(self):
    # Hyperparameter grid search
    # Cross-validation
    # Ensemble training
    # Set n_jobs=-1 in sklearn
    pass

@resources(cpu=1)  # Single core
@step
def sequential_step(self):
    # Model serialization
    # Simple preprocessing
    # Report generation
    pass
```

### GPU Support
```python
@resources(memory=16000, cpu=4, gpu=1)
@step
def gpu_training(self):
    # Deep learning models
    # Large-scale computations
    # GPU-accelerated algorithms
    pass
```

---

## 🎯 Error Handling Patterns

### Robust Training with Fallbacks
```python
@catch(var='training_error')
@step
def robust_training(self):
    try:
        # Primary training approach
        self.model = self._train_complex_model()
        self.training_success = True
    except Exception as e:
        # Fallback to simpler model
        self.model = self._train_simple_model()
        self.training_success = False
        self.training_error = str(e)
    
    self.next(self.evaluate_model)

@step
def evaluate_model(self):
    if hasattr(self, 'training_error'):
        print(f"Warning: Fell back to simple model due to: {self.training_error}")
    
    # Proceed with evaluation
    self.next(self.end)
```

### Model Validation Gates
```python
@step
def validation_gate(self):
    model_score = self._evaluate_model()
    
    if model_score < self.minimum_threshold:
        # Reject model and retrain
        self.next(self.retrain_model)
    else:
        # Accept model and proceed
        self.next(self.deploy_model)

@step
def retrain_model(self):
    # Implement retraining logic
    # Try different algorithms or parameters
    self.next(self.validation_gate)
```

---

## 🎯 Data Versioning Patterns

### Dataset Snapshots
```python
from metaflow import current

@step
def load_data(self):
    # Version the dataset
    self.data_version = current.run_id
    self.dataset = self._load_and_preprocess_data()
    
    # Store data fingerprint
    self.data_fingerprint = self._compute_data_hash()
    
    self.next(self.split_data)
```

### Model Artifacts
```python
@step
def save_model_artifacts(self):
    # Save multiple model formats
    self.model_pickle = pickle.dumps(self.model)
    self.model_joblib = joblib.dumps(self.model)
    
    # Save model metadata
    self.model_metadata = {
        'algorithm': self.algorithm_name,
        'parameters': self.model.get_params(),
        'training_time': self.training_time,
        'feature_names': self.feature_names,
        'target_names': self.target_names
    }
    
    # Save performance metrics
    self.performance_metrics = {
        'accuracy': self.accuracy,
        'precision': self.precision,
        'recall': self.recall,
        'f1_score': self.f1_score
    }
    
    self.next(self.end)
```

---

## 🎯 Monitoring and Logging Patterns

### Performance Tracking
```python
@step
def track_performance(self):
    from datetime import datetime
    
    # Log performance metrics
    self.performance_log = {
        'timestamp': datetime.now().isoformat(),
        'run_id': current.run_id,
        'model_type': self.model_type,
        'dataset_size': len(self.X_train),
        'feature_count': self.X_train.shape[1],
        'training_time': self.training_time,
        'memory_usage': self._get_memory_usage(),
        'cpu_usage': self._get_cpu_usage(),
        'accuracy': self.accuracy,
        'cv_score': self.cv_score
    }
    
    # Optional: Send to external monitoring system
    self._send_to_monitoring(self.performance_log)
    
    self.next(self.end)
```

### Model Drift Detection
```python
@step
def detect_drift(self):
    if hasattr(self, 'previous_model'):
        # Compare current model with previous
        feature_importance_drift = self._compare_feature_importance()
        performance_drift = self._compare_performance()
        
        self.drift_detected = (
            feature_importance_drift > self.drift_threshold or
            performance_drift > self.performance_threshold
        )
        
        if self.drift_detected:
            print("Model drift detected! Consider retraining.")
    
    self.next(self.end)
```

---

## 🎯 Advanced Patterns

### A/B Testing Framework
```python
class ABTestingFlow(FlowSpec):
    @step
    def start(self):
        self.model_variants = ['model_a', 'model_b']
        self.next(self.train_variants, foreach='model_variants')
    
    @step
    def train_variants(self):
        variant = self.input
        # Train different model configurations
        self.next(self.deploy_variant)
    
    @step
    def deploy_variant(self):
        # Deploy variant for A/B testing
        # Collect performance metrics
        self.next(self.collect_ab_results)
    
    @step
    def collect_ab_results(self, inputs):
        # Compare A/B test results
        # Select winning variant
        self.next(self.end)
```

### Hyperparameter Optimization
```python
class HyperparameterOptimizationFlow(FlowSpec):
    @step
    def start(self):
        # Generate hyperparameter combinations
        self.param_space = self._generate_param_space()
        self.next(self.evaluate_params, foreach='param_space')
    
    @resources(memory=4000, cpu=2)
    @step
    def evaluate_params(self):
        params = self.input
        # Train and evaluate model with current parameters
        score = self._cross_validate_model(params)
        self.param_score = score
        self.next(self.select_best_params)
    
    @step
    def select_best_params(self, inputs):
        # Select best parameters
        best_params = max(inputs, key=lambda x: x.param_score)
        self.optimal_params = best_params.input
        self.next(self.final_training)
```

---

## 🎯 Best Practices

### 1. Resource Allocation
- Use `@resources` decorator appropriately
- Start with conservative resource estimates
- Monitor actual usage and adjust
- Consider cloud costs vs. performance

### 2. Parallel Execution
- Use `@foreach` for independent tasks
- Ensure data dependencies are clear
- Avoid shared state between parallel branches
- Aggregate results systematically

### 3. Error Handling
- Use `@catch` for expected failures
- Implement fallback strategies
- Log errors for debugging
- Validate inputs and outputs

### 4. Data Management
- Version datasets and models
- Use consistent data splits
- Implement data validation
- Track data lineage

### 5. Reproducibility
- Set random seeds consistently
- Version code and dependencies
- Document parameter choices
- Use Metaflow's built-in versioning

---

---

## evaluation_metrics_guide.md

# Evaluation Metrics Comprehensive Guide

## 🎯 Classification Metrics

### Basic Metrics

#### Accuracy
**Definition**: Fraction of predictions that match the true labels
```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**When to Use:**
- Balanced datasets
- Equal cost for all types of errors
- General performance overview

**When NOT to Use:**
- Imbalanced datasets
- Different costs for false positives vs false negatives
- Medical diagnosis, fraud detection

**Interpretation:**
- Range: [0, 1], higher is better
- 0.95+ = Excellent
- 0.90-0.95 = Very Good
- 0.80-0.90 = Good
- 0.70-0.80 = Fair
- <0.70 = Poor

**Code Example:**
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

---

#### Precision
**Definition**: Fraction of positive predictions that are actually positive
```python
precision = TP / (TP + FP)
```

**When to Use:**
- Cost of false positives is high
- Spam detection (don't want to mark good emails as spam)
- Medical tests (don't want to alarm healthy patients)

**Interpretation:**
- Range: [0, 1], higher is better
- Answers: "Of all positive predictions, how many were correct?"

**Code Example:**
```python
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred, average='weighted')
```

---

#### Recall (Sensitivity)
**Definition**: Fraction of actual positives that are correctly identified
```python
recall = TP / (TP + FN)
```

**When to Use:**
- Cost of false negatives is high
- Disease detection (don't want to miss sick patients)
- Fraud detection (don't want to miss fraud cases)

**Interpretation:**
- Range: [0, 1], higher is better
- Answers: "Of all actual positives, how many were found?"

**Code Example:**
```python
from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred, average='weighted')
```

---

#### F1-Score
**Definition**: Harmonic mean of precision and recall
```python
f1 = 2 * (precision * recall) / (precision + recall)
```

**When to Use:**
- Need balance between precision and recall
- Imbalanced datasets
- General metric for binary classification

**Interpretation:**
- Range: [0, 1], higher is better
- Good balance when both precision and recall are important

**Code Example:**
```python
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='weighted')
```

---

### Advanced Classification Metrics

#### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
**Definition**: Area under the ROC curve (TPR vs FPR)

**When to Use:**
- Binary classification
- Comparing model discrimination ability
- Threshold-independent evaluation

**Interpretation:**
- Range: [0, 1]
- 0.5 = Random guessing
- 0.7-0.8 = Acceptable
- 0.8-0.9 = Excellent
- 0.9+ = Outstanding

**Code Example:**
```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# For binary classification
auc = roc_auc_score(y_true, y_prob)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
```

#### Multi-class ROC-AUC
```python
# One-vs-Rest approach
auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

# One-vs-One approach  
auc_ovo = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
```

---

#### Precision-Recall AUC
**Definition**: Area under the Precision-Recall curve

**When to Use:**
- Imbalanced datasets
- When positive class is rare
- Focus on positive class performance

**Code Example:**
```python
from sklearn.metrics import average_precision_score, precision_recall_curve

# Average precision
avg_precision = average_precision_score(y_true, y_prob)

# Plot PR curve
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
```

---

#### Cohen's Kappa
**Definition**: Agreement between predictions and truth, accounting for chance
```python
kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
```

**When to Use:**
- Imbalanced datasets
- Inter-rater reliability
- Accounting for chance agreement

**Interpretation:**
- Range: [-1, 1]
- < 0 = Poor agreement
- 0.0-0.20 = Slight agreement
- 0.21-0.40 = Fair agreement
- 0.41-0.60 = Moderate agreement
- 0.61-0.80 = Substantial agreement
- 0.81-1.00 = Almost perfect agreement

**Code Example:**
```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_true, y_pred)
```

---

#### Matthews Correlation Coefficient (MCC)
**Definition**: Correlation between predictions and truth
```python
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**When to Use:**
- Imbalanced datasets
- Binary classification
- Single metric summary

**Interpretation:**
- Range: [-1, 1]
- -1 = Total disagreement
- 0 = Random prediction
- 1 = Perfect prediction

**Code Example:**
```python
from sklearn.metrics import matthews_corrcoef
mcc = matthews_corrcoef(y_true, y_pred)
```

---

### Multi-class Averaging Strategies

#### Macro Average
- Calculate metric for each class separately
- Take unweighted mean
- Treats all classes equally

```python
f1_macro = f1_score(y_true, y_pred, average='macro')
```

#### Weighted Average
- Calculate metric for each class separately
- Take weighted mean by class support
- Accounts for class imbalance

```python
f1_weighted = f1_score(y_true, y_pred, average='weighted')
```

#### Micro Average
- Aggregate true positives, false positives, false negatives globally
- Calculate metric from aggregated counts

```python
f1_micro = f1_score(y_true, y_pred, average='micro')
```

---

## 🎯 Regression Metrics

### Basic Regression Metrics

#### Mean Squared Error (MSE)
**Definition**: Average of squared differences between predictions and actuals
```python
MSE = (1/n) * Σ(y_true - y_pred)²
```

**When to Use:**
- Penalize large errors more
- Differentiable for optimization
- General regression evaluation

**Interpretation:**
- Range: [0, ∞], lower is better
- Same units as target variable squared
- Sensitive to outliers

**Code Example:**
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

---

#### Root Mean Squared Error (RMSE)
**Definition**: Square root of MSE
```python
RMSE = sqrt(MSE)
```

**When to Use:**
- Same units as target variable
- Interpretable scale
- Common reporting metric

**Interpretation:**
- Range: [0, ∞], lower is better
- Average prediction error in original units

**Code Example:**
```python
import numpy as np
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

---

#### Mean Absolute Error (MAE)
**Definition**: Average of absolute differences
```python
MAE = (1/n) * Σ|y_true - y_pred|
```

**When to Use:**
- Robust to outliers
- Linear penalty for errors
- Easy to interpret

**Interpretation:**
- Range: [0, ∞], lower is better
- Average absolute prediction error

**Code Example:**
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

---

#### R-squared (R²)
**Definition**: Proportion of variance explained by the model
```python
R² = 1 - (SS_res / SS_tot)
SS_res = Σ(y_true - y_pred)²
SS_tot = Σ(y_true - y_mean)²
```

**When to Use:**
- General model performance
- Comparing models on same dataset
- Understanding explained variance

**Interpretation:**
- Range: (-∞, 1]
- 1.0 = Perfect prediction
- 0.0 = Same as predicting mean
- <0 = Worse than predicting mean

**Benchmarks:**
- 0.9+ = Excellent
- 0.7-0.9 = Good
- 0.5-0.7 = Moderate
- <0.5 = Poor

**Code Example:**
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

---

#### Mean Absolute Percentage Error (MAPE)
**Definition**: Average absolute percentage error
```python
MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
```

**When to Use:**
- Scale-independent comparison
- Business reporting
- Relative error assessment

**Interpretation:**
- Range: [0, ∞], lower is better
- Percentage error
- Cannot handle zero values in y_true

**Code Example:**
```python
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_true, y_pred)
```

---

#### Explained Variance Score
**Definition**: Variance explained by predictions
```python
explained_var = 1 - Var(y_true - y_pred) / Var(y_true)
```

**When to Use:**
- Similar to R² but doesn't account for bias
- Comparing model variance explanations

**Code Example:**
```python
from sklearn.metrics import explained_variance_score
explained_var = explained_variance_score(y_true, y_pred)
```

---

### Advanced Regression Metrics

#### Symmetric Mean Absolute Percentage Error (SMAPE)
**Definition**: Symmetric version of MAPE
```python
SMAPE = (100/n) * Σ|y_true - y_pred| / ((|y_true| + |y_pred|) / 2)
```

**Advantages:**
- Handles zero values better than MAPE
- Symmetric (treats over/under-prediction equally)

---

#### Mean Squared Logarithmic Error (MSLE)
**Definition**: MSE in log space
```python
MSLE = (1/n) * Σ(log(1 + y_true) - log(1 + y_pred))²
```

**When to Use:**
- Target variable has exponential growth
- Penalize under-prediction more than over-prediction
- Relative errors are important

**Code Example:**
```python
from sklearn.metrics import mean_squared_log_error
msle = mean_squared_log_error(y_true, y_pred)
```

---

## 🎯 Cross-Validation Strategies

### Basic Cross-Validation
```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

### Stratified Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

# Maintains class distribution in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro')
```

### Time Series Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit

# Respects temporal order
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
```

### Custom Scoring Functions
```python
from sklearn.metrics import make_scorer

def custom_metric(y_true, y_pred):
    # Your custom metric logic
    return score

custom_scorer = make_scorer(custom_metric, greater_is_better=True)
cv_scores = cross_val_score(model, X, y, cv=5, scoring=custom_scorer)
```

---

## 🎯 Statistical Significance Testing

### Paired t-test for Model Comparison
```python
from scipy.stats import ttest_rel

# Compare two models using CV scores
model1_scores = cross_val_score(model1, X, y, cv=5)
model2_scores = cross_val_score(model2, X, y, cv=5)

t_stat, p_value = ttest_rel(model1_scores, model2_scores)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")

if p_value < 0.05:
    print("Statistically significant difference")
else:
    print("No statistically significant difference")
```

### McNemar's Test for Classification
```python
from statsmodels.stats.contingency_tables import mcnemar

# Create contingency table
contingency_table = [[model1_correct_model2_wrong,
                     model1_wrong_model2_correct],
                    [model1_wrong_model2_wrong,
                     model1_correct_model2_correct]]

result = mcnemar(contingency_table, exact=True)
print(f"McNemar's test p-value: {result.pvalue:.3f}")
```

---

## 🎯 Learning Curves

### Basic Learning Curve
```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.legend()
plt.title('Learning Curves')
```

### Validation Curves
```python
from sklearn.model_selection import validation_curve

param_range = [0.1, 1.0, 10.0, 100.0]
train_scores, val_scores = validation_curve(
    model, X, y, param_name='C', param_range=param_range,
    cv=5, scoring='accuracy'
)

# Plot validation curves
plt.plot(param_range, train_scores.mean(axis=1), 'o-', label='Training')
plt.plot(param_range, val_scores.mean(axis=1), 'o-', label='Validation')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.legend()
```

---

## 🎯 Metric Selection Guide

### Classification Metric Selection
```
Decision Tree:
├── Balanced Dataset?
│   ├── Yes → Accuracy
│   └── No → F1-Score (macro/weighted)
├── Binary Classification?
│   ├── Cost-sensitive?
│   │   ├── High FP cost → Precision
│   │   ├── High FN cost → Recall
│   │   └── Balanced → F1-Score
│   └── Threshold-independent → ROC-AUC
├── Multi-class?
│   ├── Balanced → Accuracy
│   ├── Imbalanced → F1-macro
│   └── Class distribution matters → F1-weighted
└── Imbalanced Dataset?
    ├── Severe imbalance → Precision-Recall AUC
    ├── Multiple classes → Cohen's Kappa
    └── Binary → MCC
```

### Regression Metric Selection
```
Decision Tree:
├── Outliers present?
│   ├── Yes → MAE (robust)
│   └── No → RMSE (sensitive)
├── Scale matters?
│   ├── Yes → R²
│   └── No → MAPE
├── Business interpretation?
│   ├── Absolute error → MAE
│   ├── Relative error → MAPE
│   └── Explained variance → R²
└── Model comparison?
    └── Same dataset → R²
    └── Different datasets → MAPE
```

---

## 🎯 Best Practices

### 1. Multiple Metrics
- Never rely on a single metric
- Use complementary metrics
- Consider business context

### 2. Cross-Validation
- Always use CV for robust estimates
- Choose appropriate CV strategy
- Report mean ± standard deviation

### 3. Baseline Comparison
- Compare against simple baselines
- Random classifier/regressor
- Most frequent class/mean prediction

### 4. Statistical Testing
- Test significance when comparing models
- Use appropriate statistical tests
- Consider multiple testing correction

### 5. Visualization
- Plot confusion matrices
- Create ROC and PR curves
- Show learning curves
- Visualize prediction distributions

### 6. Documentation
- Document metric choices and rationale
- Explain business interpretation
- Report confidence intervals
- Include baseline comparisons

---

---

## hyperparameter_tuning_guide.md

# Hyperparameter Tuning Advanced Guide

## 🎯 Hyperparameter Tuning Fundamentals

### What are Hyperparameters?
**Definition**: Parameters that are set before training and control the learning process

**Examples:**
- Learning rate, number of trees, regularization strength
- Network architecture, kernel types, distance metrics
- Not learned from data, must be specified by practitioner

### Types of Hyperparameters
1. **Continuous**: Learning rate (0.001 to 0.1)
2. **Discrete**: Number of trees (50, 100, 200)
3. **Categorical**: Kernel type ('rbf', 'linear', 'poly')
4. **Conditional**: Depend on other hyperparameters

---

## 🎯 Search Strategies

### 1. Grid Search
**Description**: Exhaustive search over specified parameter grid

**Advantages:**
- Systematic and thorough
- Guaranteed to find best combination in grid
- Easy to implement and understand
- Reproducible results

**Disadvantages:**
- Computationally expensive
- Curse of dimensionality
- Limited to predefined values
- Inefficient for continuous parameters

**Implementation:**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

**When to Use:**
- Small parameter spaces (< 100 combinations)
- Sufficient computational resources
- Need to explore all combinations systematically
- Interpretability of parameter effects is important

---

### 2. Random Search
**Description**: Randomly sample parameter combinations

**Advantages:**
- More efficient than grid search
- Better for continuous parameters
- Can run for any amount of time
- Often finds good solutions quickly

**Disadvantages:**
- No guarantee of finding optimal solution
- May miss good combinations
- Less systematic than grid search

**Implementation:**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None] + list(range(5, 25)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# Random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,  # Number of parameter combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

**When to Use:**
- Large parameter spaces
- Continuous parameters
- Limited computational budget
- Quick exploration needed

---

### 3. Bayesian Optimization
**Description**: Use probabilistic model to guide search

**Advantages:**
- Very efficient for expensive evaluations
- Balances exploration vs exploitation
- Works well with few evaluations
- Can handle noisy objectives

**Disadvantages:**
- More complex to implement
- Requires additional dependencies
- Less interpretable process

**Implementation:**
```python
# Using scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Define search space
search_space = {
    'n_estimators': Integer(50, 500),
    'max_depth': Categorical([None, 5, 10, 15, 20]),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0)
}

# Bayesian optimization
bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=search_space,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X_train, y_train)
```

**When to Use:**
- Expensive model training
- Complex parameter interactions
- Limited evaluation budget
- Need efficient optimization

---

### 4. Halving Search (Successive Halving)
**Description**: Iteratively eliminate poor performers

**Advantages:**
- Efficiently handles large parameter spaces
- Focuses computational resources on promising candidates
- Built into scikit-learn (HalvingGridSearchCV, HalvingRandomSearchCV)

**Implementation:**
```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

halving_search = HalvingRandomSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    factor=3,  # Fraction of candidates eliminated each round
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

halving_search.fit(X_train, y_train)
```

---

## 🎯 Algorithm-Specific Hyperparameter Guides

### Random Forest
**Key Parameters:**
```python
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None, 0.5],
    'bootstrap': [True, False],
    'oob_score': [True, False]  # Only when bootstrap=True
}
```

**Tuning Strategy:**
1. Start with `n_estimators` (more is usually better, but diminishing returns)
2. Tune `max_depth` and `min_samples_split` together
3. Fine-tune `min_samples_leaf` and `max_features`
4. Consider `bootstrap` and `oob_score` for final optimization

**Performance vs Resource Trade-offs:**
- `n_estimators`: Linear increase in training time
- `max_depth`: Exponential increase in memory and time
- `min_samples_split/leaf`: Minimal impact on resources

---

### Gradient Boosting
**Key Parameters:**
```python
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}
```

**Tuning Strategy:**
1. Start with `learning_rate` and `n_estimators` (inversely related)
2. Tune `max_depth` (usually 3-9 for gradient boosting)
3. Optimize `subsample` for regularization
4. Fine-tune other parameters

**Common Combinations:**
- High `learning_rate` (0.1-0.2) + Low `n_estimators` (50-100)
- Low `learning_rate` (0.01-0.05) + High `n_estimators` (200-500)

---

### Support Vector Machine
**Key Parameters:**
```python
param_grid = [
    {
        'kernel': ['linear'],
        'C': [0.1, 1, 10, 100, 1000]
    },
    {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    },
    {
        'kernel': ['poly'],
        'C': [0.1, 1, 10, 100],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
]
```

**Tuning Strategy:**
1. Start with different kernels to find best type
2. For RBF kernel: tune `C` and `gamma` together
3. For polynomial: tune `degree`, `C`, and `gamma`
4. Consider `class_weight` for imbalanced data

---

### Logistic Regression
**Key Parameters:**
```python
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'solver': ['liblinear', 'lbfgs', 'sag', 'saga'],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9],  # Only for elasticnet
    'max_iter': [100, 500, 1000]
}
```

**Solver Compatibility:**
- `liblinear`: l1, l2 penalties (small datasets)
- `lbfgs`: l2, None (large datasets)
- `sag/saga`: l1, l2, elasticnet (very large datasets)

---

### Neural Networks (MLPClassifier)
**Key Parameters:**
```python
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [200, 500, 1000]
}
```

---

## 🎯 Advanced Techniques

### 1. Nested Cross-Validation
**Purpose**: Unbiased estimate of model performance with hyperparameter tuning

```python
from sklearn.model_selection import cross_val_score, KFold

# Outer CV for performance estimation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = []

for train_idx, test_idx in outer_cv.split(X):
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    # Inner CV for hyperparameter tuning
    inner_cv = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy'
    )
    
    inner_cv.fit(X_train_outer, y_train_outer)
    best_model = inner_cv.best_estimator_
    
    # Evaluate on outer test set
    score = best_model.score(X_test_outer, y_test_outer)
    nested_scores.append(score)

print(f"Nested CV Score: {np.mean(nested_scores):.3f} ± {np.std(nested_scores):.3f}")
```

### 2. Early Stopping
**Purpose**: Prevent overfitting and reduce training time

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Split training data for early stopping
X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

gb_early_stop = GradientBoostingClassifier(
    n_estimators=1000,  # Large number
    learning_rate=0.1,
    validation_fraction=0.2,
    n_iter_no_change=5,  # Stop after 5 rounds without improvement
    random_state=42
)

gb_early_stop.fit(X_train, y_train)
print(f"Optimal number of estimators: {gb_early_stop.n_estimators_}")
```

### 3. Multi-Objective Optimization
**Purpose**: Optimize multiple objectives simultaneously

```python
from sklearn.model_selection import cross_validate

def multi_objective_score(estimator, X, y):
    """Custom scorer for multiple objectives."""
    scores = cross_validate(
        estimator, X, y, cv=3,
        scoring=['accuracy', 'f1_macro', 'roc_auc_ovr_weighted'],
        return_train_score=False
    )
    
    # Weighted combination of metrics
    combined_score = (
        0.4 * scores['test_accuracy'].mean() +
        0.4 * scores['test_f1_macro'].mean() +
        0.2 * scores['test_roc_auc_ovr_weighted'].mean()
    )
    
    return combined_score

# Use in hyperparameter tuning
from sklearn.metrics import make_scorer
multi_scorer = make_scorer(multi_objective_score, greater_is_better=True)
```

### 4. Hyperparameter Optimization with Time Budget
```python
import time
from sklearn.model_selection import RandomizedSearchCV

def time_limited_search(estimator, param_dist, X, y, time_budget_minutes=30):
    """Hyperparameter search with time constraint."""
    
    start_time = time.time()
    best_score = -np.inf
    best_params = None
    iteration = 0
    
    time_budget_seconds = time_budget_minutes * 60
    
    while (time.time() - start_time) < time_budget_seconds:
        # Sample parameters
        params = {param: np.random.choice(values) if isinstance(values, list) 
                 else values.rvs() for param, values in param_dist.items()}
        
        # Quick evaluation with smaller CV
        try:
            estimator.set_params(**params)
            scores = cross_val_score(estimator, X, y, cv=3, scoring='accuracy')
            score = scores.mean()
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                
        except Exception as e:
            continue
            
        iteration += 1
        
        if iteration % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"Iteration {iteration}, Elapsed: {elapsed:.1f}min, Best Score: {best_score:.3f}")
    
    return best_params, best_score
```

---

## 🎯 Cross-Validation Strategies for Hyperparameter Tuning

### 1. Stratified K-Fold
**Best for**: Classification with imbalanced classes
```python
from sklearn.model_selection import StratifiedKFold

stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator, param_grid, cv=stratified_cv)
```

### 2. Time Series Split
**Best for**: Time series data
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(estimator, param_grid, cv=tscv)
```

### 3. Group K-Fold
**Best for**: Data with groups that shouldn't be split
```python
from sklearn.model_selection import GroupKFold

group_cv = GroupKFold(n_splits=5)
grid_search = GridSearchCV(estimator, param_grid, cv=group_cv)
# Need to pass groups parameter to fit()
grid_search.fit(X, y, groups=groups)
```

### 4. Leave-One-Out (LOO)
**Best for**: Very small datasets
```python
from sklearn.model_selection import LeaveOneOut

loo_cv = LeaveOneOut()
grid_search = GridSearchCV(estimator, param_grid, cv=loo_cv)
```

---

## 🎯 Parallel and Distributed Tuning

### 1. Joblib Parallel Processing
```python
# Use all available cores
grid_search = GridSearchCV(
    estimator, param_grid, 
    cv=5, n_jobs=-1  # Use all cores
)

# Specify number of cores
grid_search = GridSearchCV(
    estimator, param_grid, 
    cv=5, n_jobs=4  # Use 4 cores
)
```

### 2. Dask for Distributed Computing
```python
import dask
from dask.distributed import Client
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV

# Start Dask client
client = Client('localhost:8786')

# Distributed grid search
dask_grid_search = DaskGridSearchCV(
    estimator, param_grid,
    cv=5, scoring='accuracy'
)

dask_grid_search.fit(X, y)
```

### 3. Ray Tune Integration
```python
from ray import tune
from ray.tune.sklearn import TuneSearchCV

tune_search = TuneSearchCV(
    estimator,
    param_distributions=param_dist,
    n_trials=100,
    cv=5,
    scoring='accuracy'
)

tune_search.fit(X, y)
```

---

## 🎯 Monitoring and Analysis

### 1. Learning Curve Analysis
```python
def plot_validation_curve(estimator, X, y, param_name, param_range):
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores.mean(axis=1), 'o-', label='Training')
    plt.plot(param_range, val_scores.mean(axis=1), 'o-', label='Validation')
    plt.fill_between(param_range, 
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1),
                     alpha=0.1)
    plt.fill_between(param_range,
                     val_scores.mean(axis=1) - val_scores.std(axis=1), 
                     val_scores.mean(axis=1) + val_scores.std(axis=1),
                     alpha=0.1)
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Validation Curve for {param_name}')
    plt.show()
```

### 2. Hyperparameter Importance Analysis
```python
def analyze_param_importance(grid_search_results):
    """Analyze which parameters have most impact on performance."""
    import pandas as pd
    
    results_df = pd.DataFrame(grid_search_results.cv_results_)
    
    # Extract parameter columns
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    
    # Calculate correlation between parameters and scores
    correlations = {}
    for param_col in param_cols:
        # Handle categorical parameters
        if results_df[param_col].dtype == 'object':
            # One-hot encode categorical parameters
            param_encoded = pd.get_dummies(results_df[param_col])
            for encoded_col in param_encoded.columns:
                corr = param_encoded[encoded_col].corr(results_df['mean_test_score'])
                correlations[f"{param_col}_{encoded_col}"] = abs(corr)
        else:
            corr = results_df[param_col].corr(results_df['mean_test_score'])
            correlations[param_col] = abs(corr)
    
    # Sort by importance
    sorted_importance = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("Parameter Importance (correlation with performance):")
    for param, importance in sorted_importance[:10]:
        print(f"{param}: {importance:.3f}")
    
    return sorted_importance
```

### 3. Hyperparameter Interaction Analysis
```python
def plot_param_interaction(grid_search_results, param1, param2):
    """Plot interaction between two parameters."""
    import pandas as pd
    import seaborn as sns
    
    results_df = pd.DataFrame(grid_search_results.cv_results_)
    
    # Create pivot table
    pivot_table = results_df.pivot_table(
        values='mean_test_score',
        index=f'param_{param1}',
        columns=f'param_{param2}',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
    plt.title(f'Parameter Interaction: {param1} vs {param2}')
    plt.show()
```

---

## 🎯 Best Practices

### 1. Search Strategy Selection
```python
# Decision flowchart
def choose_search_strategy(n_params, budget, param_types):
    if n_params <= 3 and budget == 'high':
        return "Grid Search"
    elif budget == 'low' or n_params > 5:
        return "Random Search"
    elif 'continuous' in param_types and budget == 'medium':
        return "Bayesian Optimization"
    else:
        return "Halving Search"
```

### 2. Parameter Range Selection
- **Start wide, then narrow**: Begin with broad ranges, refine based on results
- **Log scale for multiplicative parameters**: Learning rate, regularization strength
- **Check algorithm documentation**: Some parameters have recommended ranges

### 3. Computational Efficiency
```python
# Efficient hyperparameter tuning pipeline
def efficient_hp_tuning(estimator, X, y, time_budget=60):
    """Multi-stage hyperparameter tuning."""
    
    # Stage 1: Quick random search
    stage1_params = {param: values[:3] for param, values in full_param_grid.items()}
    quick_search = RandomizedSearchCV(
        estimator, stage1_params, n_iter=20, cv=3, n_jobs=-1
    )
    quick_search.fit(X, y)
    
    # Stage 2: Refined grid search around best parameters
    best_params = quick_search.best_params_
    refined_grid = refine_param_grid(best_params, full_param_grid)
    
    final_search = GridSearchCV(
        estimator, refined_grid, cv=5, n_jobs=-1
    )
    final_search.fit(X, y)
    
    return final_search
```

### 4. Avoiding Common Pitfalls
1. **Data leakage**: Don't use test data in hyperparameter tuning
2. **Overfitting to validation set**: Use nested CV for unbiased estimates
3. **Ignoring computational costs**: Consider training time in parameter selection
4. **Not checking convergence**: Ensure algorithms have converged
5. **Forgetting random seeds**: Set random_state for reproducibility

### 5. Documentation and Tracking
```python
# Track hyperparameter experiments
experiment_log = {
    'timestamp': datetime.now(),
    'dataset': 'wine_classification',
    'algorithm': 'RandomForest',
    'search_strategy': 'GridSearch',
    'param_grid': param_grid,
    'cv_folds': 5,
    'best_params': grid_search.best_params_,
    'best_score': grid_search.best_score_,
    'n_combinations_tried': len(grid_search.cv_results_['params']),
    'total_time': grid_search.refit_time_
}

# Save experiment log
import json
with open(f"hp_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
    json.dump(experiment_log, f, indent=2, default=str)
```

---

## 🎯 Tools and Libraries

### 1. Built-in Scikit-learn
- `GridSearchCV`: Exhaustive grid search
- `RandomizedSearchCV`: Random parameter sampling
- `HalvingGridSearchCV`: Successive halving grid search
- `HalvingRandomSearchCV`: Successive halving random search

### 2. Advanced Optimization Libraries
- **Optuna**: Tree-structured Parzen Estimator (TPE)
- **Hyperopt**: Bayesian optimization
- **Scikit-optimize**: Gaussian process optimization
- **Ray Tune**: Distributed hyperparameter tuning

### 3. AutoML Libraries
- **Auto-sklearn**: Automated machine learning
- **TPOT**: Genetic programming for ML pipelines
- **H2O AutoML**: Enterprise AutoML platform

### 4. Experiment Tracking
- **MLflow**: ML lifecycle management
- **Weights & Biases**: Experiment tracking and visualization
- **Neptune**: ML experiment management
- **TensorBoard**: Visualization toolkit

---

## 🎯 Example: Complete Hyperparameter Tuning Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

def complete_hyperparameter_tuning_pipeline():
    """Complete hyperparameter tuning pipeline example."""
    
    print("🎯 Complete Hyperparameter Tuning Pipeline")
    print("=" * 50)
    
    # 1. Load and prepare data
    wine = load_wine()
    X, y = wine.data, wine.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Data shape: {X_train_scaled.shape}")
    print(f"🎯 Classes: {np.unique(y)}")
    
    # 2. Define parameter space
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # 3. Quick random search
    print("\n🔄 Stage 1: Random Search (Quick Exploration)")
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        n_iter=50,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    start_time = datetime.now()
    random_search.fit(X_train_scaled, y_train)
    random_time = (datetime.now() - start_time).total_seconds()
    
    print(f"   ⏱️ Random search time: {random_time:.1f} seconds")
    print(f"   🏆 Best random params: {random_search.best_params_}")
    print(f"   📊 Best random score: {random_search.best_score_:.3f}")
    
    # 4. Refined grid search
    print("\n🔄 Stage 2: Grid Search (Refined)")
    
    # Create refined grid around best random parameters
    best_random_params = random_search.best_params_
    
    refined_grid = {
        'n_estimators': [max(50, best_random_params['n_estimators'] - 50),
                        best_random_params['n_estimators'],
                        best_random_params['n_estimators'] + 50],
        'max_depth': [best_random_params['max_depth']],  # Keep best
        'min_samples_split': [max(2, best_random_params['min_samples_split'] - 2),
                             best_random_params['min_samples_split'],
                             best_random_params['min_samples_split'] + 2],
        'min_samples_leaf': [max(1, best_random_params['min_samples_leaf'] - 1),
                            best_random_params['min_samples_leaf'],
                            best_random_params['min_samples_leaf'] + 1],
        'max_features': [best_random_params['max_features']]  # Keep best
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        refined_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    start_time = datetime.now()
    grid_search.fit(X_train_scaled, y_train)
    grid_time = (datetime.now() - start_time).total_seconds()
    
    print(f"   ⏱️ Grid search time: {grid_time:.1f} seconds")
    print(f"   🏆 Best grid params: {grid_search.best_params_}")
    print(f"   📊 Best grid score: {grid_search.best_score_:.3f}")
    
    # 5. Final evaluation
    print("\n📊 Final Evaluation")
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"   🎯 Test accuracy: {test_accuracy:.3f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))
    
    # 6. Feature importance
    feature_importance = pd.DataFrame({
        'feature': wine.feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 5 Feature Importances:")
    for i, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # 7. Save results
    results = {
        'best_model': best_model,
        'best_params': grid_search.best_params_,
        'test_accuracy': test_accuracy,
        'feature_importance': feature_importance,
        'random_search_time': random_time,
        'grid_search_time': grid_time,
        'scaler': scaler
    }
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    joblib.dump(results, f'tuned_model_{timestamp}.pkl')
    
    print(f"\n💾 Results saved to: tuned_model_{timestamp}.pkl")
    print("✅ Hyperparameter tuning pipeline complete!")
    
    return results

# Run the pipeline
if __name__ == "__main__":
    results = complete_hyperparameter_tuning_pipeline()
```

This comprehensive guide provides everything needed for effective hyperparameter tuning in supervised learning projects, from basic concepts to advanced techniques and complete implementation examples.