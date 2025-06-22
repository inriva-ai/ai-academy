# Week 3 Resources: Supervised Learning & Metaflow Pipelines

Essential resources, references, and guides for mastering supervised learning with Metaflow and LangChain integration.

## 📚 Core Learning Resources

### Essential Reading
- **[Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)** - Chapters 3-4 (Classification & Training Models)
- **[Python Machine Learning](https://www.packtpub.com/product/python-machine-learning-third-edition/9781789955750)** - Chapters 3-5 (Classification, Data Preprocessing, Dimensionality Reduction)
- **[The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)** - Chapters 4-7 (Linear Methods, Neural Networks, Model Assessment)

### Video Tutorials
- **[Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)** - Weeks 3-6 (Classification, Neural Networks)
- **[Fast.ai Practical Deep Learning](https://course.fast.ai/)** - Lessons 1-4 (Getting Started, Deployment, Ethics)
- **[3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)** - Visual explanations

### Documentation
- **[Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)** - Supervised Learning section
- **[Metaflow Documentation](https://docs.metaflow.org/)** - Machine Learning tutorials
- **[LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)** - Model integration guides

## 🛠 Technical References

### Scikit-learn Algorithm Guide

#### Classification Algorithms
```python
# Quick reference for classification algorithms

# 1. Logistic Regression - Linear, fast, interpretable
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42, max_iter=1000)

# 2. Decision Trees - Interpretable, handles mixed data
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42, max_depth=10)

# 3. Random Forest - Robust, handles overfitting
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. Gradient Boosting - High performance, sequential learning
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# 5. Support Vector Machine - Effective for high dimensions
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=42, probability=True)

# 6. Naive Bayes - Fast, good baseline for text
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
```

#### Regression Algorithms
```python
# Quick reference for regression algorithms

# 1. Linear Regression - Simple, interpretable baseline
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# 2. Ridge Regression - L2 regularization
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)

# 3. Lasso Regression - L1 regularization, feature selection
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0, max_iter=2000)

# 4. Random Forest Regressor - Non-linear, robust
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# 5. Support Vector Regression - Non-linear relationships
from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=1.0)
```

### Evaluation Metrics Cheatsheet

#### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# Comprehensive report
report = classification_report(y_true, y_pred, target_names=class_names)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# ROC AUC (multi-class)
auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
```

#### Regression Metrics
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

# Primary metrics
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Additional metrics
mape = mean_absolute_percentage_error(y_true, y_pred)
explained_var = explained_variance_score(y_true, y_pred)
```

### Cross-Validation Strategies
```python
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, 
    TimeSeriesSplit, GroupKFold
)

# Classification - use stratified to maintain class balance
cv_scores = cross_val_score(
    model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

# Regression - standard k-fold
cv_scores = cross_val_score(
    model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42)
)

# Time series - preserve temporal order
cv_scores = cross_val_score(
    model, X, y, cv=TimeSeriesSplit(n_splits=5)
)
```

## 🌊 Metaflow Best Practices

### Pipeline Architecture Patterns

#### 1. Basic ML Pipeline
```python
class BasicMLPipeline(FlowSpec):
    @step
    def start(self):
        # Load and validate data
        
    @step
    def preprocess(self):
        # Feature engineering and scaling
        
    @step
    def train_model(self):
        # Model training
        
    @step
    def evaluate(self):
        # Model evaluation
        
    @step
    def end(self):
        # Results summary
```

#### 2. Parallel Model Training
```python
class ParallelMLPipeline(FlowSpec):
    @step
    def start(self):
        self.algorithms = ['rf', 'gb', 'svm', 'lr']
        
    @step
    def preprocess(self):
        # Data preprocessing
        
    @foreach('algorithms')
    @resources(memory=4000, cpu=2)
    @step
    def train_model(self):
        # Train each algorithm in parallel
        
    @step
    def compare_models(self, inputs):
        # Aggregate and compare results
        
    @step
    def end(self):
        # Final model selection
```

#### 3. Hyperparameter Tuning Pipeline
```python
class HPTuningPipeline(FlowSpec):
    @step
    def start(self):
        self.param_combinations = self.generate_param_grid()
        
    @foreach('param_combinations')
    @resources(memory=8000, cpu=4)
    @step
    def tune_hyperparameters(self):
        # Grid search or random search
        
    @step
    def select_best_model(self, inputs):
        # Select best hyperparameters
        
    @step
    def final_evaluation(self):
        # Evaluate best model on test set
```

### Resource Management
```python
# Memory and CPU allocation
@resources(memory=8000, cpu=4)
@step
def heavy_computation(self):
    # Resource-intensive operations

# Batch processing for cloud scaling
@batch(memory=16000, cpu=8)
@step
def cloud_training(self):
    # Large-scale model training

# Timeout for long-running operations
@timeout(seconds=3600)
@step
def long_training(self):
    # Training with time limit

# Error handling
@catch(var='training_error')
@step
def robust_training(self):
    # Graceful error handling
```

### Data Versioning and Artifacts
```python
@step
def save_model_artifacts(self):
    # Save trained models
    self.trained_models = {
        'random_forest': self.rf_model,
        'gradient_boosting': self.gb_model
    }
    
    # Save evaluation metrics
    self.evaluation_results = {
        'accuracy_scores': self.accuracies,
        'confusion_matrices': self.conf_matrices,
        'feature_importance': self.feature_importances
    }
    
    # Save preprocessed data
    self.processed_data = {
        'X_train_scaled': self.X_train_scaled,
        'X_test_scaled': self.X_test_scaled,
        'scaler': self.scaler
    }
```

## 🦜 LangChain Integration Patterns

### Model Interpretation with LLM
```python
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

def create_model_interpreter():
    llm = Ollama(model="llama3.2")
    
    prompt = PromptTemplate(
        input_variables=["model_name", "accuracy", "features"],
        template="""
        Analyze this machine learning model:
        
        Model: {model_name}
        Accuracy: {accuracy}
        Important Features: {features}
        
        Provide insights on:
        1. Performance assessment
        2. Model reliability
        3. Business recommendations
        
        Keep response concise and actionable.
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain

# Usage in Metaflow
@step
def llm_interpretation(self):
    interpreter = create_model_interpreter()
    
    for model_name, results in self.model_results.items():
        interpretation = interpreter.invoke({
            "model_name": model_name,
            "accuracy": results['accuracy'],
            "features": results['top_features']
        })
        self.interpretations[model_name] = interpretation
```

### Automated Report Generation
```python
def create_report_generator():
    llm = Ollama(model="llama3.2")
    
    prompt = PromptTemplate(
        input_variables=["best_model", "performance_summary", "recommendations"],
        template="""
        Generate an executive summary for this ML project:
        
        Best Model: {best_model}
        Performance Summary: {performance_summary}
        
        Create a business-friendly report including:
        1. Executive summary
        2. Key findings
        3. Recommendations
        4. Next steps
        
        Target audience: Business stakeholders
        """
    )
    
    return prompt | llm | StrOutputParser()
```

## 📊 Visualization Best Practices

### Model Comparison Plots
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(results_dict):
    """Create comprehensive model comparison visualization."""
    
    # Extract metrics
    models = list(results_dict.keys())
    accuracies = [results_dict[m]['accuracy'] for m in models]
    cv_means = [results_dict[m]['cv_mean'] for m in models]
    cv_stds = [results_dict[m]['cv_std'] for m in models]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Accuracy comparison
    axes[0, 0].bar(models, accuracies, color=sns.color_palette("viridis", len(models)))
    axes[0, 0].set_title('Test Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Cross-validation scores with error bars
    axes[0, 1].errorbar(models, cv_means, yerr=cv_stds, 
                       fmt='o', capsize=5, capthick=2)
    axes[0, 1].set_title('Cross-Validation Scores')
    axes[0, 1].set_ylabel('CV Accuracy')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Training time comparison
    times = [results_dict[m]['training_time'] for m in models]
    axes[1, 0].bar(models, times, color=sns.color_palette("plasma", len(models)))
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Accuracy vs Time scatter
    axes[1, 1].scatter(times, accuracies, s=100, alpha=0.7)
    for i, model in enumerate(models):
        axes[1, 1].annotate(model, (times[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 1].set_xlabel('Training Time (seconds)')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy vs Training Time')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(results_dict, y_test, target_names):
    """Plot confusion matrices for all models."""
    
    n_models = len(results_dict)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.ravel() if n_models > 1 else [axes]
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        y_pred = results['predictions']
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names,
                   ax=axes[i])
        axes[i].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
```

### Feature Importance Visualization
```python
def plot_feature_importance(model, feature_names, top_k=10):
    """Plot feature importance for tree-based models."""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Get top k features
        indices = np.argsort(importances)[-top_k:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_k), importances[indices])
        plt.yticks(range(top_k), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Feature Importances')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not have feature_importances_ attribute")

def plot_learning_curves(model, X, y, cv=5):
    """Plot learning curves to assess overfitting."""
    
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training Score')
    plt.fill_between(train_sizes, 
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1),
                     alpha=0.1)
    
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation Score')
    plt.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1),
                     alpha=0.1)
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

## 🚨 Common Pitfalls and Solutions

### Data-Related Issues
1. **Class Imbalance**
   - Use stratified sampling
   - Apply class weights
   - Consider SMOTE for oversampling
   - Use appropriate metrics (precision, recall, F1)

2. **Data Leakage**
   - Fit preprocessing only on training data
   - Avoid future information in features
   - Separate validation properly

3. **Feature Scaling**
   - Always scale for SVM, KNN, neural networks
   - StandardScaler for normal distributions
   - MinMaxScaler for bounded features
   - RobustScaler for outliers

### Model Selection Issues
1. **Overfitting**
   - Use cross-validation
   - Implement regularization
   - Reduce model complexity
   - Get more training data

2. **Underfitting**
   - Increase model complexity
   - Add more features
   - Reduce regularization
   - Use ensemble methods

3. **Poor Generalization**
   - Use proper validation strategy
   - Check for data drift
   - Monitor performance over time
   - Implement robust preprocessing

### Pipeline Issues
1. **Memory Problems**
   - Use batch processing
   - Implement data generators
   - Optimize memory usage
   - Use cloud resources

2. **Computation Time**
   - Parallelize when possible
   - Use approximate algorithms
   - Sample data for development
   - Optimize hyperparameter search

## 📈 Performance Optimization Tips

### Scikit-learn Optimization
```python
# Use n_jobs for parallel processing
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Optimize memory usage
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator, param_grid, n_jobs=-1, pre_dispatch='2*n_jobs')

# Use early stopping for gradient boosting
gb = GradientBoostingClassifier(
    n_estimators=1000,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10
)
```

### Metaflow Optimization
```python
# Use appropriate resources
@resources(memory=8000, cpu=4)
@step
def memory_intensive_step(self):
    pass

# Cache expensive computations
@step
def expensive_preprocessing(self):
    if hasattr(self, 'cached_data'):
        self.processed_data = self.cached_data
    else:
        # Expensive processing
        self.processed_data = process_data(self.raw_data)
        self.cached_data = self.processed_data
```

## 🔗 Additional Resources

### Datasets for Practice
- **[Kaggle Datasets](https://www.kaggle.com/datasets)** - Real-world datasets
- **[UCI ML Repository](https://archive.ics.uci.edu/ml/)** - Classic ML datasets
- **[OpenML](https://www.openml.org/)** - Collaborative ML platform
- **[Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets)** - Curated list

### Tools and Libraries
- **[Yellowbrick](https://www.scikit-yb.org/)** - ML visualization
- **[SHAP](https://shap.readthedocs.io/)** - Model interpretability
- **[Optuna](https://optuna.org/)** - Hyperparameter optimization
- **[MLflow](https://mlflow.org/)** - ML lifecycle management

### Communities and Forums
- **[Kaggle Learn](https://www.kaggle.com/learn)** - Free micro-courses
- **[Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)** - Community discussions
- **[Stack Overflow](https://stackoverflow.com/questions/tagged/machine-learning)** - Technical Q&A
- **[Towards Data Science](https://towardsdatascience.com/)** - Medium publication

### Blogs and Tutorials
- **[Scikit-learn Examples](https://scikit-learn.org/stable/auto_examples/)** - Official examples
- **[Machine Learning Mastery](https://machinelearningmastery.com/)** - Jason Brownlee's tutorials
- **[Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)** - Free course

---

**💡 Pro Tip**: Bookmark this guide and refer back to it during exercises. The best way to learn is by doing - try implementing each concept with your own datasets!

**🚀 Next Week Preview**: Week 4 will cover advanced ML techniques including ensemble methods, LangGraph for agent-based workflows, and complex pipeline orchestration.