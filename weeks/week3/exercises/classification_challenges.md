# Week 3: Multi-Class Classification Challenges

Welcome to the classification challenges for Week 3! These exercises will test and extend your supervised learning skills with increasingly complex multi-class classification problems.

## 🎯 Learning Objectives

By completing these challenges, you will:
- Master multi-class classification with imbalanced datasets
- Implement advanced evaluation metrics and techniques
- Build production-ready classification pipelines with Metaflow
- Integrate LLM-powered model interpretation
- Handle real-world classification scenarios

## 📋 Prerequisites

Before starting these challenges, ensure you have completed:
- [ ] Week 3 Workshop (classification fundamentals)
- [ ] Basic Metaflow pipeline creation
- [ ] Understanding of evaluation metrics (accuracy, precision, recall, F1)
- [ ] LangChain setup (optional but recommended)

## 🏆 Challenge Overview

| Challenge | Difficulty | Focus Area | Time Estimate |
|-----------|------------|------------|---------------|
| [Challenge 1](#challenge-1-customer-segmentation) | 🟢 Beginner | Basic Multi-class | 30-45 min |
| [Challenge 2](#challenge-2-medical-diagnosis) | 🟡 Intermediate | Imbalanced Classes | 45-60 min |
| [Challenge 3](#challenge-3-text-classification) | 🟡 Intermediate | Feature Engineering | 60-75 min |
| [Challenge 4](#challenge-4-pipeline-optimization) | 🔴 Advanced | Metaflow Integration | 75-90 min |
| [Challenge 5](#challenge-5-production-deployment) | 🔴 Advanced | End-to-End System | 90-120 min |
| [Bonus Challenge](#bonus-challenge-ensemble-methods) | 🟣 Expert | Advanced Techniques | 120+ min |

---

## Challenge 1: Customer Segmentation
**Difficulty:** 🟢 Beginner | **Time:** 30-45 minutes

### 📝 Problem Statement

Build a customer segmentation model for an e-commerce platform to classify customers into different value segments based on their purchasing behavior.

### 🎯 Objectives

1. Create a synthetic customer dataset with realistic features
2. Implement at least 4 different classification algorithms
3. Compare performance using multiple evaluation metrics
4. Generate business insights from the best model

### 📊 Dataset Requirements

Create a dataset with 1000+ customers and the following features:
- `age`: Customer age (18-80)
- `annual_income`: Annual income in thousands ($20k-$200k)
- `spending_score`: Spending score (1-100)
- `purchase_frequency`: Average purchases per month
- `avg_order_value`: Average order value
- `days_since_last_purchase`: Days since last purchase
- `total_purchases`: Total number of purchases

**Target classes:**
- `0`: Low Value (30% of customers)
- `1`: Medium Value (50% of customers)  
- `2`: High Value (20% of customers)

### 🛠️ Implementation Steps

#### Step 1: Data Generation
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# TODO: Create synthetic customer dataset
# Hint: Use different distributions for each customer segment
def create_customer_dataset(n_samples=1000):
    np.random.seed(42)
    
    # Generate features with realistic correlations
    # Low value customers: lower income, spending, frequency
    # High value customers: higher income, spending, frequency
    
    # Your implementation here
    pass

# Generate dataset
customer_data = create_customer_dataset()
```

#### Step 2: Algorithm Implementation
Implement and compare these algorithms:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- Support Vector Machine

#### Step 3: Evaluation Framework
```python
def comprehensive_evaluation(models, X_test, y_test, target_names):
    """
    Evaluate multiple models with comprehensive metrics.
    
    Returns:
    - Classification reports
    - Confusion matrices
    - ROC curves (if applicable)
    - Feature importance (where available)
    """
    # Your implementation here
    pass
```

#### Step 4: Business Insights
Generate insights answering:
- Which features are most important for customer segmentation?
- What are the key characteristics of each customer segment?
- How can this model be used for marketing campaigns?

### ✅ Success Criteria

- [ ] Dataset generated with realistic customer behavior patterns
- [ ] 4+ algorithms implemented and trained
- [ ] Comprehensive evaluation with multiple metrics
- [ ] Accuracy > 80% on test set
- [ ] Clear business insights documented
- [ ] Code is well-documented and reproducible

### 🎁 Bonus Points

- Implement class weight balancing for better performance
- Create customer segment profiles with statistical summaries
- Visualize decision boundaries (for 2D projections)
- Generate actionable marketing recommendations

---

## Challenge 2: Medical Diagnosis Classification
**Difficulty:** 🟡 Intermediate | **Time:** 45-60 minutes

### 📝 Problem Statement

Develop a medical diagnosis classification system that can predict disease categories based on patient symptoms and lab results, handling class imbalance typical in medical datasets.

### 🎯 Objectives

1. Work with imbalanced medical data (rare diseases)
2. Implement techniques for handling class imbalance
3. Focus on precision and recall over accuracy
4. Provide confidence scores for predictions

### 📊 Dataset Specifications

Create a medical dataset with:
- **Patients:** 2000+ samples
- **Features:** 15-20 medical indicators
- **Classes:** 5 disease categories with natural imbalance:
  - `Healthy`: 40% of patients
  - `Common Disease A`: 25% of patients
  - `Common Disease B`: 20% of patients
  - `Rare Disease C`: 10% of patients
  - `Very Rare Disease D`: 5% of patients

### 🛠️ Implementation Requirements

#### Step 1: Imbalanced Dataset Creation
```python
def create_medical_dataset():
    """
    Create realistic medical dataset with:
    - Correlated symptoms within disease categories
    - Lab values with normal ranges and disease-specific patterns
    - Natural class imbalance reflecting real medical scenarios
    """
    # Features to include:
    # - age, gender, bmi
    # - symptoms: fever, fatigue, pain_level, etc.
    # - lab_results: blood_pressure, heart_rate, glucose, etc.
    
    # Your implementation here
    pass
```

#### Step 2: Class Imbalance Techniques
Implement and compare:

1. **Class Weighting:**
```python
from sklearn.ensemble import RandomForestClassifier

# Balanced class weights
rf_balanced = RandomForestClassifier(class_weight='balanced')
```

2. **SMOTE Oversampling:**
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

3. **Threshold Tuning:**
```python
def optimize_threshold(model, X_val, y_val, target_class):
    """
    Find optimal classification threshold for specific class.
    Focus on F1-score optimization.
    """
    # Your implementation here
    pass
```

#### Step 3: Medical-Specific Evaluation
```python
def medical_classification_report(y_true, y_pred, y_pred_proba, class_names):
    """
    Generate medical-focused evaluation including:
    - Per-class precision, recall, F1
    - Sensitivity and specificity for each disease
    - Confidence-based predictions
    - Cost-sensitive metrics (false negatives are costly!)
    """
    # Your implementation here
    pass
```

#### Step 4: Confidence-Based Predictions
```python
def confident_predictions(model, X_test, confidence_threshold=0.8):
    """
    Return predictions only when model confidence exceeds threshold.
    For low-confidence cases, flag for manual review.
    """
    # Your implementation here
    pass
```

### ✅ Success Criteria

- [ ] Realistic medical dataset with proper class imbalance
- [ ] 3+ imbalance handling techniques implemented
- [ ] F1-score > 0.75 for all classes (including rare diseases)
- [ ] Sensitivity > 0.90 for rare diseases (minimize false negatives)
- [ ] Confidence-based prediction system implemented
- [ ] Medical interpretation of results provided

### 🎁 Bonus Points

- Implement cost-sensitive learning with custom loss functions
- Create LIME explanations for individual predictions
- Build ensemble of imbalance-handling techniques
- Generate doctor-friendly prediction reports

---

## Challenge 3: Text Classification Pipeline
**Difficulty:** 🟡 Intermediate | **Time:** 60-75 minutes

### 📝 Problem Statement

Build an end-to-end text classification system for news article categorization, incorporating feature engineering, model selection, and natural language processing techniques.

### 🎯 Objectives

1. Implement text preprocessing and feature extraction
2. Handle multi-class text classification
3. Compare different vectorization techniques
4. Build interpretable models for text data

### 📊 Dataset Requirements

Create or use a news classification dataset with:
- **Articles:** 3000+ news articles
- **Categories:** 6 news categories
  - Politics (20%)
  - Technology (20%)
  - Sports (15%)
  - Business (15%)
  - Entertainment (15%)
  - Health (15%)
- **Features:** Article headlines and content

### 🛠️ Implementation Steps

#### Step 1: Text Preprocessing Pipeline
```python
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """
        Complete text preprocessing pipeline:
        1. Lowercase conversion
        2. Remove special characters and numbers
        3. Remove stopwords
        4. Stemming/Lemmatization
        5. Handle negations
        """
        # Your implementation here
        pass
    
    def extract_features(self, texts, method='tfidf'):
        """
        Extract features using different methods:
        - TF-IDF
        - Count Vectorizer
        - N-grams
        - Word embeddings (bonus)
        """
        # Your implementation here
        pass
```

#### Step 2: Feature Engineering Comparison
```python
def compare_vectorization_methods(X_train_text, y_train, X_test_text, y_test):
    """
    Compare different text vectorization approaches:
    1. TF-IDF (unigrams)
    2. TF-IDF (unigrams + bigrams)
    3. Count Vectorizer
    4. TF-IDF with custom preprocessing
    """
    vectorizers = {
        'tfidf_unigram': TfidfVectorizer(max_features=5000, stop_words='english'),
        'tfidf_ngram': TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english'),
        'count_vector': CountVectorizer(max_features=5000, stop_words='english'),
        # Add more vectorizers
    }
    
    results = {}
    # Your implementation here
    
    return results
```

#### Step 3: Model Comparison for Text
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def text_classification_comparison(X_train, y_train, X_test, y_test):
    """
    Compare models particularly suited for text classification:
    - Multinomial Naive Bayes (baseline for text)
    - Logistic Regression (good with high-dimensional text)
    - Linear SVM (excellent for text)
    - Random Forest (for comparison)
    """
    models = {
        'naive_bayes': MultinomialNB(),
        'logistic_regression': LogisticRegression(max_iter=1000),
        'linear_svm': SVC(kernel='linear', probability=True),
        'random_forest': RandomForestClassifier(n_estimators=100)
    }
    
    # Your implementation here
    pass
```

#### Step 4: Text Interpretability
```python
def analyze_feature_importance_text(model, vectorizer, class_names, top_n=10):
    """
    Analyze what words/features are most important for each class.
    Works with linear models (Logistic Regression, SVM, Naive Bayes).
    """
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(model, 'coef_'):
        # For linear models
        for i, class_name in enumerate(class_names):
            # Get top positive and negative features for this class
            pass
    
    # Your implementation here
    pass

def generate_prediction_explanation(model, vectorizer, text, prediction):
    """
    Explain why a specific text was classified as a particular category.
    """
    # Your implementation here
    pass
```

### ✅ Success Criteria

- [ ] Complete text preprocessing pipeline implemented
- [ ] 3+ vectorization methods compared
- [ ] 4+ classification algorithms evaluated
- [ ] Accuracy > 85% on test set
- [ ] Feature importance analysis for each class
- [ ] Prediction explanations for sample texts
- [ ] Comprehensive comparison report

### 🎁 Bonus Points

- Implement custom feature extraction (e.g., sentiment scores, readability metrics)
- Use pre-trained word embeddings (Word2Vec, GloVe)
- Build hierarchical classification system
- Create confusion matrix analysis with common misclassifications

---

## Challenge 4: Metaflow Pipeline Optimization
**Difficulty:** 🔴 Advanced | **Time:** 75-90 minutes

### 📝 Problem Statement

Create a production-ready, optimized Metaflow pipeline for multi-class classification that includes parallel processing, hyperparameter tuning, model selection, and automated evaluation.

### 🎯 Objectives

1. Build scalable Metaflow pipeline with parallel execution
2. Implement comprehensive hyperparameter optimization
3. Create automated model selection and validation
4. Include monitoring and logging capabilities

### 🛠️ Pipeline Architecture

```python
from metaflow import FlowSpec, step, Parameter, foreach, resources, catch, retry

class OptimizedClassificationFlow(FlowSpec):
    """
    Production-ready classification pipeline with optimization.
    """
    
    dataset_name = Parameter('dataset_name', 
                            help='Dataset to use: customer, medical, news',
                            default='customer')
    
    test_size = Parameter('test_size',
                         help='Test set proportion',
                         default=0.2)
    
    cv_folds = Parameter('cv_folds',
                        help='Cross-validation folds',
                        default=5)
    
    n_jobs = Parameter('n_jobs',
                      help='Parallel jobs for optimization',
                      default=4)
    
    optimize_hyperparams = Parameter('optimize_hyperparams',
                                    help='Whether to run hyperparameter optimization',
                                    default=True)
```

#### Step 1: Advanced Data Loading and Validation
```python
@step
def start(self):
    """
    Load data with validation and quality checks.
    """
    # Data loading factory pattern
    self.data_loader = self.create_data_loader(self.dataset_name)
    self.X, self.y, self.metadata = self.data_loader.load_and_validate()
    
    # Data quality checks
    self.quality_report = self.validate_data_quality()
    
    # Algorithm configuration based on dataset characteristics
    self.algorithm_configs = self.configure_algorithms()
    
    self.next(self.data_preprocessing)

def create_data_loader(self, dataset_name):
    """Factory method for data loaders."""
    loaders = {
        'customer': CustomerDataLoader(),
        'medical': MedicalDataLoader(), 
        'news': NewsDataLoader()
    }
    return loaders.get(dataset_name, CustomerDataLoader())

def validate_data_quality(self):
    """Comprehensive data quality validation."""
    # Check for missing values, outliers, class distribution
    # Your implementation here
    pass
```

#### Step 2: Parallel Algorithm Training with Resource Optimization
```python
@resources(memory=8000, cpu=4)
@retry(times=3)
@catch(var='training_error')
@step
def train_algorithm(self):
    """
    Train individual algorithms with error handling and resource management.
    """
    self.current_algorithm = self.input
    config = self.algorithm_configs[self.current_algorithm]
    
    # Dynamic resource allocation based on algorithm type
    self.allocate_resources(config)
    
    # Train with progress tracking
    with self.algorithm_timer():
        self.model_results = self.train_with_monitoring(config)
    
    # Model validation
    self.validation_results = self.validate_model()
    
    self.next(self.hyperparameter_optimization)

@resources(memory=16000, cpu=8)
@step
def hyperparameter_optimization(self):
    """
    Advanced hyperparameter optimization with early stopping.
    """
    if not self.optimize_hyperparams:
        self.next(self.model_evaluation)
        return
    
    # Intelligent search space based on dataset characteristics
    search_space = self.create_adaptive_search_space()
    
    # Multi-objective optimization (accuracy + speed + interpretability)
    self.optimization_results = self.multi_objective_optimization(search_space)
    
    self.next(self.model_evaluation)
```

#### Step 3: Comprehensive Model Evaluation
```python
@step
def model_evaluation(self):
    """
    Comprehensive model evaluation with multiple metrics.
    """
    # Performance metrics
    self.performance_metrics = self.calculate_comprehensive_metrics()
    
    # Model interpretability analysis
    self.interpretability_analysis = self.analyze_model_interpretability()
    
    # Robustness testing
    self.robustness_results = self.test_model_robustness()
    
    # Business impact estimation
    self.business_impact = self.estimate_business_impact()
    
    self.next(self.aggregate_results)

def calculate_comprehensive_metrics(self):
    """Calculate extensive evaluation metrics."""
    metrics = {
        'accuracy_based': ['accuracy', 'balanced_accuracy'],
        'precision_recall': ['precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted'],
        'probabilistic': ['roc_auc_ovr', 'roc_auc_ovo', 'log_loss'],
        'custom_business': ['cost_sensitive_accuracy', 'profit_score']
    }
    # Your implementation here
    pass
```

#### Step 4: Automated Model Selection
```python
@step
def aggregate_results(self, inputs):
    """
    Intelligent model selection with multiple criteria.
    """
    # Collect all results
    self.all_results = self.collect_parallel_results(inputs)
    
    # Multi-criteria decision analysis
    self.model_rankings = self.rank_models_multi_criteria()
    
    # Statistical significance testing
    self.significance_tests = self.test_statistical_significance()
    
    # Ensemble recommendations
    self.ensemble_recommendations = self.recommend_ensembles()
    
    self.next(self.generate_production_report)

def rank_models_multi_criteria(self):
    """
    Rank models using multiple criteria with weights.
    """
    criteria = {
        'performance': 0.4,    # Primary metric performance
        'robustness': 0.2,     # Cross-validation stability
        'speed': 0.15,         # Training and inference speed
        'interpretability': 0.15, # Model explainability
        'memory_efficiency': 0.1  # Resource usage
    }
    # Your implementation here
    pass
```

#### Step 5: Production Report Generation
```python
@step
def generate_production_report(self):
    """
    Generate comprehensive production-ready report.
    """
    # Executive summary
    self.executive_summary = self.create_executive_summary()
    
    # Technical specifications
    self.technical_specs = self.document_technical_specifications()
    
    # Deployment recommendations
    self.deployment_guide = self.create_deployment_guide()
    
    # Monitoring setup
    self.monitoring_config = self.setup_monitoring_configuration()
    
    self.next(self.end)
```

### ✅ Success Criteria

- [ ] Complete Metaflow pipeline with parallel processing
- [ ] Advanced hyperparameter optimization implemented
- [ ] Multi-criteria model selection system
- [ ] Comprehensive evaluation framework
- [ ] Production-ready documentation generated
- [ ] Error handling and retry mechanisms
- [ ] Resource optimization for different algorithm types
- [ ] Statistical significance testing for model comparison

### 🎁 Bonus Points

- Implement A/B testing framework for model comparison
- Add automated model versioning and artifact management
- Create custom metrics for business-specific evaluation
- Build pipeline monitoring and alerting system
- Implement progressive model validation

---

## Challenge 5: Production Deployment System
**Difficulty:** 🔴 Advanced | **Time:** 90-120 minutes

### 📝 Problem Statement

Build a complete end-to-end system for deploying and monitoring multi-class classification models in production, including model serving, monitoring, and automated retraining.

### 🎯 Objectives

1. Create model serving infrastructure
2. Implement real-time monitoring and alerting
3. Build automated model validation and deployment
4. Design data drift detection system

### 🛠️ System Architecture

#### Step 1: Model Serving API
```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging

class ModelServer:
    """
    Production model serving with monitoring.
    """
    
    def __init__(self, model_path, preprocessor_path):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.prediction_log = []
        self.performance_metrics = {}
        
    def predict(self, features, return_probabilities=False):
        """
        Make prediction with monitoring.
        """
        # Input validation
        validated_features = self.validate_input(features)
        
        # Preprocessing
        processed_features = self.preprocessor.transform(validated_features)
        
        # Prediction
        prediction = self.model.predict(processed_features)
        probabilities = self.model.predict_proba(processed_features)
        
        # Logging for monitoring
        self.log_prediction(validated_features, prediction, probabilities)
        
        # Response formatting
        response = {
            'prediction': int(prediction[0]),
            'timestamp': datetime.now().isoformat(),
            'model_version': self.get_model_version()
        }
        
        if return_probabilities:
            response['probabilities'] = probabilities[0].tolist()
            response['confidence'] = float(np.max(probabilities[0]))
        
        return response
    
    def validate_input(self, features):
        """Validate input features."""
        # Your implementation here
        pass
    
    def log_prediction(self, features, prediction, probabilities):
        """Log prediction for monitoring."""
        # Your implementation here
        pass
```

#### Step 2: Real-time Monitoring System
```python
class ModelMonitor:
    """
    Real-time model performance monitoring.
    """
    
    def __init__(self, model_server, alert_thresholds):
        self.model_server = model_server
        self.alert_thresholds = alert_thresholds
        self.metrics_history = []
        
    def monitor_performance(self, true_labels=None, window_size=100):
        """
        Monitor model performance in real-time.
        """
        recent_predictions = self.get_recent_predictions(window_size)
        
        # Performance metrics
        if true_labels is not None:
            performance = self.calculate_performance_metrics(
                recent_predictions, true_labels
            )
            self.metrics_history.append(performance)
            
            # Check for performance degradation
            self.check_performance_alerts(performance)
        
        # Data drift detection
        drift_score = self.detect_data_drift(recent_predictions)
        
        # Prediction distribution monitoring
        distribution_stats = self.analyze_prediction_distribution(recent_predictions)
        
        return {
            'performance_metrics': performance if true_labels else None,
            'drift_score': drift_score,
            'distribution_stats': distribution_stats,
            'alerts': self.get_active_alerts()
        }
    
    def detect_data_drift(self, recent_predictions):
        """
        Detect data drift using statistical tests.
        """
        # Compare recent feature distributions with training data
        # Your implementation here
        pass
    
    def check_performance_alerts(self, current_performance):
        """
        Check if performance metrics trigger alerts.
        """
        # Your implementation here
        pass
```

#### Step 3: Automated Model Validation
```python
class ModelValidator:
    """
    Automated model validation for production deployment.
    """
    
    def __init__(self, validation_tests_config):
        self.tests_config = validation_tests_config
        self.validation_results = {}
    
    def validate_model(self, model, test_data, baseline_model=None):
        """
        Run comprehensive model validation tests.
        """
        validation_results = {}
        
        # Performance validation
        validation_results['performance'] = self.validate_performance(
            model, test_data
        )
        
        # Robustness testing
        validation_results['robustness'] = self.test_robustness(
            model, test_data
        )
        
        # Fairness testing
        validation_results['fairness'] = self.test_fairness(
            model, test_data
        )
        
        # Comparison with baseline
        if baseline_model:
            validation_results['baseline_comparison'] = self.compare_with_baseline(
                model, baseline_model, test_data
            )
        
        # Security testing
        validation_results['security'] = self.test_security(model)
        
        # Overall validation score
        validation_results['overall_score'] = self.calculate_validation_score(
            validation_results
        )
        
        return validation_results
    
    def validate_performance(self, model, test_data):
        """Validate model performance against thresholds."""
        # Your implementation here
        pass
    
    def test_robustness(self, model, test_data):
        """Test model robustness with adversarial examples."""
        # Your implementation here
        pass
    
    def test_fairness(self, model, test_data):
        """Test model fairness across different groups."""
        # Your implementation here
        pass
```

#### Step 4: Automated Retraining Pipeline
```python
class AutomatedRetrainingPipeline:
    """
    Automated model retraining and deployment pipeline.
    """
    
    def __init__(self, retrain_config, model_server, validator):
        self.config = retrain_config
        self.model_server = model_server
        self.validator = validator
        
    def should_retrain(self, monitoring_results):
        """
        Determine if model should be retrained based on monitoring.
        """
        triggers = {
            'performance_degradation': monitoring_results['performance_metrics']['accuracy'] < self.config['min_accuracy'],
            'data_drift': monitoring_results['drift_score'] > self.config['max_drift_score'],
            'time_based': self.check_time_trigger(),
            'data_volume': self.check_data_volume_trigger()
        }
        
        # Your implementation here
        return any(triggers.values()), triggers
    
    def retrain_model(self, new_data):
        """
        Retrain model with new data.
        """
        # Data preparation
        X_new, y_new = self.prepare_training_data(new_data)
        
        # Model training with hyperparameter optimization
        new_model = self.train_optimized_model(X_new, y_new)
        
        # Validation
        validation_results = self.validator.validate_model(
            new_model, self.config['validation_data'], self.model_server.model
        )
        
        # Deployment decision
        if validation_results['overall_score'] > self.config['min_validation_score']:
            self.deploy_model(new_model, validation_results)
            return True, validation_results
        else:
            self.log_deployment_failure(validation_results)
            return False, validation_results
    
    def deploy_model(self, new_model, validation_results):
        """
        Deploy validated model to production.
        """
        # Your implementation here
        pass
```

#### Step 5: Dashboard and Alerting
```python
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

class ModelDashboard:
    """
    Real-time model monitoring dashboard.
    """
    
    def __init__(self, model_monitor):
        self.monitor = model_monitor
        
    def create_dashboard(self):
        """
        Create Streamlit dashboard for model monitoring.
        """
        st.title("Production Model Monitoring Dashboard")
        
        # Real-time metrics
        self.display_realtime_metrics()
        
        # Performance trends
        self.display_performance_trends()
        
        # Data drift monitoring
        self.display_drift_analysis()
        
        # Prediction distribution
        self.display_prediction_distribution()
        
        # Alerts and notifications
        self.display_alerts()
        
    def display_realtime_metrics(self):
        """Display real-time performance metrics."""
        # Your implementation here
        pass
    
    def display_performance_trends(self):
        """Display performance trends over time."""
        # Your implementation here
        pass
```

### ✅ Success Criteria

- [ ] Complete model serving API with logging
- [ ] Real-time monitoring system implemented
- [ ] Data drift detection system working
- [ ] Automated model validation pipeline
- [ ] Retraining triggers and automation
- [ ] Production dashboard with alerts
- [ ] Comprehensive testing suite
- [ ] Documentation for deployment and maintenance

### 🎁 Bonus Points

- Implement blue-green deployment strategy
- Add model A/B testing framework
- Create cost-based monitoring (infrastructure costs vs performance)
- Build multi-model serving with load balancing
- Implement model explainability dashboard
- Add automated rollback mechanisms

---

## Bonus Challenge: Advanced Ensemble Methods
**Difficulty:** 🟣 Expert | **Time:** 120+ minutes

### 📝 Problem Statement

Implement advanced ensemble methods including stacking, blending, and dynamic ensemble selection for multi-class classification, with automated ensemble optimization.

### 🎯 Objectives

1. Implement multiple ensemble strategies
2. Create dynamic ensemble selection algorithms
3. Build automated ensemble optimization
4. Compare ensemble methods with individual models

### 🛠️ Advanced Ensemble Techniques

#### 1. Multi-Level Stacking
```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

class MultiLevelStackingClassifier:
    """
    Multi-level stacking with diverse base learners.
    """
    
    def __init__(self, level1_models, level2_models, final_model, cv_folds=5):
        self.level1_models = level1_models
        self.level2_models = level2_models
        self.final_model = final_model
        self.cv_folds = cv_folds
        
    def fit(self, X, y):
        """
        Train multi-level stacking ensemble.
        """
        # Level 1: Base models
        level1_predictions = self._fit_level1(X, y)
        
        # Level 2: Meta models on level 1 predictions
        level2_predictions = self._fit_level2(level1_predictions, y)
        
        # Final level: Combine level 1 and level 2
        final_features = np.concatenate([level1_predictions, level2_predictions], axis=1)
        self.final_model.fit(final_features, y)
        
        return self
    
    def _fit_level1(self, X, y):
        """Fit level 1 base models with cross-validation."""
        # Your implementation here
        pass
    
    def _fit_level2(self, level1_pred, y):
        """Fit level 2 meta models."""
        # Your implementation here
        pass
```

#### 2. Dynamic Ensemble Selection
```python
class DynamicEnsembleSelector:
    """
    Dynamic ensemble selection based on local competence.
    """
    
    def __init__(self, base_models, competence_measure='accuracy', k_neighbors=7):
        self.base_models = base_models
        self.competence_measure = competence_measure
        self.k_neighbors = k_neighbors
        
    def fit(self, X, y):
        """
        Train base models and build competence regions.
        """
        # Train all base models
        for model in self.base_models:
            model.fit(X, y)
        
        # Build competence map
        self.competence_map = self._build_competence_map(X, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using dynamic ensemble selection.
        """
        predictions = []
        
        for x in X:
            # Find local competence
            competent_models = self._select_competent_models(x)
            
            # Make prediction with selected models
            if competent_models:
                ensemble_pred = self._ensemble_predict(competent_models, x.reshape(1, -1))
            else:
                # Fallback to all models
                ensemble_pred = self._ensemble_predict(self.base_models, x.reshape(1, -1))
            
            predictions.append(ensemble_pred)
        
        return np.array(predictions)
    
    def _build_competence_map(self, X, y):
        """Build competence map for each model."""
        # Your implementation here
        pass
    
    def _select_competent_models(self, x):
        """Select competent models for specific instance."""
        # Your implementation here
        pass
```

#### 3. Automated Ensemble Optimization
```python
import optuna

class AutomatedEnsembleOptimizer:
    """
    Automated ensemble optimization using Bayesian optimization.
    """
    
    def __init__(self, base_models, optimization_metric='f1_macro', n_trials=100):
        self.base_models = base_models
        self.optimization_metric = optimization_metric
        self.n_trials = n_trials
        
    def optimize_ensemble(self, X_train, y_train, X_val, y_val):
        """
        Optimize ensemble configuration.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Create optimization study
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=self.n_trials)
        
        # Get best ensemble configuration
        best_config = study.best_params
        self.best_ensemble = self._build_ensemble_from_config(best_config)
        
        return self.best_ensemble, study.best_value
    
    def _objective(self, trial):
        """
        Objective function for ensemble optimization.
        """
        # Suggest ensemble configuration
        config = self._suggest_ensemble_config(trial)
        
        # Build and evaluate ensemble
        ensemble = self._build_ensemble_from_config(config)
        ensemble.fit(self.X_train, self.y_train)
        
        # Evaluate on validation set
        y_pred = ensemble.predict(self.X_val)
        score = self._calculate_metric(self.y_val, y_pred)
        
        return score
    
    def _suggest_ensemble_config(self, trial):
        """Suggest ensemble configuration."""
        # Your implementation here
        pass
```

### ✅ Success Criteria

- [ ] Multi-level stacking implemented and working
- [ ] Dynamic ensemble selection algorithm
- [ ] Automated ensemble optimization
- [ ] Comprehensive comparison with base models
- [ ] Performance improvement over best individual model
- [ ] Computational efficiency analysis
- [ ] Ensemble interpretability methods

### 🎁 Bonus Points

- Implement online ensemble learning
- Create ensemble diversity metrics
- Build GPU-accelerated ensemble training
- Implement ensemble pruning algorithms
- Create ensemble uncertainty quantification

---

## 📊 Submission Guidelines

### Required Deliverables

For each challenge, submit:

1. **Code Implementation**
   - Well-documented Python code
   - Jupyter notebook or Python script
   - Requirements.txt file

2. **Results Report**
   - Performance metrics and comparisons
   - Visualizations and plots
   - Analysis and insights

3. **Documentation**
   - Clear problem description
   - Implementation approach
   - Challenges faced and solutions
   - Future improvements

### Code Quality Standards

- [ ] **PEP 8 compliance** - Follow Python style guide
- [ ] **Documentation** - Docstrings for all functions/classes
- [ ] **Type hints** - Use type annotations where appropriate
- [ ] **Error handling** - Proper exception handling
- [ ] **Testing** - Include basic unit tests
- [ ] **Reproducibility** - Set random seeds, include environment info

### Report Structure

```markdown
# Challenge [X]: [Title]

## Problem Summary
Brief description of the problem and approach

## Implementation Details
- Data preparation steps
- Algorithm choices and rationale
- Hyperparameter tuning approach
- Evaluation methodology

## Results
- Performance metrics table
- Visualizations (confusion matrices, ROC curves, etc.)
- Comparison with baselines

## Analysis
- Key findings and insights
- Model interpretability analysis
- Business implications

## Conclusions
- Summary of results
- Limitations and future work
- Lessons learned
```

## 🎯 Evaluation Criteria

| Criteria | Weight | Description |
|----------|--------|-------------|
| **Correctness** | 30% | Implementation works and produces expected results |
| **Performance** | 25% | Model performance meets or exceeds targets |
| **Code Quality** | 20% | Clean, documented, well-structured code |
| **Analysis** | 15% | Thorough analysis and insights |
| **Innovation** | 10% | Creative solutions and bonus implementations |

## 🚀 Next Steps

After completing these challenges:

1. **Review Solutions** - Compare your approach with provided solutions
2. **Optimize Further** - Experiment with advanced techniques
3. **Apply to Real Data** - Use your own datasets
4. **Prepare for Week 4** - Advanced ML and LangGraph integration

## 💡 Tips for Success

- **Start Simple** - Begin with basic implementations, then add complexity
- **Iterate Quickly** - Test frequently and debug incrementally
- **Document Everything** - Good documentation helps with debugging and sharing
- **Focus on Understanding** - Don't just copy code, understand the concepts
- **Collaborate** - Discuss approaches with peers and mentors

## 📚 Additional Resources

- [Scikit-learn Classification Guide](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [Metaflow Best Practices](https://docs.metaflow.org/scaling/remote-tasks/introduction)
- [Model Evaluation Strategies](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Ensemble Methods Guide](https://scikit-learn.org/stable/modules/ensemble.html)

---

**Good luck with your classification challenges! Remember: the goal is not just to complete the exercises, but to deeply understand the concepts and be able to apply them to real-world problems. 🚀**