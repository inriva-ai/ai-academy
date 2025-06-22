# Week 3: Supervised Learning with Metaflow Pipelines - Complete Materials

This directory contains all materials for Week 3, focusing on **supervised learning fundamentals**, **Metaflow ML pipelines**, and **LangChain model interpretation**.

## 📁 Complete File Structure

```
weeks/week3/
├── README.md                           # This overview file
├── notebooks/
│   ├── week3_workshop.ipynb           # Interactive workshop notebook
│   ├── classification_deep_dive.ipynb  # Comprehensive classification tutorial
│   ├── regression_fundamentals.ipynb   # Complete regression analysis
│   └── model_comparison_lab.ipynb     # Side-by-side algorithm comparison
├── flows/
│   ├── supervised_learning_flow.py    # Complete ML pipeline with parallel training
│   ├── model_comparison_flow.py       # Advanced model comparison with hyperparameter tuning
│   └── hybrid_evaluation_flow.py      # ML + LLM interpretation system
├── data/
│   ├── wine_quality.csv              # Wine quality dataset info
│   ├── housing_boston.csv            # Boston housing data info
│   ├── customer_churn.csv            # Customer churn prediction data
│   └── credit_risk.csv               # Credit risk assessment data
├── exercises/
│   ├── classification_challenges.md   # Multi-class classification problems
│   ├── regression_practice.md        # Regression techniques and evaluation
│   ├── pipeline_optimization.md      # Metaflow optimization exercises
│   └── hybrid_evaluation.md          # ML + LLM integration challenges
├── solutions/
│   ├── completed_workshop.ipynb      # Full workshop solutions
│   ├── exercise_solutions.py         # Complete exercise implementations
│   └── advanced_examples.py          # Production-ready examples
└── resources/
    ├── sklearn_algorithms_guide.md    # Comprehensive algorithm reference
    ├── metaflow_ml_patterns.md       # ML-specific Metaflow patterns
    ├── evaluation_metrics_guide.md   # Complete metrics and interpretation guide
    └── hyperparameter_tuning_guide.md # Advanced tuning strategies
```

## 🎯 Week 3 Objectives

By completing this week, you will:

### Core ML Skills
- ✅ **Implement multiple supervised learning algorithms** (classification & regression)
- ✅ **Build scalable ML pipelines** using Metaflow with parallel execution
- ✅ **Compare and evaluate models** using comprehensive metrics
- ✅ **Perform hyperparameter tuning** with cross-validation
- ✅ **Master feature engineering** and selection techniques

### MLOps Skills
- ✅ **Design production-ready pipelines** with error handling and resource management
- ✅ **Implement parallel processing** with Metaflow `@foreach` decorator
- ✅ **Manage computational resources** efficiently
- ✅ **Version and track experiments** automatically

### LangChain Integration
- ✅ **Integrate LLM-powered model interpretation** using LangChain
- ✅ **Create hybrid evaluation systems** combining quantitative and qualitative analysis
- ✅ **Generate natural language reports** for business stakeholders
- ✅ **Build fallback systems** for robust LLM integration

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Ensure your environment from Week 2 is activated
conda activate aiml-academy

# Install additional packages if needed
pip install xgboost optuna yellowbrick

# Verify Ollama is running (for LangChain integration)
ollama list
ollama pull llama3.2  # If not already available
```

### 2. Start with the Workshop
```bash
cd weeks/week3/notebooks
jupyter notebook week3_workshop.ipynb
```

### 3. Run Metaflow Pipelines
```bash
cd weeks/week3/flows

# Basic supervised learning pipeline
python supervised_learning_flow.py run

# Advanced model comparison with hyperparameter tuning
python model_comparison_flow.py run --tuning_method grid --n_jobs 4

# Hybrid ML + LLM evaluation
python hybrid_evaluation_flow.py run --use_llm True
```

### 4. Practice with Exercises
```bash
# Work through exercises in order
cd weeks/week3/exercises
# 1. classification_challenges.md
# 2. regression_practice.md  
# 3. pipeline_optimization.md
# 4. hybrid_evaluation.md
```

## 📊 Key Datasets

### Primary Datasets
1. **Wine Classification** (sklearn.datasets.load_wine)
   - 178 samples, 13 features, 3 classes
   - Perfect for multi-class classification
   - Clean data, minimal preprocessing needed

2. **Housing Regression** (synthetic dataset)
   - 1000 samples, 10 features, continuous target
   - Realistic housing price prediction
   - Good for regression techniques

3. **Breast Cancer** (sklearn.datasets.load_breast_cancer)
   - 569 samples, 30 features, 2 classes
   - Binary classification benchmark
   - Medical domain application

### Extended Datasets (for advanced exercises)
- Customer churn prediction
- Credit risk assessment
- Titanic survival prediction
- Boston housing (alternative regression)

## 🤖 Algorithm Coverage

### Classification Algorithms
- **Linear Models**: Logistic Regression
- **Tree-Based**: Decision Trees, Random Forest, Gradient Boosting
- **Support Vector Machines**: RBF and Polynomial kernels
- **Probabilistic**: Naive Bayes
- **Instance-Based**: K-Nearest Neighbors
- **Ensemble Methods**: Voting Classifiers, Bagging

### Regression Algorithms
- **Linear Models**: Linear Regression, Ridge, Lasso
- **Tree-Based**: Random Forest Regressor, Gradient Boosting Regressor
- **Support Vector Regression**: RBF and Linear kernels
- **Advanced**: Elastic Net, Polynomial Regression

### Evaluation Techniques
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- **Regression**: MSE, RMSE, MAE, R², Explained Variance
- **Cross-Validation**: K-Fold, Stratified K-Fold, Time Series Split
- **Hyperparameter Tuning**: Grid Search, Random Search

## 🌊 Metaflow Pipeline Patterns

### 1. Basic ML Pipeline
```python
@step
def start(self): # Data loading and validation
@step  
def preprocess(self): # Feature engineering and scaling
@step
def train_model(self): # Model training
@step
def evaluate(self): # Model evaluation
@step
def end(self): # Results summary
```

### 2. Parallel Model Training
```python
@step
def start(self): # Define algorithms list
@step
def preprocess(self): # Data preparation
@foreach('algorithms')
@resources(memory=4000, cpu=2)
@step
def train_model(self): # Train each algorithm in parallel
@step
def compare_models(self, inputs): # Aggregate results
@step
def end(self): # Final model selection
```

### 3. Hyperparameter Tuning Pipeline
```python
@step
def start(self): # Generate parameter combinations
@foreach('param_combinations')  
@resources(memory=8000, cpu=4)
@step
def tune_hyperparameters(self): # Grid/random search
@step
def select_best_model(self, inputs): # Best parameter selection
@step
def final_evaluation(self): # Test set evaluation
```

## 🦜 LangChain Integration Patterns

### Model Interpretation Chain
```python
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# Create interpretation chain
interpretation_chain = (
    PromptTemplate.from_template("""
    Analyze this ML model: {model_name}
    Accuracy: {accuracy}
    Key Features: {features}
    
    Provide business insights on performance and recommendations.
    """) 
    | Ollama(model="llama3.2")
    | StrOutputParser()
)
```

### Hybrid Evaluation System
```python
@step
def llm_interpretation(self):
    """Generate LLM-powered model explanations."""
    if LANGCHAIN_AVAILABLE:
        for model_name, results in self.model_results.items():
            explanation = self.generate_llm_explanation(model_name, results)
            self.interpretations[model_name] = explanation
    else:
        self.interpretations = self.generate_fallback_explanations()
```

## 📈 Performance Benchmarks

### Expected Results (Wine Dataset)
- **Random Forest**: ~97-99% accuracy
- **Gradient Boosting**: ~95-98% accuracy  
- **SVM (RBF)**: ~95-98% accuracy
- **Logistic Regression**: ~92-96% accuracy
- **Naive Bayes**: ~90-94% accuracy

### Pipeline Performance
- **Sequential Training**: ~30-60 seconds for 5 algorithms
- **Parallel Training**: ~10-20 seconds for 5 algorithms (4 cores)
- **Hyperparameter Tuning**: ~2-10 minutes depending on grid size
- **LLM Interpretation**: ~10-30 seconds per model (Ollama local)

## 🛠 Troubleshooting Guide

### Common Issues and Solutions

#### Metaflow Issues
```bash
# Pipeline fails to start
metaflow configure  # Reconfigure Metaflow
python flow.py show  # Validate flow structure

# Resource allocation errors
@resources(memory=2000, cpu=1)  # Reduce resource requirements
export METAFLOW_DEFAULT_MEMORY=4000  # Set environment defaults

# Parallel execution issues  
# Check foreach parameter is properly defined
self.algorithm_names = list(self.algorithms.keys())
```

#### LangChain/Ollama Issues
```bash
# Ollama not responding
ollama serve  # Start Ollama service
ollama list   # Check available models
ollama pull llama3.2  # Download model if missing

# LangChain import errors
pip install --upgrade langchain langchain-community
pip install ollama  # Ensure Ollama Python package installed

# Fallback when LLM unavailable
LANGCHAIN_AVAILABLE = False  # Use rule-based interpretation
```

#### Memory and Performance Issues
```python
# Reduce dataset size for development
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, random_state=42)

# Use fewer algorithms for testing
algorithms = {'rf': RandomForestClassifier(n_estimators=10)}

# Reduce hyperparameter grid
param_grid = {'n_estimators': [50, 100]}  # Instead of [50, 100, 200, 500]
```

#### Data-Related Issues
```python
# Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Address class imbalance
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight='balanced')

# Scale features properly
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 📚 Learning Path Recommendations

### Beginner Path
1. Start with `week3_workshop.ipynb` - complete all sections
2. Run `supervised_learning_flow.py` with default parameters
3. Complete `classification_challenges.md` exercises
4. Experiment with different algorithms and datasets

### Intermediate Path
1. Complete all workshop materials
2. Run all three Metaflow pipelines with different configurations
3. Complete all exercise sets
4. Implement custom evaluation metrics and visualizations

### Advanced Path
1. Master all materials and exercises
2. Create custom Metaflow decorators and components
3. Implement advanced ensemble methods
4. Design domain-specific evaluation frameworks
5. Integrate with external model registries and monitoring systems

## 🔄 Week 3 → Week 4 Transition

### Skills You've Developed
✅ **Supervised Learning Mastery**: Classification and regression algorithms  
✅ **MLOps Foundations**: Scalable pipeline design with Metaflow  
✅ **Model Evaluation**: Comprehensive metrics and cross-validation  
✅ **LangChain Integration**: Hybrid ML + LLM systems  
✅ **Production Patterns**: Error handling, resource management, parallel processing  

### Preparing for Week 4: Advanced ML & LangGraph
🎯 **Ensemble Methods**: Stacking, voting, and advanced ensemble techniques  
🎯 **LangGraph Introduction**: Agent-based workflows and complex orchestration  
🎯 **Advanced Pipeline Patterns**: Multi-stage workflows and conditional execution  
🎯 **Model Monitoring**: Performance tracking and drift detection  
🎯 **Production Deployment**: Container-based deployment and scaling strategies  

### Recommended Preparation
- Review ensemble methods in scikit-learn documentation
- Explore LangGraph tutorials and examples
- Practice with more complex datasets and business problems
- Experiment with model interpretability tools (SHAP, LIME)

## 🏆 Capstone Project Ideas

Use Week 3 skills to build portfolio projects:

### Beginner Projects
1. **Wine Quality Prediction System**: End-to-end classification pipeline with web interface
2. **House Price Estimator**: Regression model with feature importance analysis
3. **Customer Segmentation**: Classification-based customer analysis with business insights

### Advanced Projects
1. **Automated ML Pipeline**: Self-optimizing model selection and hyperparameter tuning
2. **Hybrid AI Advisor**: ML predictions with LLM-powered explanations and recommendations  
3. **Model Monitoring Dashboard**: Real-time performance tracking with alerting system

## 💡 Best Practices Summary

### Code Organization
- Use clear, descriptive variable names and function documentation
- Implement proper error handling and logging
- Follow scikit-learn conventions and patterns
- Structure Metaflow pipelines with single-responsibility steps

### Model Development
- Always use stratified sampling for classification
- Implement proper cross-validation strategies
- Scale features appropriately for different algorithms
- Save and version trained models and preprocessing pipelines

### Pipeline Design
- Use parallel processing for independent operations
- Implement graceful error handling and recovery
- Cache expensive computations and intermediate results
- Design for reproducibility with random state management

### LLM Integration
- Always implement fallback mechanisms for LLM failures
- Use structured prompts with clear instructions
- Cache LLM responses to avoid redundant API calls
- Validate LLM outputs before using in production systems

## 📞 Getting Help

### Resources for Support
1. **Course Materials**: Comprehensive resources in `/resources/` directory
2. **Solution Examples**: Complete implementations in `/solutions/` directory
3. **Community Forums**: Stack Overflow, Reddit r/MachineLearning
4. **Documentation**: Scikit-learn, Metaflow, and LangChain official docs

### Common Questions and Answers

**Q: Which algorithm should I use for my specific problem?**
A: Start with Random Forest for baseline performance, then try Gradient Boosting for potentially better results. Use the model comparison pipeline to evaluate multiple algorithms systematically.

**Q: How do I handle imbalanced datasets?**
A: Use stratified sampling, class weights (`class_weight='balanced'`), and appropriate metrics (precision, recall, F1) instead of just accuracy.

**Q: My Metaflow pipeline is running slowly. How can I optimize it?**
A: Use parallel processing with `@foreach`, allocate appropriate resources with `@resources`, and consider using `@batch` for cloud scaling.

**Q: LangChain integration isn't working. What should I do?**
A: Check if Ollama is running (`ollama serve`), verify model availability (`ollama list`), and ensure fallback mechanisms are implemented.

---

**🎉 Congratulations on completing Week 3!** You've mastered supervised learning fundamentals and built production-ready ML pipelines. You're now ready to tackle advanced ML techniques and agent-based workflows in Week 4.

**🚀 Next: Week 4 - Advanced ML and LangGraph Agent Systems**