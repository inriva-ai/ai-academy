# Week 4: Advanced ML & LangGraph Workshop
# ==========================================
# Ensemble Methods and Agent-Based ML Systems

# %% [markdown]
# # Week 4: Advanced ML & LangGraph - Building Intelligent ML Systems
# 
# Welcome to Week 4! Today we'll master **ensemble methods** and introduce **LangGraph** 
# for building agent-based ML systems. This marks our transition from traditional ML 
# to hybrid AI systems.
# 
# ## 🎯 Workshop Objectives
# 
# 1. **Master ensemble methods** (Random Forest, Gradient Boosting, Stacking)
# 2. **Understand ensemble theory** and when to use different approaches  
# 3. **Build your first LangGraph agent** for ML tasks
# 4. **Create multi-agent systems** for complex workflows
# 5. **Integrate ensemble ML with agent-based interpretation**

# %% [markdown]
# ## 📚 Part 1: Advanced Ensemble Methods (60 minutes)
# 
# Let's start by exploring different ensemble strategies and understanding 
# when to use each approach.

# %% 
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensemble methods
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Visualization
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")

# %% [markdown]
# ### 1.1 Understanding Ensemble Methods
# 
# Ensemble methods combine predictions from multiple models to create a stronger predictor.
# There are three main strategies:
# 
# 1. **Bagging**: Train models in parallel on different data subsets (reduces variance)
# 2. **Boosting**: Train models sequentially, each correcting previous errors (reduces bias)
# 3. **Stacking**: Use a meta-model to combine predictions from base models

# %%
# Load and prepare data
print("📊 Loading Wine Dataset for Multi-class Classification")
wine = load_wine()
X, y = wine.data, wine.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {np.unique(y)}")

# %% [markdown]
# ### 1.2 Comparing Individual Models vs Ensembles

# %%
# Define base models
base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Naive Bayes': GaussianNB()
}

# Train and evaluate base models
print("🔬 Training Individual Models:")
print("-" * 50)

base_results = {}
for name, model in base_models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    base_results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"{name:20} | Test Acc: {accuracy:.3f} | CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# %% [markdown]
# ### 1.3 Ensemble Method 1: Voting Classifier

# %%
print("\n🗳️ Ensemble Method 1: Voting Classifier")
print("=" * 50)

# Soft voting (uses probability estimates)
voting_soft = VotingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    voting='soft'
)

# Hard voting (uses class predictions)
voting_hard = VotingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    voting='hard'
)

# Train and evaluate
for voting_type, voting_clf in [('Soft Voting', voting_soft), ('Hard Voting', voting_hard)]:
    voting_clf.fit(X_train_scaled, y_train)
    y_pred = voting_clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(voting_clf, X_train_scaled, y_train, cv=5)
    
    print(f"{voting_type:20} | Test Acc: {accuracy:.3f} | CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# %% [markdown]
# ### 1.4 Ensemble Method 2: Bagging (Random Forest)

# %%
print("\n🌲 Ensemble Method 2: Bagging & Random Forest")
print("=" * 50)

# Random Forest (bagging with feature randomness)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred_rf = rf_clf.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_cv_scores = cross_val_score(rf_clf, X_train_scaled, y_train, cv=5)

print(f"Random Forest        | Test Acc: {rf_accuracy:.3f} | CV: {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': wine.feature_names,
    'importance': rf_clf.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Most Important Features (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.5 Ensemble Method 3: Boosting (Gradient Boosting)

# %%
print("\n🚀 Ensemble Method 3: Boosting Methods")
print("=" * 50)

# AdaBoost
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_clf.fit(X_train_scaled, y_train)
ada_accuracy = accuracy_score(y_test, ada_clf.predict(X_test_scaled))

# Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_clf.fit(X_train_scaled, y_train)
gb_accuracy = accuracy_score(y_test, gb_clf.predict(X_test_scaled))

print(f"AdaBoost             | Test Acc: {ada_accuracy:.3f}")
print(f"Gradient Boosting    | Test Acc: {gb_accuracy:.3f}")

# Visualize boosting progression
n_estimators_range = [10, 20, 50, 100, 150, 200]
train_scores = []
test_scores = []

for n_est in n_estimators_range:
    gb = GradientBoostingClassifier(n_estimators=n_est, random_state=42)
    gb.fit(X_train_scaled, y_train)
    train_scores.append(gb.score(X_train_scaled, y_train))
    test_scores.append(gb.score(X_test_scaled, y_test))

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'o-', label='Training Score')
plt.plot(n_estimators_range, test_scores, 'o-', label='Test Score')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Gradient Boosting Performance vs Number of Estimators')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ### 1.6 Ensemble Method 4: Stacking

# %%
print("\n📚 Ensemble Method 4: Stacking Classifier")
print("=" * 50)

# Define base models for stacking
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svc', SVC(kernel='rbf', probability=True, random_state=42)),
    ('nb', GaussianNB())
]

# Meta-learner
meta_learner = LogisticRegression(max_iter=1000)

# Create stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5  # Use 5-fold CV to train meta-learner
)

# Train and evaluate
stacking_clf.fit(X_train_scaled, y_train)
stacking_pred = stacking_clf.predict(X_test_scaled)
stacking_accuracy = accuracy_score(y_test, stacking_pred)

print(f"Stacking Classifier  | Test Acc: {stacking_accuracy:.3f}")

# Compare all ensemble methods
ensemble_results = {
    'Soft Voting': accuracy_score(y_test, voting_soft.predict(X_test_scaled)),
    'Random Forest': rf_accuracy,
    'Gradient Boosting': gb_accuracy,
    'Stacking': stacking_accuracy
}

# Visualize comparison
plt.figure(figsize=(10, 6))
methods = list(ensemble_results.keys())
scores = list(ensemble_results.values())
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']

bars = plt.bar(methods, scores, color=colors)
plt.ylabel('Test Accuracy')
plt.title('Comparison of Ensemble Methods')
plt.ylim(0.9, 1.0)

# Add value labels on bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
             f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 🤖 Part 2: Introduction to LangGraph (30 minutes)
# 
# Now let's introduce LangGraph - a framework for building stateful, 
# multi-agent applications with LLMs.

# %%
print("\n🤖 Part 2: LangGraph for Agent-Based ML Systems")
print("=" * 50)

# First, let's check if LangGraph is available
try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Annotated, Sequence
    import operator
    print("✅ LangGraph imported successfully!")
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("⚠️ LangGraph not installed. Install with: pip install langgraph")
    print("   We'll demonstrate the concepts with mock code.")
    LANGGRAPH_AVAILABLE = False

# %% [markdown]
# ### 2.1 LangGraph Concepts
# 
# LangGraph allows us to build agents that can:
# - Maintain state across multiple steps
# - Make decisions based on conditions
# - Coordinate multiple agents
# - Integrate with external tools (like ML models)

# %%
if LANGGRAPH_AVAILABLE:
    # Define the state for our ML agent
    class MLAgentState(TypedDict):
        """State for our ML pipeline agent."""
        dataset: str
        models_to_train: Sequence[str]
        trained_models: dict
        best_model: str
        evaluation_results: dict
        messages: Annotated[Sequence[str], operator.add]
    
    print("✅ State schema defined for ML Agent")
else:
    print("📝 Example state schema for ML Agent:")
    print("""
    class MLAgentState:
        dataset: str                 # Current dataset name
        models_to_train: List[str]   # Models to train
        trained_models: dict         # Trained model objects
        best_model: str             # Best performing model
        evaluation_results: dict     # Performance metrics
        messages: List[str]         # Agent communication log
    """)

# %% [markdown]
# ### 2.2 Building a Simple ML Pipeline Agent

# %%
# Create mock functions for agent nodes
def data_loader_node(state):
    """Load and prepare dataset."""
    print(f"📊 Loading dataset: {state.get('dataset', 'wine')}")
    state['messages'].append(f"Dataset loaded: {state.get('dataset', 'wine')}")
    return state

def model_trainer_node(state):
    """Train multiple models."""
    models = state.get('models_to_train', ['rf', 'gb'])
    print(f"🏋️ Training models: {models}")
    
    # Mock training results
    state['trained_models'] = {model: f"trained_{model}" for model in models}
    state['messages'].append(f"Trained {len(models)} models")
    return state

def evaluator_node(state):
    """Evaluate trained models."""
    print("📈 Evaluating models...")
    
    # Mock evaluation results
    state['evaluation_results'] = {
        'rf': 0.95,
        'gb': 0.93,
        'svm': 0.91
    }
    
    # Find best model
    best = max(state['evaluation_results'].items(), key=lambda x: x[1])
    state['best_model'] = best[0]
    state['messages'].append(f"Best model: {best[0]} with accuracy {best[1]}")
    return state

def report_generator_node(state):
    """Generate final report."""
    print("📝 Generating report...")
    state['messages'].append("Final report generated")
    return state

print("✅ Agent nodes defined")

# %% [markdown]
# ### 2.3 Creating the Agent Graph

# %%
if LANGGRAPH_AVAILABLE:
    # Build the graph
    workflow = StateGraph(MLAgentState)
    
    # Add nodes
    workflow.add_node("load_data", data_loader_node)
    workflow.add_node("train_models", model_trainer_node)
    workflow.add_node("evaluate", evaluator_node)
    workflow.add_node("report", report_generator_node)
    
    # Define edges
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "train_models")
    workflow.add_edge("train_models", "evaluate")
    workflow.add_edge("evaluate", "report")
    workflow.add_edge("report", END)
    
    # Compile the graph
    app = workflow.compile()
    
    print("✅ ML Pipeline Agent compiled successfully!")
    
    # Visualize the graph structure (if graphviz available)
    try:
        from IPython.display import Image, display
        display(Image(app.get_graph().draw_png()))
    except Exception as e:
        print(f"📊 Graph visualization not available: {e}")
else:
    print("📝 Example agent workflow:")
    print("""
    [Start] → [Load Data] → [Train Models] → [Evaluate] → [Report] → [End]
    """)

# %% [markdown]
# ## 🔗 Part 3: Building Your First Agent (30 minutes)
# 
# Let's build a practical agent that helps with model selection and explanation.

# %%
print("\n🔗 Part 3: Model Selection Agent")
print("=" * 50)

class ModelSelectionAgent:
    """Agent for intelligent model selection and explanation."""
    
    def __init__(self, models_dict):
        self.models = models_dict
        self.results = {}
        
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train all models and evaluate performance."""
        print("🏋️ Training models...")
        
        for name, model in self.models.items():
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            self.results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'overfit_score': train_score - test_score
            }
            
            print(f"  ✓ {name}: Test={test_score:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
    
    def select_best_model(self):
        """Select best model based on multiple criteria."""
        print("\n🎯 Selecting best model...")
        
        # Score each model
        for name, metrics in self.results.items():
            # Composite score: high test score, low overfit, stable CV
            score = (
                metrics['test_score'] * 0.5 +  # Test performance
                (1 - metrics['overfit_score']) * 0.3 +  # Generalization
                (1 - metrics['cv_std']) * 0.2  # Stability
            )
            metrics['composite_score'] = score
        
        # Find best
        best_model = max(self.results.items(), 
                        key=lambda x: x[1]['composite_score'])
        
        return best_model[0], best_model[1]
    
    def generate_explanation(self, model_name, metrics):
        """Generate natural language explanation for model selection."""
        explanation = f"""
📊 Model Selection Report
========================

Selected Model: {model_name}

Performance Metrics:
- Test Accuracy: {metrics['test_score']:.3f}
- Cross-Validation: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}
- Train-Test Gap: {metrics['overfit_score']:.3f}

Reasoning:
"""
        
        # Add specific reasoning
        if metrics['overfit_score'] < 0.05:
            explanation += "✓ Excellent generalization (minimal overfitting)\n"
        elif metrics['overfit_score'] < 0.1:
            explanation += "✓ Good generalization (low overfitting)\n"
        else:
            explanation += "⚠ Some overfitting detected\n"
            
        if metrics['cv_std'] < 0.02:
            explanation += "✓ Very stable across different data splits\n"
        elif metrics['cv_std'] < 0.05:
            explanation += "✓ Reasonably stable performance\n"
        else:
            explanation += "⚠ High variance across data splits\n"
            
        if metrics['test_score'] > 0.95:
            explanation += "✓ Outstanding test performance\n"
        elif metrics['test_score'] > 0.90:
            explanation += "✓ Strong test performance\n"
            
        return explanation

# Create and run the agent
agent_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Stacking': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
        ],
        final_estimator=LogisticRegression()
    )
}

agent = ModelSelectionAgent(agent_models)
agent.train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test)

# Get best model and explanation
best_name, best_metrics = agent.select_best_model()
explanation = agent.generate_explanation(best_name, best_metrics)
print(explanation)

# %% [markdown]
# ## 🌐 Part 4: Multi-Agent ML System (30 minutes)
# 
# Let's create a more complex system with multiple specialized agents 
# working together.

# %%
print("\n🌐 Part 4: Multi-Agent ML System")
print("=" * 50)

class DataAnalystAgent:
    """Agent responsible for data analysis and preprocessing."""
    
    def analyze_dataset(self, X, y, feature_names):
        """Analyze dataset characteristics."""
        analysis = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'class_balance': dict(zip(*np.unique(y, return_counts=True))),
            'feature_stats': {}
        }
        
        # Feature statistics
        for i, name in enumerate(feature_names):
            analysis['feature_stats'][name] = {
                'mean': X[:, i].mean(),
                'std': X[:, i].std(),
                'min': X[:, i].min(),
                'max': X[:, i].max()
            }
        
        return analysis
    
    def recommend_preprocessing(self, analysis):
        """Recommend preprocessing steps based on analysis."""
        recommendations = []
        
        # Check class imbalance
        class_counts = list(analysis['class_balance'].values())
        if max(class_counts) / min(class_counts) > 2:
            recommendations.append("Consider class balancing (SMOTE or class weights)")
        
        # Check feature scales
        feature_ranges = []
        for feat_stats in analysis['feature_stats'].values():
            feature_ranges.append(feat_stats['max'] - feat_stats['min'])
        
        if max(feature_ranges) / min(feature_ranges) > 10:
            recommendations.append("Feature scaling recommended (StandardScaler)")
        
        return recommendations

class ModelTrainerAgent:
    """Agent responsible for model training and optimization."""
    
    def __init__(self):
        self.trained_models = {}
    
    def train_model(self, model_name, model, X_train, y_train, optimize=False):
        """Train a single model with optional hyperparameter optimization."""
        print(f"  Training {model_name}...")
        
        if optimize and model_name == 'Random Forest':
            # Example hyperparameter optimization
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                model, param_grid, cv=3, n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            self.trained_models[model_name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
        else:
            model.fit(X_train, y_train)
            self.trained_models[model_name] = {
                'model': model,
                'best_params': None,
                'best_score': None
            }
        
        return self.trained_models[model_name]

class EvaluatorAgent:
    """Agent responsible for model evaluation and comparison."""
    
    def evaluate_models(self, models_dict, X_test, y_test):
        """Evaluate all models and rank them."""
        evaluation_results = {}
        
        for name, model_info in models_dict.items():
            model = model_info['model']
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'model_info': model_info
            }
        
        # Rank models
        ranked = sorted(evaluation_results.items(), 
                       key=lambda x: x[1]['accuracy'], 
                       reverse=True)
        
        return evaluation_results, ranked

# Create multi-agent system
print("🤝 Creating Multi-Agent System...")

# Initialize agents
data_agent = DataAnalystAgent()
trainer_agent = ModelTrainerAgent()
evaluator_agent = EvaluatorAgent()

# Step 1: Data Analysis
print("\n1️⃣ Data Analyst Agent analyzing dataset...")
data_analysis = data_agent.analyze_dataset(X, y, wine.feature_names)
recommendations = data_agent.recommend_preprocessing(data_analysis)

print(f"   Dataset: {data_analysis['n_samples']} samples, {data_analysis['n_features']} features")
print(f"   Classes: {data_analysis['n_classes']} ({list(data_analysis['class_balance'].values())} samples each)")
print(f"   Recommendations: {recommendations}")

# Step 2: Model Training
print("\n2️⃣ Model Trainer Agent training models...")
models_to_train = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

for name, model in models_to_train.items():
    trainer_agent.train_model(name, model, X_train_scaled, y_train, 
                             optimize=(name == 'Random Forest'))

# Step 3: Model Evaluation
print("\n3️⃣ Evaluator Agent assessing models...")
eval_results, rankings = evaluator_agent.evaluate_models(
    trainer_agent.trained_models, X_test_scaled, y_test
)

print("\n📊 Model Rankings:")
for i, (name, metrics) in enumerate(rankings):
    print(f"   {i+1}. {name}: {metrics['accuracy']:.3f} accuracy")

# %% [markdown]
# ## 🎯 Putting It All Together: Hybrid ML + Agent System

# %%
print("\n🎯 Final Integration: Ensemble ML with Agent Orchestration")
print("=" * 60)

class MLPipelineOrchestrator:
    """Main orchestrator that coordinates all agents."""
    
    def __init__(self):
        self.data_agent = DataAnalystAgent()
        self.trainer_agent = ModelTrainerAgent()
        self.evaluator_agent = EvaluatorAgent()
        self.ensemble_models = {}
        
    def run_pipeline(self, X, y, feature_names, test_size=0.2):
        """Run complete ML pipeline with agent coordination."""
        print("🚀 Starting ML Pipeline Orchestration...\n")
        
        # Phase 1: Data Analysis
        print("Phase 1: Data Analysis")
        print("-" * 30)
        analysis = self.data_agent.analyze_dataset(X, y, feature_names)
        recommendations = self.data_agent.recommend_preprocessing(analysis)
        
        print(f"✓ Dataset analyzed: {analysis['n_samples']} samples")
        print(f"✓ Preprocessing recommendations: {len(recommendations)}")
        
        # Phase 2: Data Preparation
        print("\nPhase 2: Data Preparation")
        print("-" * 30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("✓ Data split and scaled")
        
        # Phase 3: Model Training
        print("\nPhase 3: Ensemble Model Training")
        print("-" * 30)
        
        # Train diverse models for ensemble
        base_models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'lr': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        # Create different ensemble strategies
        self.ensemble_models = {
            'Voting (Soft)': VotingClassifier(
                estimators=list(base_models.items()),
                voting='soft'
            ),
            'Stacking': StackingClassifier(
                estimators=list(base_models.items())[:3],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5
            ),
            'Random Forest (Bagging)': RandomForestClassifier(
                n_estimators=200, 
                max_features='sqrt',
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        
        # Train all ensemble models
        for name, model in self.ensemble_models.items():
            self.trainer_agent.train_model(
                name, model, X_train_scaled, y_train
            )
        
        # Phase 4: Evaluation
        print("\nPhase 4: Model Evaluation")
        print("-" * 30)
        eval_results, rankings = self.evaluator_agent.evaluate_models(
            self.trainer_agent.trained_models, 
            X_test_scaled, 
            y_test
        )
        
        # Phase 5: Generate Report
        print("\nPhase 5: Final Report")
        print("-" * 30)
        best_model_name = rankings[0][0]
        best_model_metrics = rankings[0][1]
        
        report = f"""
📊 ML PIPELINE EXECUTION REPORT
================================

Dataset Summary:
- Samples: {analysis['n_samples']}
- Features: {analysis['n_features']}  
- Classes: {analysis['n_classes']}

Best Performing Model: {best_model_name}
- Accuracy: {best_model_metrics['accuracy']:.3f}
- Precision: {best_model_metrics['precision']:.3f}
- Recall: {best_model_metrics['recall']:.3f}
- F1-Score: {best_model_metrics['f1_score']:.3f}

All Model Rankings:
"""
        for i, (name, metrics) in enumerate(rankings):
            report += f"{i+1}. {name}: {metrics['accuracy']:.3f}\n"
            
        print(report)
        
        return {
            'best_model': self.trainer_agent.trained_models[best_model_name]['model'],
            'all_results': eval_results,
            'rankings': rankings,
            'analysis': analysis
        }

# Run the complete orchestrated pipeline
orchestrator = MLPipelineOrchestrator()
pipeline_results = orchestrator.run_pipeline(X, y, wine.feature_names)

# %% [markdown]
# ## 📊 Workshop Summary and Next Steps
# 
# ### What We've Learned:
# 
# 1. **Ensemble Methods**:
#    - Voting: Combines predictions from multiple models
#    - Bagging: Reduces variance through bootstrap sampling
#    - Boosting: Reduces bias through sequential learning
#    - Stacking: Uses meta-learner to combine base models
# 
# 2. **LangGraph Concepts**:
#    - State management for complex workflows
#    - Agent-based design patterns
#    - Multi-agent coordination
#    - Integration with ML pipelines
# 
# 3. **Hybrid Systems**:
#    - Combining traditional ML with agent orchestration
#    - Specialized agents for different tasks
#    - Intelligent model selection and explanation

# %%
print("\n✅ Workshop Complete!")
print("=" * 50)
print("\n🎯 Next Steps:")
print("1. Complete the ensemble challenges in exercises/")
print("2. Build your own LangGraph agent for a specific task")
print("3. Experiment with different ensemble strategies")
print("4. Create a multi-agent system for your domain")
print("\n📚 Resources:")
print("- LangGraph Documentation: https://langchain-ai.github.io/langgraph/")
print("- Scikit-learn Ensemble Guide: https://scikit-learn.org/stable/modules/ensemble.html")
print("- Week 4 exercises and solutions in the course repository")
print("\n🚀 Ready for Week 5: Unsupervised Learning & Advanced LangGraph!")
