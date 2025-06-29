# Week 4: Ensemble Methods & LangGraph Exercises

Welcome to Week 4 exercises! These challenges will help you master ensemble methods and build your first LangGraph agents for ML workflows.

## 🎯 Learning Objectives

By completing these exercises, you will:
- Implement and compare different ensemble strategies
- Build custom ensemble methods for specific problems
- Create LangGraph agents for ML tasks
- Design multi-agent systems for complex workflows
- Integrate ensemble ML with agent-based interpretation

## 📋 Prerequisites

Before starting:
- [ ] Complete Week 4 workshop notebook
- [ ] Run the ensemble_pipeline.py successfully
- [ ] Install LangGraph: `pip install langgraph`
- [ ] Review ensemble theory and LangGraph concepts

## 🏆 Exercise Overview

| Exercise | Difficulty | Focus Area | Time Estimate |
|----------|------------|------------|---------------|
| [Exercise 1](#exercise-1-custom-ensemble) | 🟢 Beginner | Custom Voting | 30-45 min |
| [Exercise 2](#exercise-2-stacking-mastery) | 🟡 Intermediate | Advanced Stacking | 45-60 min |
| [Exercise 3](#exercise-3-first-langgraph-agent) | 🟡 Intermediate | LangGraph Basics | 60-75 min |
| [Exercise 4](#exercise-4-ml-pipeline-agent) | 🔴 Advanced | Agent ML Pipeline | 75-90 min |
| [Exercise 5](#exercise-5-multi-agent-ensemble) | 🔴 Advanced | Multi-Agent System | 90-120 min |
| [Bonus Challenge](#bonus-challenge) | 🟣 Expert | Production System | 120+ min |

---

## Exercise 1: Custom Weighted Voting Ensemble
**Difficulty:** 🟢 Beginner | **Time:** 30-45 minutes

### 📝 Problem Statement

Create a custom weighted voting ensemble where weights are determined by individual model performance on a validation set. This ensemble should dynamically adjust weights based on model confidence.

### 🎯 Tasks

1. **Implement Custom Weighted Voting**
   ```python
   class DynamicWeightedVotingClassifier:
       """
       Voting classifier with dynamic weights based on validation performance.
       """
       def __init__(self, estimators, weight_metric='accuracy'):
           self.estimators = estimators
           self.weight_metric = weight_metric
           self.weights_ = None
           self.models_ = {}
           
       def fit(self, X, y, validation_split=0.2):
           """
           Train all models and determine optimal weights.
           """
           # TODO: Implement training and weight calculation
           pass
           
       def predict_proba(self, X):
           """
           Weighted probability predictions.
           """
           # TODO: Implement weighted voting
           pass
   ```

2. **Compare with Standard Voting**
   - Train on multiple datasets (wine, breast_cancer, iris)
   - Compare against sklearn's VotingClassifier
   - Visualize weight distributions

3. **Add Confidence Thresholding**
   - Implement confidence-based prediction
   - Handle uncertain predictions differently

### 💡 Hints
- Use validation set performance to determine weights
- Consider normalizing weights to sum to 1
- Try different weight metrics (accuracy, F1, AUC)

### ✅ Success Criteria
- [ ] Custom voting classifier implemented
- [ ] Outperforms standard voting on at least 2 datasets
- [ ] Clear visualization of weight assignments
- [ ] Documentation of design decisions

---

## Exercise 2: Advanced Stacking with Meta-Features
**Difficulty:** 🟡 Intermediate | **Time:** 45-60 minutes

### 📝 Problem Statement

Build an advanced stacking ensemble that uses meta-features (predictions from base models + original features) and implements multiple levels of stacking.

### 🎯 Tasks

1. **Two-Level Stacking Architecture**
   ```python
   class MultiLevelStackingClassifier:
       """
       Multi-level stacking with meta-feature engineering.
       """
       def __init__(self, level1_estimators, level2_estimators, 
                    final_estimator, use_probas=True, 
                    include_original_features=True):
           # Initialize architecture
           pass
           
       def _create_meta_features(self, X, models, use_probas=True):
           """
           Create meta-features from model predictions.
           """
           # TODO: Generate meta-features
           pass
   ```

2. **Meta-Feature Engineering**
   - Prediction probabilities as features
   - Prediction entropy/uncertainty
   - Model agreement indicators
   - Original feature subset selection

3. **Optimization Strategy**
   - Cross-validation for meta-learner training
   - Feature importance in meta-model
   - Preventing overfitting in stacking

### 💡 Hints
- Use out-of-fold predictions to avoid overfitting
- Consider different meta-learner algorithms
- Visualize information flow through levels

### ✅ Success Criteria
- [ ] Multi-level stacking implemented
- [ ] Meta-feature engineering documented
- [ ] Performance improvement demonstrated
- [ ] Overfitting analysis included

---

## Exercise 3: Your First LangGraph Agent
**Difficulty:** 🟡 Intermediate | **Time:** 60-75 minutes

### 📝 Problem Statement

Build a LangGraph agent that assists with model selection by analyzing dataset characteristics and recommending appropriate algorithms.

### 🎯 Tasks

1. **Define Agent State**
   ```python
   from typing import TypedDict, List, Dict, Any
   from langgraph.graph import StateGraph, END
   
   class ModelSelectorState(TypedDict):
       """State for model selection agent."""
       dataset_path: str
       dataset_analysis: Dict[str, Any]
       recommended_models: List[str]
       model_rationale: Dict[str, str]
       training_results: Dict[str, float]
       final_recommendation: str
   ```

2. **Implement Agent Nodes**
   - **Data Analysis Node**: Analyze dataset characteristics
   - **Model Recommendation Node**: Suggest models based on analysis
   - **Training Node**: Train recommended models
   - **Evaluation Node**: Compare and select best model

3. **Create Conditional Routing**
   - Route based on dataset size
   - Different paths for classification/regression
   - Handle edge cases (imbalanced data, high dimensions)

### 💡 Example Structure
```python
def should_balance_classes(state):
    """Routing function for class balancing."""
    analysis = state['dataset_analysis']
    if analysis['class_imbalance_ratio'] > 3:
        return "balance_classes"
    return "proceed_training"

# Build graph
workflow = StateGraph(ModelSelectorState)
workflow.add_node("analyze", analyze_dataset)
workflow.add_node("recommend", recommend_models)
workflow.add_conditional_edges(
    "analyze",
    should_balance_classes,
    {
        "balance_classes": "balance",
        "proceed_training": "recommend"
    }
)
```

### ✅ Success Criteria
- [ ] Complete agent with 4+ nodes
- [ ] Conditional routing implemented
- [ ] Clear state management
- [ ] Agent successfully selects models

---

## Exercise 4: ML Pipeline Orchestration Agent
**Difficulty:** 🔴 Advanced | **Time:** 75-90 minutes

### 📝 Problem Statement

Create a sophisticated LangGraph agent that orchestrates an entire ML pipeline, from data preprocessing to model deployment, with intelligent decision-making at each step.

### 🎯 Tasks

1. **Complex State Management**
   ```python
   class MLPipelineState(TypedDict):
       # Data state
       raw_data: Any
       processed_data: Any
       feature_engineering_steps: List[str]
       
       # Model state
       candidate_models: Dict[str, Any]
       trained_models: Dict[str, Any]
       ensemble_models: Dict[str, Any]
       
       # Evaluation state
       validation_metrics: Dict[str, Dict[str, float]]
       test_metrics: Dict[str, Dict[str, float]]
       
       # Decision state
       decisions_made: List[Dict[str, Any]]
       pipeline_metadata: Dict[str, Any]
   ```

2. **Implement Pipeline Nodes**
   - Data validation and cleaning
   - Feature engineering decisions
   - Model selection and training
   - Ensemble creation
   - Performance monitoring
   - Deployment readiness check

3. **Advanced Features**
   - Parallel model training branches
   - Dynamic ensemble selection
   - Automated hyperparameter tuning
   - Performance-based re-routing

4. **Add Explainability**
   - Track all decisions made
   - Generate pipeline execution report
   - Visualize agent decision flow

### 💡 Implementation Guide
```python
class MLPipelineOrchestrator:
    def __init__(self):
        self.workflow = StateGraph(MLPipelineState)
        self._build_graph()
        
    def _build_graph(self):
        # Add all nodes
        self.workflow.add_node("validate_data", self.validate_data_node)
        self.workflow.add_node("engineer_features", self.feature_engineering_node)
        # ... more nodes
        
        # Add conditional edges
        self.workflow.add_conditional_edges(
            "validate_data",
            self.route_based_on_data_quality,
            {
                "clean_data": "data_cleaning",
                "proceed": "engineer_features",
                "abort": END
            }
        )
```

### ✅ Success Criteria
- [ ] Complete pipeline orchestration
- [ ] At least 3 conditional routing decisions
- [ ] Parallel processing implemented
- [ ] Clear decision tracking
- [ ] Performance optimization demonstrated

---

## Exercise 5: Multi-Agent Ensemble System
**Difficulty:** 🔴 Advanced | **Time:** 90-120 minutes

### 📝 Problem Statement

Build a multi-agent system where specialized agents collaborate to create and optimize ensemble models. Each agent has a specific role and they communicate to achieve optimal performance.

### 🎯 Tasks

1. **Design Agent Roles**
   - **Data Analyst Agent**: Analyzes data and suggests preprocessing
   - **Model Specialist Agent**: Expert in specific algorithm families
   - **Ensemble Architect Agent**: Designs ensemble strategies
   - **Performance Monitor Agent**: Tracks and optimizes performance
   - **Coordinator Agent**: Orchestrates agent collaboration

2. **Implement Communication Protocol**
   ```python
   class AgentMessage(TypedDict):
       sender: str
       receiver: str
       message_type: str  # 'request', 'response', 'broadcast'
       content: Dict[str, Any]
       timestamp: float
       priority: int
   
   class MultiAgentEnsembleSystem:
       def __init__(self):
           self.agents = {}
           self.message_queue = []
           self.shared_state = {}
   ```

3. **Agent Collaboration Patterns**
   - Request-Response for specific expertise
   - Broadcast for system-wide updates
   - Voting for ensemble decisions
   - Negotiation for resource allocation

4. **Optimization Loop**
   - Agents propose improvements
   - Coordinator evaluates proposals
   - Best proposals implemented
   - Performance feedback to all agents

### 💡 Example Agent Implementation
```python
class ModelSpecialistAgent:
    def __init__(self, specialty='tree_based'):
        self.specialty = specialty
        self.models = self._init_models()
        
    def receive_message(self, message: AgentMessage):
        if message['message_type'] == 'request':
            return self.handle_request(message)
        elif message['message_type'] == 'broadcast':
            self.update_knowledge(message)
            
    def propose_model(self, data_characteristics):
        # Propose best model based on specialty
        pass
```

### ✅ Success Criteria
- [ ] 5 specialized agents implemented
- [ ] Clear communication protocol
- [ ] Agents successfully collaborate
- [ ] Performance improvement through collaboration
- [ ] System handles edge cases gracefully

---

## Bonus Challenge: Production-Ready AutoML System
**Difficulty:** 🟣 Expert | **Time:** 120+ minutes

### 📝 Problem Statement

Combine everything learned to build a production-ready AutoML system using Metaflow for pipeline orchestration and LangGraph for intelligent decision-making.

### 🎯 Requirements

1. **Metaflow Pipeline Integration**
   - LangGraph agents as Metaflow steps
   - Parallel ensemble training
   - Artifact management
   - Resource optimization

2. **Advanced Features**
   - Automatic feature engineering
   - Neural architecture search for deep models
   - Bayesian optimization for hyperparameters
   - Multi-objective optimization (accuracy vs latency)

3. **Production Considerations**
   - Model versioning and rollback
   - A/B testing framework
   - Performance monitoring
   - Automated retraining triggers

4. **API and Interface**
   ```python
   class AutoMLSystem:
       def fit(self, X, y, task='classification', 
               time_budget=3600, optimization_metric='accuracy'):
           """Automatically build best model within constraints."""
           pass
           
       def predict(self, X, model_version='latest'):
           """Prediction with model versioning."""
           pass
           
       def explain(self, sample_index):
           """Explain prediction using best method."""
           pass
   ```

### ✅ Success Criteria
- [ ] Complete AutoML pipeline
- [ ] Handles classification and regression
- [ ] Production-ready error handling
- [ ] Performance benchmarking included
- [ ] Clear documentation and examples

---

## 📝 Submission Guidelines

For each exercise:

1. **Code Implementation**
   - Clean, well-documented code
   - Follow PEP 8 style guidelines
   - Include docstrings and type hints

2. **Results Documentation**
   - Performance metrics and comparisons
   - Visualizations of key concepts
   - Analysis of design decisions

3. **Reflection Questions**
   - What challenges did you encounter?
   - How did ensemble methods improve performance?
   - What are the tradeoffs in your design?
   - How would you scale this for production?

## 🎁 Extra Credit Opportunities

1. **Implement XGBoost/LightGBM Integration**
   - Add advanced boosting libraries
   - Compare with sklearn implementations
   - Optimize for specific metrics

2. **Create Ensemble Explanation System**
   - Use SHAP/LIME for ensemble models
   - Aggregate explanations across models
   - Build interpretation dashboard

3. **Design Domain-Specific Ensemble**
   - Choose a specific domain (medical, financial, etc.)
   - Create specialized ensemble for that domain
   - Include domain knowledge in design

## 💡 Resources

- [Ensemble Learning Guide](https://scikit-learn.org/stable/modules/ensemble.html)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Metaflow Best Practices](https://docs.metaflow.org/metaflow/basics)
- [Multi-Agent Systems](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)

---

**Good luck with your exercises! Remember: the goal is not just to complete them, but to deeply understand ensemble methods and agent-based systems. 🚀**