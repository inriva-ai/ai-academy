# LangGraph Basics: Building Your First Agents
# =============================================
# A gentle introduction to LangGraph for ML workflows

# %% [markdown]
# # LangGraph Fundamentals
# 
# Welcome to LangGraph! This notebook provides a step-by-step introduction to building
# agent-based systems for ML workflows. LangGraph allows us to create stateful,
# multi-step applications with intelligent routing and decision-making.
# 
# ## What You'll Learn:
# 1. Core LangGraph concepts (State, Nodes, Edges)
# 2. Building your first agent
# 3. Conditional routing and decision-making
# 4. Integrating with ML workflows
# 5. Multi-agent coordination basics

# %%
# Setup and imports
import warnings
warnings.filterwarnings('ignore')

print("🔧 Setting up LangGraph environment...")

# Core imports
from typing import TypedDict, Annotated, List, Dict, Any, Sequence
import operator
import json
import random
import time
from datetime import datetime

# Check LangGraph availability
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor, ToolInvocation
    from langgraph.checkpoint import MemorySaver
    print("✅ LangGraph imported successfully!")
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("⚠️ LangGraph not installed. Install with: pip install langgraph")
    print("   We'll demonstrate concepts with mock implementations.")
    LANGGRAPH_AVAILABLE = False

# ML imports
import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# %% [markdown]
# ## 1. Understanding LangGraph State
# 
# The state is the core concept in LangGraph. It represents all the information
# that flows through your agent system.

# %%
if LANGGRAPH_AVAILABLE:
    # Define a simple state for a data analysis agent
    class DataAnalysisState(TypedDict):
        """
        State definition for our data analysis agent.
        Each field represents information that persists across nodes.
        """
        # Input data
        dataset_name: str
        data: Any
        
        # Analysis results
        n_samples: int
        n_features: int
        feature_names: List[str]
        class_distribution: Dict[str, int]
        
        # Messages and history
        messages: Annotated[Sequence[str], operator.add]
        analysis_complete: bool
    
    print("✅ State schema defined!")
    print("\nState fields:")
    print("- dataset_name: Name of the dataset")
    print("- data: The actual data")
    print("- n_samples, n_features: Data dimensions")
    print("- messages: List of status messages (accumulates)")
    print("- analysis_complete: Boolean flag")
else:
    print("📝 Example state structure:")
    print("""
    State is like a shared notebook that all parts of your agent can read and write:
    - Input fields: What the agent receives
    - Working fields: Intermediate calculations
    - Output fields: Final results
    - History fields: Track what happened
    """)

# %% [markdown]
# ## 2. Creating Agent Nodes
# 
# Nodes are the building blocks of your agent. Each node is a function that:
# 1. Receives the current state
# 2. Performs some operation
# 3. Updates the state
# 4. Returns the modified state

# %%
# Node 1: Data Loader
def load_data_node(state: Dict) -> Dict:
    """
    Node that loads a dataset based on the dataset name.
    """
    print(f"📊 [Data Loader] Loading dataset: {state['dataset_name']}")
    
    # Simulate loading different datasets
    if state['dataset_name'] == 'iris':
        from sklearn.datasets import load_iris
        data = load_iris()
        X, y = data.data, data.target
        feature_names = data.feature_names
    else:
        # Create synthetic dataset
        X, y = make_classification(
            n_samples=1000, 
            n_features=20, 
            n_informative=15,
            n_classes=3,
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(20)]
    
    # Update state
    state['data'] = (X, y)
    state['feature_names'] = feature_names
    state['messages'].append(f"Loaded dataset: {state['dataset_name']}")
    
    return state

# Node 2: Data Analyzer
def analyze_data_node(state: Dict) -> Dict:
    """
    Node that analyzes the loaded data.
    """
    print("🔍 [Data Analyzer] Analyzing dataset...")
    
    X, y = state['data']
    
    # Perform analysis
    state['n_samples'] = X.shape[0]
    state['n_features'] = X.shape[1]
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    state['class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
    
    # Add analysis message
    state['messages'].append(
        f"Analysis complete: {state['n_samples']} samples, "
        f"{state['n_features']} features, "
        f"{len(state['class_distribution'])} classes"
    )
    
    return state

# Node 3: Report Generator
def generate_report_node(state: Dict) -> Dict:
    """
    Node that generates a summary report.
    """
    print("📝 [Report Generator] Creating report...")
    
    report = f"""
    Dataset Analysis Report
    ======================
    Dataset: {state['dataset_name']}
    Samples: {state['n_samples']}
    Features: {state['n_features']}
    
    Class Distribution:
    {json.dumps(state['class_distribution'], indent=2)}
    
    Feature Names:
    {', '.join(state['feature_names'][:5])}...
    """
    
    state['messages'].append("Report generated successfully")
    state['analysis_complete'] = True
    
    print(report)
    
    return state

print("✅ Agent nodes defined!")

# %% [markdown]
# ## 3. Building Your First Agent Graph
# 
# Now let's connect the nodes into a working agent using StateGraph.

# %%
if LANGGRAPH_AVAILABLE:
    # Create the graph
    workflow = StateGraph(DataAnalysisState)
    
    # Add nodes to the graph
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("analyze_data", analyze_data_node)
    workflow.add_node("generate_report", generate_report_node)
    
    # Define the flow
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "analyze_data")
    workflow.add_edge("analyze_data", "generate_report")
    workflow.add_edge("generate_report", END)
    
    # Compile the graph
    app = workflow.compile()
    
    print("✅ Agent graph compiled successfully!")
    print("\nGraph structure:")
    print("Start → Load Data → Analyze Data → Generate Report → End")
    
    # Visualize if possible
    try:
        # This would display the graph visually in Jupyter
        # from IPython.display import Image, display
        # display(Image(app.get_graph().draw_png()))
        pass
    except:
        print("(Graph visualization not available)")
    
    # Run the agent
    print("\n🚀 Running the agent...")
    print("=" * 50)
    
    # Initial state
    initial_state = {
        "dataset_name": "iris",
        "messages": [],
        "analysis_complete": False
    }
    
    # Execute the agent
    final_state = app.invoke(initial_state)
    
    print("\n✅ Agent execution complete!")
    print(f"\nMessages log:")
    for msg in final_state['messages']:
        print(f"  - {msg}")
else:
    print("📝 Agent workflow example:")
    print("""
    # Define the workflow
    workflow = StateGraph(DataAnalysisState)
    
    # Add nodes
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("analyze_data", analyze_data_node)
    workflow.add_node("generate_report", generate_report_node)
    
    # Connect nodes
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "analyze_data")
    workflow.add_edge("analyze_data", "generate_report")
    workflow.add_edge("generate_report", END)
    
    # Compile and run
    app = workflow.compile()
    result = app.invoke(initial_state)
    """)

# %% [markdown]
# ## 4. Conditional Routing
# 
# One of LangGraph's powerful features is conditional routing - making decisions
# about which path to take based on the current state.

# %%
# Define a more complex state with routing
if LANGGRAPH_AVAILABLE:
    class MLPipelineState(TypedDict):
        """State for ML pipeline with conditional routing."""
        dataset_name: str
        data: Any
        n_samples: int
        n_classes: int
        
        # Routing decisions
        needs_balancing: bool
        model_type: str  # 'simple' or 'complex'
        
        # Results
        model: Any
        accuracy: float
        messages: Annotated[Sequence[str], operator.add]

# Routing function
def route_based_on_data_size(state: Dict) -> str:
    """
    Decide which model to use based on dataset size.
    """
    if state['n_samples'] < 1000:
        return "simple_model"
    else:
        return "complex_model"

def route_based_on_balance(state: Dict) -> str:
    """
    Check if dataset needs balancing.
    """
    class_dist = state.get('class_distribution', {})
    if class_dist:
        counts = list(class_dist.values())
        imbalance_ratio = max(counts) / min(counts)
        if imbalance_ratio > 2:
            return "balance_data"
    return "skip_balancing"

# Nodes for different paths
def train_simple_model_node(state: Dict) -> Dict:
    """Train a simple model for small datasets."""
    print("🤖 [Simple Model] Training logistic regression...")
    
    from sklearn.linear_model import LogisticRegression
    X, y = state['data']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    
    state['model'] = model
    state['accuracy'] = accuracy
    state['model_type'] = 'simple'
    state['messages'].append(f"Simple model trained: {accuracy:.3f} accuracy")
    
    return state

def train_complex_model_node(state: Dict) -> Dict:
    """Train a complex model for large datasets."""
    print("🤖 [Complex Model] Training random forest ensemble...")
    
    X, y = state['data']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    
    state['model'] = model
    state['accuracy'] = accuracy
    state['model_type'] = 'complex'
    state['messages'].append(f"Complex model trained: {accuracy:.3f} accuracy")
    
    return state

def balance_data_node(state: Dict) -> Dict:
    """Balance the dataset using SMOTE or similar."""
    print("⚖️ [Data Balancer] Balancing classes...")
    
    # Simulate balancing (in real implementation, use SMOTE or similar)
    state['messages'].append("Data balanced using oversampling")
    state['needs_balancing'] = False
    
    return state

# Build conditional workflow
if LANGGRAPH_AVAILABLE:
    # Create new workflow with conditional routing
    ml_workflow = StateGraph(MLPipelineState)
    
    # Add all nodes
    ml_workflow.add_node("load_data", load_data_node)
    ml_workflow.add_node("analyze_data", analyze_data_node)
    ml_workflow.add_node("balance_data", balance_data_node)
    ml_workflow.add_node("train_simple", train_simple_model_node)
    ml_workflow.add_node("train_complex", train_complex_model_node)
    ml_workflow.add_node("generate_report", generate_report_node)
    
    # Set entry point
    ml_workflow.set_entry_point("load_data")
    
    # Add edges
    ml_workflow.add_edge("load_data", "analyze_data")
    
    # Conditional routing after analysis
    ml_workflow.add_conditional_edges(
        "analyze_data",
        route_based_on_balance,
        {
            "balance_data": "balance_data",
            "skip_balancing": "train_simple"  # Default to simple for now
        }
    )
    
    # After balancing, decide on model
    ml_workflow.add_conditional_edges(
        "balance_data",
        route_based_on_data_size,
        {
            "simple_model": "train_simple",
            "complex_model": "train_complex"
        }
    )
    
    # Both training paths lead to report
    ml_workflow.add_edge("train_simple", "generate_report")
    ml_workflow.add_edge("train_complex", "generate_report")
    ml_workflow.add_edge("generate_report", END)
    
    print("✅ Conditional workflow created!")
    print("\nWorkflow includes:")
    print("- Automatic model selection based on data size")
    print("- Optional data balancing for imbalanced datasets")
    print("- Different training paths for different scenarios")

# %% [markdown]
# ## 5. Building an ML Assistant Agent
# 
# Let's create a practical agent that helps with the entire ML workflow,
# making intelligent decisions at each step.

# %%
class MLAssistant:
    """
    An intelligent ML assistant that guides through the entire workflow.
    """
    
    def __init__(self):
        self.workflow = None
        self.setup_workflow()
    
    def setup_workflow(self):
        """Set up the complete ML assistant workflow."""
        if not LANGGRAPH_AVAILABLE:
            print("LangGraph not available - using mock implementation")
            return
        
        # Define comprehensive state
        class MLAssistantState(TypedDict):
            # Input
            task_description: str
            data: Any
            
            # Analysis
            data_characteristics: Dict[str, Any]
            preprocessing_steps: List[str]
            
            # Model selection
            candidate_models: List[str]
            selected_model: str
            hyperparameters: Dict[str, Any]
            
            # Training
            trained_model: Any
            validation_scores: Dict[str, float]
            
            # Results
            final_report: str
            recommendations: List[str]
            messages: Annotated[Sequence[str], operator.add]
        
        # Create workflow
        self.workflow = StateGraph(MLAssistantState)
        
        # Add nodes
        self.workflow.add_node("understand_task", self._understand_task)
        self.workflow.add_node("analyze_data", self._analyze_data)
        self.workflow.add_node("select_preprocessing", self._select_preprocessing)
        self.workflow.add_node("recommend_models", self._recommend_models)
        self.workflow.add_node("optimize_hyperparameters", self._optimize_hyperparameters)
        self.workflow.add_node("train_model", self._train_model)
        self.workflow.add_node("evaluate_model", self._evaluate_model)
        self.workflow.add_node("generate_recommendations", self._generate_recommendations)
        
        # Define flow
        self.workflow.set_entry_point("understand_task")
        self.workflow.add_edge("understand_task", "analyze_data")
        self.workflow.add_edge("analyze_data", "select_preprocessing")
        self.workflow.add_edge("select_preprocessing", "recommend_models")
        self.workflow.add_edge("recommend_models", "optimize_hyperparameters")
        self.workflow.add_edge("optimize_hyperparameters", "train_model")
        self.workflow.add_edge("train_model", "evaluate_model")
        self.workflow.add_edge("evaluate_model", "generate_recommendations")
        self.workflow.add_edge("generate_recommendations", END)
        
        # Compile
        self.app = self.workflow.compile()
        
    def _understand_task(self, state):
        """Understand the ML task from description."""
        state['messages'].append(f"Understanding task: {state['task_description']}")
        return state
    
    def _analyze_data(self, state):
        """Analyze dataset characteristics."""
        X, y = state['data']
        
        state['data_characteristics'] = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'feature_types': 'numerical',  # Simplified
            'missing_values': False,
            'imbalance_ratio': self._calculate_imbalance(y)
        }
        
        state['messages'].append("Data analysis complete")
        return state
    
    def _calculate_imbalance(self, y):
        """Calculate class imbalance ratio."""
        unique, counts = np.unique(y, return_counts=True)
        return max(counts) / min(counts)
    
    def _select_preprocessing(self, state):
        """Select appropriate preprocessing steps."""
        steps = []
        
        # Based on data characteristics
        if state['data_characteristics']['missing_values']:
            steps.append('imputation')
        
        if state['data_characteristics']['imbalance_ratio'] > 2:
            steps.append('balancing')
        
        steps.extend(['scaling', 'feature_selection'])
        
        state['preprocessing_steps'] = steps
        state['messages'].append(f"Selected preprocessing: {', '.join(steps)}")
        return state
    
    def _recommend_models(self, state):
        """Recommend models based on task and data."""
        n_samples = state['data_characteristics']['n_samples']
        n_features = state['data_characteristics']['n_features']
        
        candidates = []
        
        # Rule-based recommendations
        if n_samples < 1000:
            candidates.extend(['logistic_regression', 'svm'])
        
        if n_features < 50:
            candidates.extend(['random_forest', 'gradient_boosting'])
        
        if n_samples > 5000:
            candidates.append('neural_network')
        
        # Always include a baseline
        candidates.append('naive_bayes')
        
        state['candidate_models'] = list(set(candidates))
        state['messages'].append(f"Recommended models: {', '.join(candidates)}")
        return state
    
    def _optimize_hyperparameters(self, state):
        """Optimize hyperparameters for selected model."""
        # Simplified - just set some defaults
        state['selected_model'] = state['candidate_models'][0]
        
        hyperparams = {
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'logistic_regression': {'C': 1.0, 'max_iter': 1000},
            'svm': {'C': 1.0, 'kernel': 'rbf'},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1}
        }
        
        state['hyperparameters'] = hyperparams.get(
            state['selected_model'], 
            {}
        )
        
        state['messages'].append(f"Optimized hyperparameters for {state['selected_model']}")
        return state
    
    def _train_model(self, state):
        """Train the selected model."""
        # Simplified training
        X, y = state['data']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train based on selected model
        if state['selected_model'] == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**state['hyperparameters'])
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
        
        model.fit(X_train, y_train)
        
        # Calculate scores
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        state['trained_model'] = model
        state['validation_scores'] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'overfit_score': train_score - test_score
        }
        
        state['messages'].append(f"Model trained: {test_score:.3f} test accuracy")
        return state
    
    def _evaluate_model(self, state):
        """Evaluate model performance."""
        scores = state['validation_scores']
        
        # Simple evaluation logic
        performance = "good" if scores['test_accuracy'] > 0.8 else "needs improvement"
        overfit = "yes" if scores['overfit_score'] > 0.1 else "no"
        
        state['messages'].append(f"Performance: {performance}, Overfitting: {overfit}")
        return state
    
    def _generate_recommendations(self, state):
        """Generate final recommendations."""
        recommendations = []
        
        # Based on results
        if state['validation_scores']['test_accuracy'] < 0.8:
            recommendations.append("Consider ensemble methods for better performance")
            recommendations.append("Try feature engineering")
        
        if state['validation_scores']['overfit_score'] > 0.1:
            recommendations.append("Add regularization to reduce overfitting")
            recommendations.append("Collect more training data")
        
        state['recommendations'] = recommendations
        state['final_report'] = self._create_report(state)
        
        return state
    
    def _create_report(self, state):
        """Create a comprehensive report."""
        report = f"""
        ML Assistant Report
        ==================
        
        Task: {state['task_description']}
        
        Data Characteristics:
        - Samples: {state['data_characteristics']['n_samples']}
        - Features: {state['data_characteristics']['n_features']}
        - Classes: {state['data_characteristics']['n_classes']}
        
        Selected Model: {state['selected_model']}
        Performance:
        - Train Accuracy: {state['validation_scores']['train_accuracy']:.3f}
        - Test Accuracy: {state['validation_scores']['test_accuracy']:.3f}
        
        Recommendations:
        {chr(10).join(f"- {rec}" for rec in state['recommendations'])}
        """
        
        return report
    
    def assist(self, task_description, X, y):
        """Run the ML assistant."""
        if not LANGGRAPH_AVAILABLE:
            print("LangGraph not available - cannot run assistant")
            return None
        
        initial_state = {
            'task_description': task_description,
            'data': (X, y),
            'messages': []
        }
        
        print("🤖 ML Assistant starting...")
        result = self.app.invoke(initial_state)
        
        print("\n" + result['final_report'])
        
        return result

# Create and demonstrate the assistant
if LANGGRAPH_AVAILABLE:
    print("🚀 Creating ML Assistant...")
    assistant = MLAssistant()
    
    # Create sample data
    X, y = make_classification(
        n_samples=1500, 
        n_features=20, 
        n_informative=15,
        n_classes=3,
        random_state=42
    )
    
    # Run assistant
    result = assistant.assist(
        "Classify customer segments based on behavior data",
        X, y
    )
    
    print("\n📝 Process Log:")
    for msg in result['messages']:
        print(f"  - {msg}")

# %% [markdown]
# ## 6. Next Steps and Advanced Patterns
# 
# ### What We've Learned:
# 1. **State Management**: How to define and update agent state
# 2. **Node Creation**: Building modular processing units
# 3. **Graph Construction**: Connecting nodes into workflows
# 4. **Conditional Routing**: Making decisions based on state
# 5. **ML Integration**: Using agents for ML workflows
# 
# ### Advanced Patterns to Explore:
# 
# 1. **Parallel Execution**
#    ```python
#    workflow.add_edge("preprocess", ["model1", "model2", "model3"])
#    ```
# 
# 2. **Human-in-the-Loop**
#    ```python
#    workflow.add_node("human_review", human_review_node)
#    workflow.add_conditional_edges("evaluate", needs_human_review)
#    ```
# 
# 3. **Tool Integration**
#    ```python
#    from langchain.tools import Tool
#    ml_tools = [train_model_tool, evaluate_tool, optimize_tool]
#    ```
# 
# 4. **Memory and Persistence**
#    ```python
#    checkpointer = MemorySaver()
#    app = workflow.compile(checkpointer=checkpointer)
#    ```

# %%
print("\n✅ Tutorial Complete!")
print("=" * 50)
print("\n🎯 Your Next Steps:")
print("1. Install LangGraph: pip install langgraph")
print("2. Try modifying the ML Assistant for your use case")
print("3. Experiment with conditional routing patterns")
print("4. Build a multi-agent system for ensemble learning")
print("5. Integrate with your Metaflow pipelines")
print("\n📚 Resources:")
print("- LangGraph Docs: https://langchain-ai.github.io/langgraph/")
print("- Multi-Agent Tutorial: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/")
print("- Week 4 Exercises: Practice building your own agents!")
print("\n🚀 Ready to build intelligent ML systems with LangGraph!")
