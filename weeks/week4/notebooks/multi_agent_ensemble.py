"""
Week 4: Multi-Agent Ensemble System with LangGraph
=================================================
This advanced example demonstrates a multi-agent system where specialized
agents collaborate to build and optimize ensemble models.

Run this script:
    python multi_agent_ensemble.py
"""

import numpy as np
import pandas as pd
from typing import TypedDict, Annotated, List, Dict, Any, Sequence, Literal
import operator
from datetime import datetime
import json

# ML imports
from sklearn.datasets import load_wine, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Check LangGraph availability
try:
    from langgraph.graph import StateGraph, END
    # from langgraph.prebuilt import ToolExecutor
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("⚠️ LangGraph not installed. Install with: pip install langgraph")
    LANGGRAPH_AVAILABLE = False


# Define the shared state for all agents
class EnsembleSystemState(TypedDict):
    """
    Shared state for the multi-agent ensemble system.
    All agents read and write to this state.
    """
    # Data information
    dataset_name: str
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    n_samples: int
    n_features: int
    n_classes: int
    
    # Data characteristics (from Data Analyst)
    data_report: Dict[str, Any]
    preprocessing_recommendations: List[str]
    
    # Model recommendations (from Model Specialists)
    tree_models_report: Dict[str, Any]
    linear_models_report: Dict[str, Any]
    ensemble_recommendations: List[Dict[str, Any]]
    
    # Training results (from Trainer Agent)
    trained_models: Dict[str, Any]
    model_scores: Dict[str, Dict[str, float]]
    
    # Ensemble design (from Ensemble Architect)
    ensemble_strategy: str
    ensemble_config: Dict[str, Any]
    ensemble_model: Any
    ensemble_performance: Dict[str, float]
    
    # Optimization (from Optimizer Agent)
    optimization_history: List[Dict[str, Any]]
    best_configuration: Dict[str, Any]
    final_model: Any
    
    # Communication log
    messages: Annotated[Sequence[Dict[str, str]], operator.add]
    
    # Control flow
    phase: str  # 'analysis', 'training', 'ensemble', 'optimization', 'complete'


class DataAnalystAgent:
    """
    Agent responsible for analyzing data and providing insights.
    """
    
    def __init__(self):
        self.name = "DataAnalyst"
    
    def analyze(self, state: Dict) -> Dict:
        """Analyze dataset and provide recommendations."""
        print(f"📊 [{self.name}] Analyzing dataset...")
        
        X_train = state['X_train']
        y_train = state['y_train']
        
        # Perform analysis
        analysis = {
            'basic_stats': {
                'n_samples': X_train.shape[0],
                'n_features': X_train.shape[1],
                'n_classes': len(np.unique(y_train))
            },
            'feature_stats': {
                'mean': np.mean(X_train, axis=0).tolist()[:5],  # First 5 features
                'std': np.std(X_train, axis=0).tolist()[:5],
                'has_missing': np.any(np.isnan(X_train)),
                'correlation_matrix': np.corrcoef(X_train.T)[:5, :5].tolist()
            },
            'target_distribution': dict(zip(*np.unique(y_train, return_counts=True))),
            'imbalance_ratio': self._calculate_imbalance_ratio(y_train)
        }
        
        # Generate recommendations
        recommendations = []
        
        if analysis['imbalance_ratio'] > 2:
            recommendations.append("class_balancing")
            
        if analysis['feature_stats']['has_missing']:
            recommendations.append("imputation")
            
        if X_train.shape[1] > 50:
            recommendations.append("feature_selection")
            
        recommendations.extend(["scaling", "cross_validation"])
        return {
            'data_report': analysis,
            'preprocessing_recommendations': recommendations,
            'messages': [{
                'agent': self.name,
                'message': f"Analysis complete. Found {len(recommendations)} preprocessing recommendations."
            }]
        }
    
    def _calculate_imbalance_ratio(self, y):
        """Calculate class imbalance ratio."""
        counts = np.bincount(y)
        return max(counts) / min(counts) if min(counts) > 0 else float('inf')


class TreeBasedSpecialist:
    """
    Agent specialized in tree-based models.
    """
    
    def __init__(self):
        self.name = "TreeSpecialist"
        self.expertise = ['random_forest', 'gradient_boosting', 'extra_trees']
    
    def recommend_models(self, state: Dict) -> Dict:
        """Recommend tree-based models based on data characteristics."""
        print(f"🌲 [{self.name}] Analyzing tree-based model suitability...")
        
        data_report = state['data_report']
        recommendations = []
        
        # Random Forest - good for most cases
        rf_config = {
            'model_type': 'RandomForestClassifier',
            'base_params': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': 42
            },
            'reasoning': "Random Forest: Robust to overfitting, handles non-linear patterns well"
        }
        
        # Adjust based on data
        if data_report['basic_stats']['n_samples'] < 1000:
            rf_config['base_params']['n_estimators'] = 50
            rf_config['base_params']['max_depth'] = 10
            
        recommendations.append(rf_config)
        
        # Gradient Boosting - for high accuracy
        if data_report['basic_stats']['n_samples'] > 500:
            gb_config = {
                'model_type': 'GradientBoostingClassifier',
                'base_params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42
                },
                'reasoning': "Gradient Boosting: High accuracy potential, good for competitions"
            }
            recommendations.append(gb_config)
        return {
            'tree_models_report': {
                'specialist': self.name,
                'recommendations': recommendations,
                'confidence': 0.85 if len(recommendations) > 1 else 0.7
            },
            'messages': [{
                'agent': self.name,
                'message': f"Recommended {len(recommendations)} tree-based models."
            }]
        }
    

class LinearModelsSpecialist:
    """
    Agent specialized in linear and kernel-based models.
    """
    
    def __init__(self):
        self.name = "LinearSpecialist"
        self.expertise = ['logistic_regression', 'svm', 'naive_bayes']
    
    def recommend_models(self, state: Dict) -> Dict:
        """Recommend linear/kernel models based on data characteristics."""
        print(f"📏 [{self.name}] Analyzing linear model suitability...")
        
        data_report = state['data_report']
        recommendations = []
        
        # Logistic Regression - baseline
        lr_config = {
            'model_type': 'LogisticRegression',
            'base_params': {
                'max_iter': 1000,
                'random_state': 42
            },
            'reasoning': "Logistic Regression: Fast, interpretable baseline"
        }
        
        # Add regularization for high dimensions
        if data_report['basic_stats']['n_features'] > 20:
            lr_config['base_params']['penalty'] = 'l2'
            lr_config['base_params']['C'] = 0.1
            
        recommendations.append(lr_config)
        
        # SVM for non-linear patterns
        if data_report['basic_stats']['n_samples'] < 5000:
            svm_config = {
                'model_type': 'SVC',
                'base_params': {
                    'kernel': 'rbf',
                    'probability': True,
                    'random_state': 42
                },
                'reasoning': "SVM: Good for non-linear boundaries, robust to outliers"
            }
            recommendations.append(svm_config)
        return {
            'linear_models_report': {
                'specialist': self.name,
                'recommendations': recommendations,
                'confidence': 0.8
            },
            'messages': [{
                'agent': self.name,
                'message': f"Recommended {len(recommendations)} linear/kernel models."
            }]
        }
    

class TrainerAgent:
    """
    Agent responsible for training all recommended models.
    """
    
    def __init__(self):
        self.name = "Trainer"
    
    def train_models(self, state: Dict) -> Dict:
        """Train all recommended models."""
        print(f"🏋️ [{self.name}] Training recommended models...")
        
        # Collect all model recommendations
        all_models = []
        
        if 'tree_models_report' in state:
            all_models.extend(state['tree_models_report']['recommendations'])
            
        if 'linear_models_report' in state:
            all_models.extend(state['linear_models_report']['recommendations'])
        
        # Train each model
        trained_models = {}
        model_scores = {}
        
        for config in all_models:
            model_type = config['model_type']
            params = config['base_params']
            
            # Create model instance
            if model_type == 'RandomForestClassifier':
                model = RandomForestClassifier(**params)
            elif model_type == 'GradientBoostingClassifier':
                model = GradientBoostingClassifier(**params)
            elif model_type == 'LogisticRegression':
                model = LogisticRegression(**params)
            elif model_type == 'SVC':
                model = SVC(**params)
            else:
                continue
            
            # Train model
            model.fit(state['X_train'], state['y_train'])
            
            # Evaluate
            train_score = model.score(state['X_train'], state['y_train'])
            test_score = model.score(state['X_test'], state['y_test'])
            
            # Cross-validation
            cv_scores = cross_val_score(model, state['X_train'], state['y_train'], cv=5)
            
            # Store results
            trained_models[model_type] = model
            model_scores[model_type] = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'config': config
            }
            
            print(f"   ✓ {model_type}: Test Acc = {test_score:.3f}")
        return {
            'trained_models': trained_models,
            'model_scores': model_scores,
            'messages': [{
                'agent': self.name,
                'message': f"Trained {len(trained_models)} models successfully."
            }]
        }
    

class EnsembleArchitect:
    """
    Agent responsible for designing ensemble strategies.
    """
    
    def __init__(self):
        self.name = "EnsembleArchitect"
    
    def design_ensemble(self, state: Dict) -> Dict:
        """Design optimal ensemble strategy based on model performance."""
        print(f"🏗️ [{self.name}] Designing ensemble strategy...")
        
        model_scores = state['model_scores']
        trained_models = state['trained_models']
        
        # Analyze model diversity
        diversity_score = self._calculate_diversity(model_scores)
        
        # Select ensemble strategy
        if diversity_score > 0.7 and len(trained_models) >= 3:
            strategy = "stacking"
            print(f"   High diversity ({diversity_score:.2f}) → Using stacking")
        elif len(trained_models) >= 2:
            strategy = "voting"
            print(f"   Moderate diversity → Using voting")
        else:
            strategy = "single_best"
            print(f"   Low diversity → Using best single model")
        
        # Create ensemble
        if strategy == "voting":
            # Select top models for voting
            sorted_models = sorted(
                model_scores.items(),
                key=lambda x: x[1]['test_accuracy'],
                reverse=True
            )[:3]
            
            estimators = [
                (name, trained_models[name]) 
                for name, _ in sorted_models
            ]
            
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            config = {
                'strategy': 'voting',
                'models': [name for name, _ in sorted_models],
                'voting_type': 'soft'
            }
            
        elif strategy == "stacking":
            # Use top models as base, logistic regression as meta
            sorted_models = sorted(
                model_scores.items(),
                key=lambda x: x[1]['test_accuracy'],
                reverse=True
            )[:3]
            
            estimators = [
                (name, trained_models[name]) 
                for name, _ in sorted_models
            ]
            
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5
            )
            config = {
                'strategy': 'stacking',
                'base_models': [name for name, _ in sorted_models],
                'meta_learner': 'LogisticRegression'
            }
            
        else:
            # Use best single model
            best_model = max(
                model_scores.items(),
                key=lambda x: x[1]['test_accuracy']
            )
            ensemble = trained_models[best_model[0]]
            config = {
                'strategy': 'single_best',
                'model': best_model[0]
            }
        
        # Train ensemble
        if strategy in ['voting', 'stacking']:
            ensemble.fit(state['X_train'], state['y_train'])
        
        # Evaluate ensemble
        ensemble_train_score = ensemble.score(state['X_train'], state['y_train'])
        ensemble_test_score = ensemble.score(state['X_test'], state['y_test'])
        return {
            'ensemble_strategy': strategy,
            'ensemble_config': config,
            'ensemble_model': ensemble,
            'ensemble_performance': {
                'train_accuracy': ensemble_train_score,
                'test_accuracy': ensemble_test_score,
                'improvement': ensemble_test_score - max(
                    scores['test_accuracy'] 
                    for scores in model_scores.values()
                )
            },
            'messages': [{
                'agent': self.name,
                'message': f"Created {strategy} ensemble with {ensemble_test_score:.3f} test accuracy."
            }]
        }
    
    def _calculate_diversity(self, model_scores):
        """Calculate diversity score based on performance variance."""
        accuracies = [scores['test_accuracy'] for scores in model_scores.values()]
        if len(accuracies) < 2:
            return 0.0
        
        # Simple diversity: variance in accuracies
        variance = np.var(accuracies)
        # Normalize to 0-1 range
        return min(variance * 100, 1.0)


class OptimizerAgent:
    """
    Agent responsible for final optimization and recommendations.
    """
    
    def __init__(self):
        self.name = "Optimizer"
    
    def optimize_and_finalize(self, state: Dict) -> Dict:
        """Perform final optimization and generate recommendations."""
        print(f"⚡ [{self.name}] Performing final optimization...")
        
        # Analyze all results
        single_model_scores = state['model_scores']
        ensemble_performance = state['ensemble_performance']
        
        # Generate optimization history
        optimization_history = []
        
        # Record all attempts
        for model_name, scores in single_model_scores.items():
            optimization_history.append({
                'iteration': len(optimization_history) + 1,
                'model': model_name,
                'test_accuracy': scores['test_accuracy'],
                'type': 'single'
            })
        
        optimization_history.append({
            'iteration': len(optimization_history) + 1,
            'model': state['ensemble_strategy'],
            'test_accuracy': ensemble_performance['test_accuracy'],
            'type': 'ensemble'
        })
        
        # Determine best configuration
        if ensemble_performance['improvement'] > 0.01:
            best_config = {
                'type': 'ensemble',
                'strategy': state['ensemble_strategy'],
                'config': state['ensemble_config'],
                'performance': ensemble_performance
            }
            final_model = state['ensemble_model']
        else:
            # Ensemble didn't improve, use best single
            best_single = max(
                single_model_scores.items(),
                key=lambda x: x[1]['test_accuracy']
            )
            best_config = {
                'type': 'single',
                'model': best_single[0],
                'config': best_single[1]['config'],
                'performance': {
                    'test_accuracy': best_single[1]['test_accuracy']
                }
            }
            final_model = state['trained_models'][best_single[0]]
        
        # Generate final recommendations
        recommendations = self._generate_recommendations(state, best_config)
        
        # Update state
        state['optimization_history'] = optimization_history
        state['best_configuration'] = best_config
        state['final_model'] = final_model
        
        state['messages'].append({
            'agent': self.name,
            'message': f"Optimization complete. Best accuracy: {best_config['performance']['test_accuracy']:.3f}"
        })
        
        # Set phase to complete
        state['phase'] = 'complete'
        
        return state
    
    def _generate_recommendations(self, state, best_config):
        """Generate actionable recommendations."""
        recommendations = []
        
        # Based on performance
        if best_config['performance']['test_accuracy'] < 0.8:
            recommendations.append("Consider collecting more training data")
            recommendations.append("Try advanced feature engineering")
            
        # Based on data characteristics
        if state['data_report']['imbalance_ratio'] > 3:
            recommendations.append("Implement SMOTE or class weighting")
            
        # Based on ensemble performance
        if state['ensemble_performance']['improvement'] < 0:
            recommendations.append("Models may be too similar - try more diverse algorithms")
            
        return recommendations


class MultiAgentEnsembleSystem:
    """
    Orchestrator for the multi-agent ensemble system.
    """
    
    def __init__(self):
        self.agents = {
            'data_analyst': DataAnalystAgent(),
            'tree_specialist': TreeBasedSpecialist(),
            'linear_specialist': LinearModelsSpecialist(),
            'trainer': TrainerAgent(),
            'ensemble_architect': EnsembleArchitect(),
            'optimizer': OptimizerAgent()
        }
        
        if LANGGRAPH_AVAILABLE:
            self._build_graph()
        else:
            self.app = None
    
    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(EnsembleSystemState)
        
        # Add nodes for each agent
        workflow.add_node("analyze_data", self.agents['data_analyst'].analyze)
        workflow.add_node("tree_recommendations", self.agents['tree_specialist'].recommend_models)
        workflow.add_node("linear_recommendations", self.agents['linear_specialist'].recommend_models)
        workflow.add_node("train_models", self.agents['trainer'].train_models)
        workflow.add_node("design_ensemble", self.agents['ensemble_architect'].design_ensemble)
        workflow.add_node("optimize", self.agents['optimizer'].optimize_and_finalize)
        
        # Define flow
        workflow.set_entry_point("analyze_data")
        
        # Parallel specialist recommendations
        workflow.add_edge("analyze_data", "tree_recommendations")
        workflow.add_edge("analyze_data", "linear_recommendations")
        
        # Both specialists lead to training
        workflow.add_edge("tree_recommendations", "train_models")
        workflow.add_edge("linear_recommendations", "train_models")
        
        # Sequential flow after training
        workflow.add_edge("train_models", "design_ensemble")
        workflow.add_edge("design_ensemble", "optimize")
        workflow.add_edge("optimize", END)
        
        # Compile
        self.app = workflow.compile()
    
    def run(self, X, y, dataset_name="custom"):
        """Run the multi-agent system."""
        print("🤖 Multi-Agent Ensemble System Starting...")
        print("=" * 60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initial state
        initial_state = {
            'dataset_name': dataset_name,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(y_train)),
            'messages': [],
            'phase': 'analysis'
        }
        
        if self.app:
            # Run with LangGraph
            final_state = self.app.invoke(initial_state)
        else:
            # Run sequentially without LangGraph
            state = initial_state
            state = self.agents['data_analyst'].analyze(state)
            state = self.agents['tree_specialist'].recommend_models(state)
            state = self.agents['linear_specialist'].recommend_models(state)
            state = self.agents['trainer'].train_models(state)
            state = self.agents['ensemble_architect'].design_ensemble(state)
            state = self.agents['optimizer'].optimize_and_finalize(state)
            final_state = state
        
        # Generate report
        self._generate_report(final_state)
        
        return final_state
    
    def _generate_report(self, state):
        """Generate final system report."""
        print("\n" + "=" * 60)
        print("📊 MULTI-AGENT ENSEMBLE SYSTEM REPORT")
        print("=" * 60)
        
        print(f"\nDataset: {state['dataset_name']}")
        print(f"Samples: {state['n_samples']} train, {len(state['X_test'])} test")
        print(f"Features: {state['n_features']}")
        print(f"Classes: {state['n_classes']}")
        
        print("\n🏆 Model Performance:")
        print("-" * 40)
        for model_name, scores in state['model_scores'].items():
            print(f"{model_name:25} | Test Acc: {scores['test_accuracy']:.3f}")
        
        print(f"\n{'Ensemble (' + state['ensemble_strategy'] + ')':25} | "
              f"Test Acc: {state['ensemble_performance']['test_accuracy']:.3f} "
              f"(+{state['ensemble_performance']['improvement']:.3f})")
        
        print("\n📋 Agent Communications:")
        print("-" * 40)
        for msg in state['messages'][-5:]:  # Last 5 messages
            print(f"[{msg['agent']}] {msg['message']}")
        
        print("\n✅ Final Configuration:")
        print(f"Type: {state['best_configuration']['type']}")
        print(f"Best Test Accuracy: {state['best_configuration']['performance']['test_accuracy']:.3f}")
        
        print("\n" + "=" * 60)


def main():
    """Demonstrate the multi-agent ensemble system."""
    
    # Create sample dataset
    print("📊 Creating sample dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Create and run system
    system = MultiAgentEnsembleSystem()
    result = system.run(X, y, dataset_name="synthetic_classification")
    
    print("\n✅ Multi-Agent System Complete!")
    print(f"Final model type: {result['best_configuration']['type']}")
    print(f"Final accuracy: {result['best_configuration']['performance']['test_accuracy']:.3f}")
    
    # Try with a real dataset
    print("\n" + "=" * 60)
    print("🍷 Running on Wine Dataset...")
    wine = load_wine()
    result_wine = system.run(wine.data, wine.target, dataset_name="wine")
    
    print("\n🎉 All experiments complete!")


if __name__ == "__main__":
    main()
