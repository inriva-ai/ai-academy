"""
Week 4: Hybrid LangGraph + Metaflow Pipeline
===========================================
This advanced example shows how to integrate LangGraph agents
within Metaflow pipelines for intelligent ML orchestration.

Run this pipeline:
    python hybrid_langgraph_metaflow.py run
"""

from metaflow import FlowSpec, step, Parameter, current, card
from metaflow.cards import Markdown
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime

# LangGraph imports (with fallback)
try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Annotated, Sequence, Dict, Any, List
    import operator
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    print("⚠️ LangGraph not installed. Some features will be limited.")
    print(f"   Error: {e}")
    LANGGRAPH_AVAILABLE = False


class HybridMLPipeline(FlowSpec):
    """
    A hybrid pipeline that uses LangGraph agents within Metaflow steps
    for intelligent decision-making throughout the ML workflow.
    """
    
    # Parameters
    dataset = Parameter('dataset',
                       help='Dataset to use: wine or synthetic',
                       default='wine')
    
    use_agents = Parameter('use_agents',
                          help='Whether to use LangGraph agents',
                          type=bool,
                          default=True)
    
    @step
    def start(self):
        """
        Initialize pipeline and load dataset.
        """
        print(f"🚀 Starting Hybrid ML Pipeline")
        print(f"   Dataset: {self.dataset}")
        print(f"   Using agents: {self.use_agents}")
        
        # Load dataset
        if self.dataset == 'wine':
            data = load_wine()
            self.X = data.data
            self.y = data.target
            self.feature_names = data.feature_names
            self.target_names = data.target_names
        else:
            # Synthetic dataset
            self.X, self.y = make_classification(
                n_samples=1000,
                n_features=20,
                n_informative=15,
                n_classes=3,
                random_state=42
            )
            self.feature_names = [f"feature_{i}" for i in range(20)]
            self.target_names = ["class_0", "class_1", "class_2"]
        
        print(f"📊 Dataset loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        
        self.next(self.analyze_data)
    
    @step
    def analyze_data(self):
        """
        Use a LangGraph agent to analyze data and make preprocessing decisions.
        """
        print("\n📊 Data Analysis Step")
        
        if self.use_agents and LANGGRAPH_AVAILABLE:
            # Use LangGraph agent for analysis
            analysis_agent = self._create_analysis_agent()
            
            initial_state = {
                'X': self.X,
                'y': self.y,
                'feature_names': self.feature_names,
                'messages': []
            }
            
            # Run agent
            result = analysis_agent.invoke(initial_state)
            
            self.data_insights = result['insights']
            self.preprocessing_plan = result['preprocessing_plan']
            self.agent_messages = result['messages']
            
            print("\n🤖 Agent Analysis Complete:")
            print(f"   Insights: {self.data_insights}")
            print(f"   Preprocessing plan: {self.preprocessing_plan}")
            
        else:
            # Fallback to simple analysis
            self.data_insights = {
                'n_samples': self.X.shape[0],
                'n_features': self.X.shape[1],
                'n_classes': len(np.unique(self.y)),
                'needs_scaling': True
            }
            self.preprocessing_plan = ['scaling', 'train_test_split']
            self.agent_messages = ["Simple analysis completed"]
        
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        """
        Preprocess data based on agent recommendations.
        """
        print("\n🔧 Preprocessing Step")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Apply preprocessing based on plan
        if 'scaling' in self.preprocessing_plan:
            print("   Applying StandardScaler...")
            scaler = StandardScaler()
            self.X_train_processed = scaler.fit_transform(self.X_train)
            self.X_test_processed = scaler.transform(self.X_test)
        else:
            self.X_train_processed = self.X_train
            self.X_test_processed = self.X_test
        
        print(f"✅ Preprocessing complete: {len(self.preprocessing_plan)} steps applied")
        
        self.next(self.model_selection)
    
    @step
    def model_selection(self):
        """
        Use a LangGraph agent to select and configure models.
        """
        print("\n🤖 Model Selection Step")
        
        if self.use_agents and LANGGRAPH_AVAILABLE:
            # Create model selection agent
            selection_agent = self._create_model_selection_agent()
            
            initial_state = {
                'data_insights': self.data_insights,
                'n_samples_train': len(self.X_train),
                'n_features': self.X_train.shape[1],
                'messages': []
            }
            
            result = selection_agent.invoke(initial_state)
            
            self.selected_models = result['selected_models']
            self.model_configs = result['model_configs']
            self.agent_messages += result['messages']
            
            print("\n🤖 Agent Selected Models:")
            for model_info in self.selected_models:
                print(f"   - {model_info['name']}: {model_info['reason']}")
        
        else:
            # Fallback to default models
            self.selected_models = [
                {'name': 'RandomForest', 'type': 'ensemble', 'reason': 'Good default choice'},
                {'name': 'LogisticRegression', 'type': 'linear', 'reason': 'Fast baseline'}
            ]
            self.model_configs = {
                'RandomForest': {'n_estimators': 100, 'random_state': 42},
                'LogisticRegression': {'max_iter': 1000, 'random_state': 42}
            }
        
        self.next(self.train_models)
    
    @step
    def train_models(self):
        """
        Train selected models with agent-recommended configurations.
        """
        print("\n🏋️ Model Training Step")
        
        self.trained_models = {}
        self.model_scores = {}
        
        # Model mapping
        model_classes = {
            'RandomForest': RandomForestClassifier,
            'GradientBoosting': GradientBoostingClassifier,
            'LogisticRegression': LogisticRegression
        }
        
        # Train each selected model
        for model_info in self.selected_models:
            model_name = model_info['name']
            
            if model_name in model_classes:
                print(f"   Training {model_name}...")
                
                # Get configuration
                config = self.model_configs.get(model_name, {})
                
                # Create and train model
                model = model_classes[model_name](**config)
                model.fit(self.X_train_processed, self.y_train)
                
                # Evaluate
                train_score = model.score(self.X_train_processed, self.y_train)
                test_score = model.score(self.X_test_processed, self.y_test)
                
                self.trained_models[model_name] = model
                self.model_scores[model_name] = {
                    'train': train_score,
                    'test': test_score,
                    'overfit': train_score - test_score
                }
                
                print(f"     ✓ {model_name}: Train={train_score:.3f}, Test={test_score:.3f}")
        
        self.next(self.ensemble_optimization)
    
    @step
    def ensemble_optimization(self):
        """
        Use a LangGraph agent to design optimal ensemble strategy.
        """
        print("\n🎯 Ensemble Optimization Step")
        
        if self.use_agents and LANGGRAPH_AVAILABLE and len(self.trained_models) > 1:
            # Create ensemble optimization agent
            ensemble_agent = self._create_ensemble_agent()
            
            initial_state = {
                'model_scores': self.model_scores,
                'model_types': {m['name']: m['type'] for m in self.selected_models},
                'messages': []
            }
            
            result = ensemble_agent.invoke(initial_state)
            
            self.ensemble_strategy = result['ensemble_strategy']
            self.ensemble_rationale = result['rationale']
            self.agent_messages += result['messages']
            
            print(f"\n🤖 Agent Ensemble Strategy: {self.ensemble_strategy}")
            print(f"   Rationale: {self.ensemble_rationale}")
            
            # Implement ensemble
            if self.ensemble_strategy == 'voting':
                from sklearn.ensemble import VotingClassifier
                estimators = [(name, model) for name, model in self.trained_models.items()]
                self.ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
                self.ensemble_model.fit(self.X_train_processed, self.y_train)
                
                ensemble_score = self.ensemble_model.score(self.X_test_processed, self.y_test)
                print(f"   Ensemble Test Score: {ensemble_score:.3f}")
            else:
                # Use best single model
                best_model = max(self.model_scores.items(), key=lambda x: x[1]['test'])
                self.ensemble_model = self.trained_models[best_model[0]]
                ensemble_score = best_model[1]['test']
        
        else:
            # No ensemble needed
            best_model = max(self.model_scores.items(), key=lambda x: x[1]['test'])
            self.ensemble_model = self.trained_models[best_model[0]]
            self.ensemble_strategy = "single_best"
            self.ensemble_rationale = "Using best performing model"
        
        self.next(self.generate_report)
    
    @card
    @step
    def generate_report(self):
        """
        Generate comprehensive report with agent insights.
        """
        print("\n📝 Report Generation Step")
        
        # Create report card
        current.card.append(Markdown("# Hybrid ML Pipeline Report"))
        current.card.append(Markdown(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}"))
        current.card.append(Markdown(f"**Dataset**: {self.dataset}"))
        
        # Data insights
        current.card.append(Markdown("\n## Data Analysis"))
        for key, value in self.data_insights.items():
            current.card.append(Markdown(f"- {key}: {value}"))
        
        # Model performance
        current.card.append(Markdown("\n## Model Performance"))
        results_df = pd.DataFrame(self.model_scores).T
        current.card.append(Markdown(results_df.to_html()))
        
        # Ensemble strategy
        current.card.append(Markdown("\n## Ensemble Strategy"))
        current.card.append(Markdown(f"**Strategy**: {self.ensemble_strategy}"))
        current.card.append(Markdown(f"**Rationale**: {self.ensemble_rationale}"))
        
        # Agent messages
        if hasattr(self, 'agent_messages'):
            current.card.append(Markdown("\n## Agent Decision Log"))
            for msg in self.agent_messages[-20:]:  # Last 20 messages
                current.card.append(Markdown(f"- {msg}"))
        
        # Final recommendations
        self._generate_recommendations()
        
        print("✅ Report generated!")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Pipeline completion.
        """
        print("\n✅ Hybrid Pipeline Complete!")
        print("=" * 50)
        
        # Summary
        best_model = max(self.model_scores.items(), key=lambda x: x[1]['test'])
        print(f"Best Single Model: {best_model[0]} ({best_model[1]['test']:.3f})")
        print(f"Ensemble Strategy: {self.ensemble_strategy}")
        
        if self.use_agents:
            print("\n🤖 Agent Contributions:")
            print("- Data analysis and preprocessing recommendations")
            print("- Intelligent model selection based on data characteristics")
            print("- Ensemble strategy optimization")
        
        # print(f"\n📊 View detailed report with:")
        # print(f"   python hybrid_langgraph_metaflow.py card view {current.run_id}")
    
    # ========================================================================
    # LangGraph Agent Definitions
    # ========================================================================
    
    def _create_analysis_agent(self):
        """Create data analysis agent."""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        class AnalysisState(TypedDict):
            X: Any
            y: Any
            feature_names: List[str]
            insights: Dict[str, Any]
            preprocessing_plan: List[str]
            messages: Annotated[Sequence[str], operator.add]
        
        def analyze_node(state):
            X, y = state['X'], state['y']
            
            insights = {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'class_balance': np.bincount(y).tolist(),
                'feature_variance': np.var(X, axis=0).mean(),
                'needs_scaling': np.std(X, axis=0).max() > 10
            }
            # Return only changes, assign messages
            return {
                'insights': insights,
                'messages': ["Data analysis complete"]
            }
        
        def plan_preprocessing_node(state):
            insights = state['insights']
            plan = []
            
            if insights['needs_scaling']:
                plan.append('scaling')
            
            plan.append('train_test_split')
            
            # Check for imbalance
            class_counts = insights['class_balance']
            if max(class_counts) / min(class_counts) > 2:
                plan.append('balance_classes')
            return {
                'preprocessing_plan': plan,
                'messages': [f"Preprocessing plan: {', '.join(plan)}"]
            }
        
        # Build graph
        workflow = StateGraph(AnalysisState)
        workflow.add_node("analyze", analyze_node)
        workflow.add_node("plan_preprocessing", plan_preprocessing_node)
        
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "plan_preprocessing")
        workflow.add_edge("plan_preprocessing", END)
        
        return workflow.compile()
    
    def _create_model_selection_agent(self):
        """Create model selection agent."""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        class SelectionState(TypedDict):
            data_insights: Dict[str, Any]
            n_samples_train: int
            n_features: int
            selected_models: List[Dict[str, str]]
            model_configs: Dict[str, Dict[str, Any]]
            messages: Annotated[Sequence[str], operator.add]
        
        def select_models_node(state):
            n_samples = state['n_samples_train']
            n_features = state['n_features']
            
            selected = []
            configs = {}
            
            # Always include Random Forest
            selected.append({
                'name': 'RandomForest',
                'type': 'ensemble',
                'reason': 'Robust to overfitting, handles non-linearity'
            })
            configs['RandomForest'] = {
                'n_estimators': 100 if n_samples > 500 else 50,
                'max_depth': None if n_features < 50 else 20,
                'random_state': 42
            }
            
            # Add Gradient Boosting for larger datasets
            if n_samples > 500:
                selected.append({
                    'name': 'GradientBoosting',
                    'type': 'ensemble',
                    'reason': 'High accuracy potential for sufficient data'
                })
                configs['GradientBoosting'] = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42
                }
            
            # Always include a linear baseline
            selected.append({
                'name': 'LogisticRegression',
                'type': 'linear',
                'reason': 'Fast, interpretable baseline'
            })
            configs['LogisticRegression'] = {
                'max_iter': 1000,
                'random_state': 42
            }
            return {
                'selected_models': selected,
                'model_configs': configs,
                'messages': [f"Selected {len(selected)} models based on data characteristics"]
            }
        
        # Build graph
        workflow = StateGraph(SelectionState)
        workflow.add_node("select_models", select_models_node)
        workflow.set_entry_point("select_models")
        workflow.add_edge("select_models", END)
        
        return workflow.compile()
    
    def _create_ensemble_agent(self):
        """Create ensemble optimization agent."""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        class EnsembleState(TypedDict):
            model_scores: Dict[str, Dict[str, float]]
            model_types: Dict[str, str]
            ensemble_strategy: str
            rationale: str
            messages: Annotated[Sequence[str], operator.add]
        
        def analyze_diversity_node(state):
            scores = state['model_scores']
            
            # Calculate performance variance
            test_scores = [s['test'] for s in scores.values()]
            performance_variance = np.var(test_scores)
            
            # Check if models are diverse enough
            has_ensemble = any(t == 'ensemble' for t in state['model_types'].values())
            has_linear = any(t == 'linear' for t in state['model_types'].values())
            return {
                'diversity_score': performance_variance,
                'has_diverse_types': has_ensemble and has_linear,
                'messages': [f"Model diversity score: {performance_variance:.3f}"]
            }
        
        def select_strategy_node(state):
            diversity = state.get('diversity_score', 0)
            n_models = len(state['model_scores'])
            
            # Decision logic
            if n_models >= 3 and diversity > 0.001 and state.get('has_diverse_types', False):
                strategy = 'voting'
                rationale = "Multiple diverse models with varying strengths - voting ensemble recommended"
            elif n_models >= 2 and diversity > 0.0005:
                strategy = 'weighted_voting'
                rationale = "Some model diversity - weighted voting based on performance"
            else:
                strategy = 'single_best'
                rationale = "Models too similar or too few - using best single model"
            return {
                'ensemble_strategy': strategy,
                'rationale': rationale,
                'messages': [f"Selected strategy: {strategy}"]
            }
        
        # Build graph
        workflow = StateGraph(EnsembleState)
        workflow.add_node("analyze_diversity", analyze_diversity_node)
        workflow.add_node("select_strategy", select_strategy_node)
        
        workflow.set_entry_point("analyze_diversity")
        workflow.add_edge("analyze_diversity", "select_strategy")
        workflow.add_edge("select_strategy", END)
        
        return workflow.compile()
    
    def _generate_recommendations(self):
        """Generate actionable recommendations."""
        recommendations = []
        
        # Based on model performance
        best_score = max(s['test'] for s in self.model_scores.values())
        if best_score < 0.8:
            recommendations.append("Consider collecting more training data")
            recommendations.append("Try advanced feature engineering techniques")
        
        # Based on overfitting
        max_overfit = max(s['overfit'] for s in self.model_scores.values())
        if max_overfit > 0.1:
            recommendations.append("Implement regularization to reduce overfitting")
            recommendations.append("Consider using cross-validation for hyperparameter tuning")
        
        # Based on ensemble performance
        if self.ensemble_strategy == 'single_best':
            recommendations.append("Try more diverse model types for ensemble benefits")
        
        if recommendations:
            current.card.append(Markdown("\n## Recommendations"))
            for rec in recommendations:
                current.card.append(Markdown(f"- {rec}"))


if __name__ == '__main__':
    HybridMLPipeline()
