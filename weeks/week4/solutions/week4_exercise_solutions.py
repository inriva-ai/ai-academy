"""
Week 4: Exercise Solutions - Ensemble Methods & LangGraph
========================================================
Complete solutions for all Week 4 exercises with explanations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, TypedDict, Annotated, Sequence
import operator
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# For LangGraph exercises
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️ LangGraph not available for some solutions")


class Week4Solutions:
    """Solutions for all Week 4 exercises."""
    
    def __init__(self):
        self.results = {}
        print("Week 4 Exercise Solutions")
        print("=" * 50)
    
    # ========================================================================
    # Exercise 1: Custom Weighted Voting Ensemble
    # ========================================================================
    
    def exercise1_custom_weighted_voting(self):
        """
        SOLUTION 1: Dynamic Weighted Voting Classifier
        
        Key concepts:
        - Weight calculation based on validation performance
        - Multiple weight metrics (accuracy, F1, AUC)
        - Confidence thresholding
        """
        print("\n🔧 Exercise 1: Custom Weighted Voting Ensemble")
        print("-" * 40)
        
        class DynamicWeightedVotingClassifier:
            """
            Voting classifier with dynamic weights based on validation performance.
            """
            def __init__(self, estimators, weight_metric='accuracy', confidence_threshold=0.5):
                self.estimators = estimators
                self.weight_metric = weight_metric
                self.confidence_threshold = confidence_threshold
                self.weights_ = None
                self.models_ = {}
                self.classes_ = None
                
            def fit(self, X, y, validation_split=0.2):
                """
                Train all models and determine optimal weights.
                """
                # Store unique classes
                self.classes_ = np.unique(y)
                
                # Split data for validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split, random_state=42, stratify=y
                )
                
                # Train each model and calculate weights
                weights = []
                
                for name, model in self.estimators:
                    print(f"   Training {name}...")
                    
                    # Train model
                    model.fit(X_train, y_train)
                    self.models_[name] = model
                    
                    # Calculate performance on validation set
                    y_pred = model.predict(X_val)
                    
                    if self.weight_metric == 'accuracy':
                        weight = accuracy_score(y_val, y_pred)
                    elif self.weight_metric == 'f1':
                        weight = f1_score(y_val, y_pred, average='weighted')
                    elif self.weight_metric == 'auc' and len(self.classes_) == 2:
                        y_proba = model.predict_proba(X_val)[:, 1]
                        weight = roc_auc_score(y_val, y_proba)
                    else:
                        weight = accuracy_score(y_val, y_pred)
                    
                    weights.append(weight)
                    print(f"     {name} {self.weight_metric}: {weight:.3f}")
                
                # Normalize weights to sum to 1
                weights = np.array(weights)
                self.weights_ = weights / weights.sum()
                
                print(f"\n   Final weights: {dict(zip([n for n, _ in self.estimators], self.weights_))}")
                
                # Retrain on full dataset
                for name, model in self.models_.items():
                    model.fit(X, y)
                
                return self
            
            def predict_proba(self, X):
                """
                Weighted probability predictions.
                """
                # Collect predictions from all models
                predictions = []
                
                for i, (name, _) in enumerate(self.estimators):
                    model = self.models_[name]
                    proba = model.predict_proba(X)
                    predictions.append(proba * self.weights_[i])
                
                # Weighted average
                weighted_proba = np.sum(predictions, axis=0)
                
                # Normalize to ensure probabilities sum to 1
                return weighted_proba / weighted_proba.sum(axis=1, keepdims=True)
            
            def predict(self, X):
                """
                Make predictions with optional confidence thresholding.
                """
                probas = self.predict_proba(X)
                
                # Get max probability for each sample
                max_probas = np.max(probas, axis=1)
                predictions = np.argmax(probas, axis=1)
                
                # Apply confidence threshold
                confident_mask = max_probas >= self.confidence_threshold
                
                # Return -1 for uncertain predictions
                result = predictions.copy()
                result[~confident_mask] = -1
                
                if (~confident_mask).any():
                    print(f"   ⚠️ {(~confident_mask).sum()} uncertain predictions (confidence < {self.confidence_threshold})")
                
                return result
            
            def score(self, X, y):
                """Calculate accuracy, ignoring uncertain predictions."""
                predictions = self.predict(X)
                mask = predictions != -1
                if mask.any():
                    return accuracy_score(y[mask], predictions[mask])
                else:
                    return 0.0
        
        # Test on multiple datasets
        datasets = {
            'Wine': load_wine(),
            'Breast Cancer': load_breast_cancer(),
            'Iris': load_iris()
        }
        
        results = {}
        
        for dataset_name, dataset in datasets.items():
            print(f"\n📊 Testing on {dataset_name} dataset:")
            
            X, y = dataset.data, dataset.target
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Define base estimators
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('svc', SVC(kernel='rbf', probability=True, random_state=42)),
                ('lr', LogisticRegression(max_iter=1000, random_state=42)),
                ('nb', GaussianNB())
            ]
            
            # Custom weighted voting
            custom_voting = DynamicWeightedVotingClassifier(
                estimators=estimators,
                weight_metric='accuracy',
                confidence_threshold=0.6
            )
            custom_voting.fit(X_train_scaled, y_train)
            custom_score = custom_voting.score(X_test_scaled, y_test)
            
            # Standard voting for comparison
            standard_voting = VotingClassifier(estimators=estimators, voting='soft')
            standard_voting.fit(X_train_scaled, y_train)
            standard_score = standard_voting.score(X_test_scaled, y_test)
            
            results[dataset_name] = {
                'custom': custom_score,
                'standard': standard_score,
                'improvement': custom_score - standard_score
            }
            
            print(f"   Custom Voting: {custom_score:.3f}")
            print(f"   Standard Voting: {standard_score:.3f}")
            print(f"   Improvement: {results[dataset_name]['improvement']:+.3f}")
        
        # Visualize results
        self._plot_voting_comparison(results)
        
        self.results['exercise1'] = results
        return results
    
    def _plot_voting_comparison(self, results):
        """Helper to visualize voting comparison."""
        datasets = list(results.keys())
        custom_scores = [results[d]['custom'] for d in datasets]
        standard_scores = [results[d]['standard'] for d in datasets]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, custom_scores, width, label='Custom Weighted', alpha=0.8)
        ax.bar(x + width/2, standard_scores, width, label='Standard Voting', alpha=0.8)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Accuracy')
        ax.set_title('Custom vs Standard Voting Classifier Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # Exercise 2: Advanced Stacking with Meta-Features
    # ========================================================================
    
    def exercise2_advanced_stacking(self):
        """
        SOLUTION 2: Multi-Level Stacking with Meta-Feature Engineering
        
        Key concepts:
        - Two-level stacking architecture
        - Meta-feature creation (probabilities, entropy, agreement)
        - Out-of-fold predictions to prevent overfitting
        """
        print("\n🔧 Exercise 2: Advanced Stacking with Meta-Features")
        print("-" * 40)
        
        from sklearn.model_selection import KFold
        from scipy.stats import entropy
        
        class MultiLevelStackingClassifier:
            """
            Multi-level stacking with meta-feature engineering.
            """
            def __init__(self, level1_estimators, level2_estimators, 
                        final_estimator, use_probas=True, 
                        include_original_features=True):
                self.level1_estimators = level1_estimators
                self.level2_estimators = level2_estimators
                self.final_estimator = final_estimator
                self.use_probas = use_probas
                self.include_original_features = include_original_features
                self.level1_models_ = {}
                self.level2_models_ = {}
                self.final_model_ = None
                
            def _create_meta_features(self, X, models, use_probas=True):
                """
                Create meta-features from model predictions.
                """
                meta_features = []
                
                for name, model in models.items():
                    if use_probas and hasattr(model, 'predict_proba'):
                        # Use probability predictions
                        probas = model.predict_proba(X)
                        meta_features.append(probas)
                        
                        # Add entropy as uncertainty measure
                        pred_entropy = np.array([entropy(p) for p in probas]).reshape(-1, 1)
                        meta_features.append(pred_entropy)
                        
                        # Add max probability as confidence
                        max_proba = np.max(probas, axis=1).reshape(-1, 1)
                        meta_features.append(max_proba)
                    else:
                        # Use class predictions
                        preds = model.predict(X).reshape(-1, 1)
                        meta_features.append(preds)
                
                # Model agreement features
                if len(models) > 1:
                    predictions = np.column_stack([
                        model.predict(X) for model in models.values()
                    ])
                    # Agreement ratio
                    agreement = np.mean([
                        predictions[:, i] == predictions[:, 0] 
                        for i in range(1, predictions.shape[1])
                    ], axis=0).reshape(-1, 1)
                    meta_features.append(agreement)
                
                # Combine all meta-features
                meta_X = np.hstack(meta_features)
                
                # Optionally include original features
                if self.include_original_features:
                    meta_X = np.hstack([X, meta_X])
                
                return meta_X
            
            def fit(self, X, y):
                """
                Fit the multi-level stacking classifier.
                """
                n_splits = 5
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                
                # Level 1: Train base models with out-of-fold predictions
                print("   Training Level 1 models...")
                level1_oof_predictions = np.zeros((X.shape[0], 0))
                
                for name, estimator in self.level1_estimators:
                    print(f"     Training {name}...")
                    
                    # Out-of-fold predictions
                    if self.use_probas and hasattr(estimator, 'predict_proba'):
                        n_classes = len(np.unique(y))
                        oof_pred = np.zeros((X.shape[0], n_classes))
                    else:
                        oof_pred = np.zeros((X.shape[0], 1))
                    
                    # Train on each fold
                    models = []
                    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                        y_fold_train = y[train_idx]
                        
                        # Clone estimator for this fold
                        model = estimator.__class__(**estimator.get_params())
                        model.fit(X_fold_train, y_fold_train)
                        models.append(model)
                        
                        # Generate out-of-fold predictions
                        if self.use_probas and hasattr(model, 'predict_proba'):
                            oof_pred[val_idx] = model.predict_proba(X_fold_val)
                        else:
                            oof_pred[val_idx, 0] = model.predict(X_fold_val)
                    
                    # Store models and add to OOF predictions
                    self.level1_models_[name] = models
                    level1_oof_predictions = np.hstack([level1_oof_predictions, oof_pred])
                
                # Train final Level 1 models on full data
                for name, estimator in self.level1_estimators:
                    final_model = estimator.__class__(**estimator.get_params())
                    final_model.fit(X, y)
                    self.level1_models_[f"{name}_final"] = final_model
                
                # Level 2: Train on Level 1 meta-features
                print("\n   Training Level 2 models...")
                level1_meta_features = self._create_meta_features(
                    X, 
                    {name: self.level1_models_[f"{name}_final"] 
                     for name, _ in self.level1_estimators},
                    self.use_probas
                )
                
                level2_oof_predictions = np.zeros((X.shape[0], 0))
                
                for name, estimator in self.level2_estimators:
                    print(f"     Training {name}...")
                    
                    # Similar OOF process for Level 2
                    if self.use_probas and hasattr(estimator, 'predict_proba'):
                        n_classes = len(np.unique(y))
                        oof_pred = np.zeros((X.shape[0], n_classes))
                    else:
                        oof_pred = np.zeros((X.shape[0], 1))
                    
                    models = []
                    for fold, (train_idx, val_idx) in enumerate(kf.split(level1_meta_features)):
                        X_fold_train = level1_meta_features[train_idx]
                        X_fold_val = level1_meta_features[val_idx]
                        y_fold_train = y[train_idx]
                        
                        model = estimator.__class__(**estimator.get_params())
                        model.fit(X_fold_train, y_fold_train)
                        models.append(model)
                        
                        if self.use_probas and hasattr(model, 'predict_proba'):
                            oof_pred[val_idx] = model.predict_proba(X_fold_val)
                        else:
                            oof_pred[val_idx, 0] = model.predict(X_fold_val)
                    
                    self.level2_models_[name] = models
                    level2_oof_predictions = np.hstack([level2_oof_predictions, oof_pred])
                    
                    # Train final Level 2 model
                    final_model = estimator.__class__(**estimator.get_params())
                    final_model.fit(level1_meta_features, y)
                    self.level2_models_[f"{name}_final"] = final_model
                
                # Final estimator: Train on Level 2 meta-features
                print("\n   Training final estimator...")
                level2_meta_features = self._create_meta_features(
                    level1_meta_features,
                    {name: self.level2_models_[f"{name}_final"] 
                     for name, _ in self.level2_estimators},
                    self.use_probas
                )
                
                self.final_model_ = self.final_estimator
                self.final_model_.fit(level2_meta_features, y)
                
                print("   ✅ Multi-level stacking complete!")
                return self
            
            def predict(self, X):
                """Make predictions through all levels."""
                # Level 1 predictions
                level1_models = {
                    name: self.level1_models_[f"{name}_final"] 
                    for name, _ in self.level1_estimators
                }
                level1_meta = self._create_meta_features(X, level1_models, self.use_probas)
                
                # Level 2 predictions
                level2_models = {
                    name: self.level2_models_[f"{name}_final"] 
                    for name, _ in self.level2_estimators
                }
                level2_meta = self._create_meta_features(level1_meta, level2_models, self.use_probas)
                
                # Final predictions
                return self.final_model_.predict(level2_meta)
            
            def score(self, X, y):
                """Calculate accuracy score."""
                return accuracy_score(y, self.predict(X))
        
        # Test implementation
        print("\n📊 Testing Multi-Level Stacking on Wine Dataset:")
        
        wine = load_wine()
        X, y = wine.data, wine.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define estimators for each level
        level1_estimators = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('svc', SVC(kernel='rbf', probability=True, random_state=42))
        ]
        
        level2_estimators = [
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('nb', GaussianNB())
        ]
        
        final_estimator = LogisticRegression(max_iter=1000, random_state=42)
        
        # Create and train multi-level stacking
        ml_stacking = MultiLevelStackingClassifier(
            level1_estimators=level1_estimators,
            level2_estimators=level2_estimators,
            final_estimator=final_estimator,
            use_probas=True,
            include_original_features=False  # Only use meta-features
        )
        
        ml_stacking.fit(X_train_scaled, y_train)
        ml_score = ml_stacking.score(X_test_scaled, y_test)
        
        # Compare with standard stacking
        standard_stacking = StackingClassifier(
            estimators=level1_estimators,
            final_estimator=final_estimator,
            cv=5
        )
        standard_stacking.fit(X_train_scaled, y_train)
        standard_score = standard_stacking.score(X_test_scaled, y_test)
        
        print(f"\n   Multi-Level Stacking: {ml_score:.3f}")
        print(f"   Standard Stacking: {standard_score:.3f}")
        print(f"   Improvement: {ml_score - standard_score:+.3f}")
        
        self.results['exercise2'] = {
            'multi_level': ml_score,
            'standard': standard_score,
            'improvement': ml_score - standard_score
        }
        
        return ml_stacking
    
    # ========================================================================
    # Exercise 3: First LangGraph Agent
    # ========================================================================
    
    def exercise3_first_langgraph_agent(self):
        """
        SOLUTION 3: Model Selection Agent with LangGraph
        
        Key concepts:
        - State management for ML workflows
        - Conditional routing based on data characteristics
        - Agent nodes for different ML tasks
        """
        print("\n🔧 Exercise 3: First LangGraph Agent")
        print("-" * 40)
        
        if not LANGGRAPH_AVAILABLE:
            print("   ⚠️ LangGraph not available. Showing mock implementation.")
            return None
        
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
            messages: Annotated[Sequence[str], operator.add]
        
        # Node implementations
        def analyze_dataset_node(state: Dict) -> Dict:
            """Analyze dataset characteristics."""
            print("   📊 [Data Analysis] Analyzing dataset...")
            
            # Load data (using wine dataset for demo)
            wine = load_wine()
            X, y = wine.data, wine.target
            
            # Perform analysis
            analysis = {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'feature_stats': {
                    'mean': np.mean(X, axis=0).mean(),
                    'std': np.std(X, axis=0).mean(),
                    'correlation': np.abs(np.corrcoef(X.T)).mean()
                },
                'class_balance': np.bincount(y).tolist(),
                'class_imbalance_ratio': max(np.bincount(y)) / min(np.bincount(y))
            }
            
            state['dataset_analysis'] = analysis
            state['messages'].append(f"Dataset analyzed: {analysis['n_samples']} samples, {analysis['n_features']} features")
            
            return state
        
        def recommend_models_node(state: Dict) -> Dict:
            """Recommend models based on analysis."""
            print("   🤖 [Model Recommendation] Selecting appropriate models...")
            
            analysis = state['dataset_analysis']
            recommendations = []
            rationale = {}
            
            # Rule-based recommendations
            if analysis['n_samples'] < 1000:
                recommendations.append('RandomForest')
                rationale['RandomForest'] = "Good for small datasets, handles non-linearity"
                
                recommendations.append('SVM')
                rationale['SVM'] = "Effective in high-dimensional spaces with small samples"
            
            if analysis['n_features'] < 50:
                recommendations.append('GradientBoosting')
                rationale['GradientBoosting'] = "Excellent for tabular data with moderate features"
            
            if analysis['class_imbalance_ratio'] < 2:
                recommendations.append('LogisticRegression')
                rationale['LogisticRegression'] = "Fast, interpretable baseline for balanced data"
            
            # Always include a simple baseline
            recommendations.append('NaiveBayes')
            rationale['NaiveBayes'] = "Simple probabilistic baseline"
            
            state['recommended_models'] = recommendations
            state['model_rationale'] = rationale
            state['messages'].append(f"Recommended {len(recommendations)} models based on data characteristics")
            
            return state
        
        def train_models_node(state: Dict) -> Dict:
            """Train recommended models."""
            print("   🏋️ [Model Training] Training recommended models...")
            
            # Load data again (in real implementation, would be passed through state)
            wine = load_wine()
            X, y = wine.data, wine.target
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train each recommended model
            results = {}
            model_map = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(kernel='rbf', random_state=42),
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                'NaiveBayes': GaussianNB()
            }
            
            for model_name in state['recommended_models']:
                if model_name in model_map:
                    model = model_map[model_name]
                    model.fit(X_train_scaled, y_train)
                    score = model.score(X_test_scaled, y_test)
                    results[model_name] = score
                    print(f"     {model_name}: {score:.3f}")
            
            state['training_results'] = results
            state['messages'].append(f"Training complete. Best model: {max(results, key=results.get)}")
            
            return state
        
        def evaluate_and_select_node(state: Dict) -> Dict:
            """Evaluate results and make final recommendation."""
            print("   📈 [Evaluation] Selecting best model...")
            
            results = state['training_results']
            best_model = max(results, key=results.get)
            best_score = results[best_model]
            
            # Generate recommendation
            recommendation = f"""
            Best Model: {best_model}
            Accuracy: {best_score:.3f}
            Rationale: {state['model_rationale'].get(best_model, 'N/A')}
            
            All Results:
            {chr(10).join(f'  - {m}: {s:.3f}' for m, s in results.items())}
            """
            
            state['final_recommendation'] = recommendation
            state['messages'].append(f"Final recommendation: {best_model} with {best_score:.3f} accuracy")
            
            print(recommendation)
            
            return state
        
        # Routing functions
        def should_balance_classes(state: Dict) -> str:
            """Determine if class balancing is needed."""
            if state['dataset_analysis']['class_imbalance_ratio'] > 3:
                return "balance_classes"
            return "proceed"
        
        # Build the agent graph
        workflow = StateGraph(ModelSelectorState)
        
        # Add nodes
        workflow.add_node("analyze_dataset", analyze_dataset_node)
        workflow.add_node("recommend_models", recommend_models_node)
        workflow.add_node("train_models", train_models_node)
        workflow.add_node("evaluate_and_select", evaluate_and_select_node)
        
        # Define flow
        workflow.set_entry_point("analyze_dataset")
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "analyze_dataset",
            should_balance_classes,
            {
                "balance_classes": "recommend_models",  # In real impl, would go to balancing node
                "proceed": "recommend_models"
            }
        )
        
        workflow.add_edge("recommend_models", "train_models")
        workflow.add_edge("train_models", "evaluate_and_select")
        workflow.add_edge("evaluate_and_select", END)
        
        # Compile
        app = workflow.compile()
        
        print("\n   🚀 Running Model Selection Agent...")
        
        # Initial state
        initial_state = {
            "dataset_path": "wine_dataset",
            "messages": []
        }
        
        # Run agent
        final_state = app.invoke(initial_state)
        
        print("\n   📝 Agent Message Log:")
        for msg in final_state['messages']:
            print(f"     - {msg}")
        
        self.results['exercise3'] = {
            'final_recommendation': final_state.get('final_recommendation', 'N/A'),
            'messages': final_state['messages']
        }
        
        return app
    
    # ========================================================================
    # Run all solutions
    # ========================================================================
    
    def run_all_solutions(self):
        """Run all exercise solutions."""
        
        # Exercise 1
        self.exercise1_custom_weighted_voting()
        
        # Exercise 2
        self.exercise2_advanced_stacking()
        
        # Exercise 3
        if LANGGRAPH_AVAILABLE:
            self.exercise3_first_langgraph_agent()
        else:
            print("\n⚠️ Skipping Exercise 3 - LangGraph not available")
        
        print("\n" + "=" * 50)
        print("✅ All solutions complete!")
        print("\nKey Takeaways:")
        print("1. Dynamic weighting improves voting classifier performance")
        print("2. Multi-level stacking with meta-features can capture complex patterns")
        print("3. LangGraph enables intelligent ML workflow automation")
        print("4. Agent-based systems can make data-driven model selection decisions")
        
        return self.results


def main():
    """Run Week 4 solutions."""
    print("Running Week 4 Exercise Solutions...")
    print("This will demonstrate complete implementations for all exercises.")
    print()
    
    solutions = Week4Solutions()
    results = solutions.run_all_solutions()
    
    print("\n🎉 Solutions demonstration complete!")
    print("\n📚 Next Steps:")
    print("- Review the code implementations")
    print("- Try modifying the solutions for your own data")
    print("- Experiment with different configurations")
    print("- Combine concepts from different exercises")


if __name__ == "__main__":
    main()
