"""
Week 4: Advanced Ensemble Pipeline with Metaflow
==============================================
This pipeline demonstrates parallel ensemble training with
comprehensive evaluation and model selection.

Run this pipeline:
    python ensemble_pipeline.py run
    
With parameters:
    python ensemble_pipeline.py run --n_estimators 200 --test_size 0.3
"""


from metaflow import FlowSpec, Parameter, step, current, card
from metaflow.cards import Image, Markdown

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')



class EnsemblePipeline(FlowSpec):
    """
    Advanced ensemble learning pipeline with parallel training,
    hyperparameter optimization, and comprehensive evaluation.
    """
    
    # Parameters
    dataset = Parameter('dataset',
                       help='Dataset to use: wine or breast_cancer',
                       default='wine')
    
    n_estimators = Parameter('n_estimators',
                            help='Number of estimators for ensemble methods',
                            type=int,
                            default=100)
    
    test_size = Parameter('test_size',
                         help='Test set size as fraction',
                         type=float,
                         default=0.2)
    
    optimize_hyperparams = Parameter('optimize_hyperparams',
                                   help='Whether to optimize hyperparameters',
                                   type=bool,
                                   default=True)
    
    @step
    def start(self):
        """
        Load dataset and prepare for training.
        """
        print(f"🚀 Starting Ensemble Pipeline")
        print(f"   Dataset: {self.dataset}")
        print(f"   N_estimators: {self.n_estimators}")
        print(f"   Test size: {self.test_size}")
        
        # Load dataset
        if self.dataset == 'wine':
            data = load_wine()
        elif self.dataset == 'breast_cancer':
            data = load_breast_cancer()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        
        self.X = data.data
        self.y = data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        
        # Dataset info
        self.dataset_info = {
            'name': self.dataset,
            'n_samples': self.X.shape[0],
            'n_features': self.X.shape[1],
            'n_classes': len(self.target_names),
            'feature_names': list(self.feature_names),
            'target_names': list(self.target_names)
        }
        
        print(f"📊 Dataset loaded: {self.dataset_info['n_samples']} samples, "
              f"{self.dataset_info['n_features']} features, "
              f"{self.dataset_info['n_classes']} classes")
        
        self.next(self.split_data)
    
    @step
    def split_data(self):
        """
        Split data into train/test sets and scale features.
        """
        print("📊 Splitting and scaling data...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=42, 
            stratify=self.y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Define ensemble strategies
        self.ensemble_strategies = [
            'voting_soft',
            'voting_hard', 
            'stacking',
            'random_forest',
            'extra_trees',
            'gradient_boosting',
            'adaboost',
            'bagging'
        ]
        
        print(f"✅ Data split: {len(self.X_train)} train, {len(self.X_test)} test")
        print(f"🎯 Training {len(self.ensemble_strategies)} ensemble strategies")
        
        self.next(self.train_ensembles, foreach='ensemble_strategies')
    
    @step
    def train_ensembles(self):
        """
        Train different ensemble strategies in parallel.
        """
        self.ensemble_name = self.input
        print(f"🏋️ Training {self.ensemble_name} ensemble...")
        
        # Create base models for voting/stacking
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('svc', SVC(kernel='rbf', probability=True, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('nb', GaussianNB())
        ]
        
        # Define ensemble based on strategy
        if self.ensemble_name == 'voting_soft':
            self.model = VotingClassifier(estimators=base_models, voting='soft')
            
        elif self.ensemble_name == 'voting_hard':
            self.model = VotingClassifier(estimators=base_models, voting='hard')
            
        elif self.ensemble_name == 'stacking':
            self.model = StackingClassifier(
                estimators=base_models[:3],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5
            )
            
        elif self.ensemble_name == 'random_forest':
            if self.optimize_hyperparams:
                # Hyperparameter optimization
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'max_features': ['sqrt', 'log2']
                }
                
                rf = RandomForestClassifier(random_state=42)
                grid_search = GridSearchCV(
                    rf, param_grid, cv=5, 
                    scoring='accuracy', verbose=0
                )
                grid_search.fit(self.X_train_scaled, self.y_train)
                self.model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_
            else:
                self.model = RandomForestClassifier(
                    n_estimators=self.n_estimators, 
                    random_state=42
                )
                self.best_params = None
                
        elif self.ensemble_name == 'extra_trees':
            self.model = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                random_state=42
            )
            self.best_params = None
            
        elif self.ensemble_name == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.best_params = None
            
        elif self.ensemble_name == 'adaboost':
            self.model = AdaBoostClassifier(
                n_estimators=self.n_estimators,
                learning_rate=1.0,
                random_state=42
            )
            self.best_params = None
            
        elif self.ensemble_name == 'bagging':
            self.model = BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=self.n_estimators,
                random_state=42
            )
            self.best_params = None
        
        # Train model
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        self.y_pred_train = self.model.predict(self.X_train_scaled)
        self.y_pred_test = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics
        self.train_accuracy = accuracy_score(self.y_train, self.y_pred_train)
        self.test_accuracy = accuracy_score(self.y_test, self.y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, self.X_train_scaled, self.y_train, cv=5
        )
        self.cv_mean = cv_scores.mean()
        self.cv_std = cv_scores.std()
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = self.model.feature_importances_
        else:
            self.feature_importances = None
        
        print(f"✅ {self.ensemble_name} trained:")
        print(f"   Train accuracy: {self.train_accuracy:.3f}")
        print(f"   Test accuracy: {self.test_accuracy:.3f}")
        print(f"   CV score: {self.cv_mean:.3f} ± {self.cv_std:.3f}")
        
        self.next(self.join)
    
    @step
    def join(self, inputs):
        """
        Combine results from all ensemble methods.
        """
        print("📊 Combining results from all ensembles...")
        
        # Collect all results
        self.ensemble_results = {}
        
        for input in inputs:
            self.ensemble_results[input.ensemble_name] = {
                'model': input.model,
                'train_accuracy': input.train_accuracy,
                'test_accuracy': input.test_accuracy,
                'cv_mean': input.cv_mean,
                'cv_std': input.cv_std,
                'best_params': getattr(input, 'best_params', None),
                'feature_importances': input.feature_importances,
                'y_pred_test': input.y_pred_test
            }
        
        # Copy other attributes from first input
        self.X_train_scaled = inputs[0].X_train_scaled
        self.X_test_scaled = inputs[0].X_test_scaled
        self.y_train = inputs[0].y_train
        self.y_test = inputs[0].y_test
        self.feature_names = inputs[0].feature_names
        self.target_names = inputs[0].target_names
        self.dataset_info = inputs[0].dataset_info
        
        # Find best model
        best_model = max(
            self.ensemble_results.items(),
            key=lambda x: x[1]['test_accuracy']
        )
        
        self.best_ensemble_name = best_model[0]
        self.best_ensemble_metrics = best_model[1]
        
        print(f"🏆 Best ensemble: {self.best_ensemble_name}")
        print(f"   Test accuracy: {self.best_ensemble_metrics['test_accuracy']:.3f}")
        
        self.next(self.evaluate_ensembles)
    
    @card
    @step
    def evaluate_ensembles(self):
        """
        Comprehensive evaluation of all ensemble methods.
        """
        print("📈 Evaluating all ensemble methods...")
        
        # Create comparison dataframe
        comparison_data = []
        
        for name, metrics in self.ensemble_results.items():
            comparison_data.append({
                'Ensemble': name,
                'Train Accuracy': metrics['train_accuracy'],
                'Test Accuracy': metrics['test_accuracy'],
                'CV Mean': metrics['cv_mean'],
                'CV Std': metrics['cv_std'],
                'Overfit Score': metrics['train_accuracy'] - metrics['test_accuracy']
            })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        self.comparison_df = self.comparison_df.sort_values(
            'Test Accuracy', ascending=False
        )
        
        print("\n📊 Ensemble Comparison:")
        print(self.comparison_df.to_string(index=False))
        
        # Generate visualizations
        current.card.append(Image.from_matplotlib(self._create_performance_plot(), label='Ensemble Performance'))
        current.card.append(Image.from_matplotlib(self._create_feature_importance_plot(), label=f'Feature Importance - {name}'))
        current.card.append(Image.from_matplotlib(self._create_confusion_matrix(), label=f'Confusion Matrix - {self.best_ensemble_name}'))

        self.next(self.create_meta_ensemble)
    
    @step
    def create_meta_ensemble(self):
        """
        Create a meta-ensemble using the best performing models.
        """
        print("🎯 Creating meta-ensemble from top performers...")
        
        # Select top 3 models
        top_models = self.comparison_df.head(3)['Ensemble'].tolist()
        
        # Create meta-ensemble
        meta_estimators = []
        for name in top_models:
            model = self.ensemble_results[name]['model']
            meta_estimators.append((name, model))
        
        self.meta_ensemble = VotingClassifier(
            estimators=meta_estimators,
            voting='hard'
        )
        
        # Train meta-ensemble
        self.meta_ensemble.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate
        meta_pred = self.meta_ensemble.predict(self.X_test_scaled)
        self.meta_accuracy = accuracy_score(self.y_test, meta_pred)
        
        print(f"✅ Meta-ensemble created from: {', '.join(top_models)}")
        print(f"   Meta-ensemble accuracy: {self.meta_accuracy:.3f}")
        
        # Compare with best single model
        improvement = self.meta_accuracy - self.best_ensemble_metrics['test_accuracy']
        print(f"   Improvement over best single: {improvement:+.3f}")
        
        self.next(self.generate_report)
    
    @card
    @step
    def generate_report(self):
        """
        Generate comprehensive report with visualizations.
        """
        print("📝 Generating final report...")
        
        # Create report card
        current.card.append(Markdown(f"# Ensemble Learning Pipeline Report"))
        current.card.append(Markdown(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        )
        current.card.append(Markdown(f"**Dataset**: {self.dataset_info['name']}"))
        
        # Dataset summary
        current.card.append(Markdown("## Dataset Summary"))
        current.card.append(Markdown(f"- Samples: {self.dataset_info['n_samples']}"))
        current.card.append(Markdown(f"- Features: {self.dataset_info['n_features']}"))
        current.card.append(Markdown(f"- Classes: {self.dataset_info['n_classes']}"))
        
        # Model comparison
        current.card.append(Markdown("## Ensemble Comparison"))
        current.card.append(Markdown(self.comparison_df.to_html(index=False)))
        
        # Best model details
        current.card.append(Markdown(f"## Best Ensemble: {self.best_ensemble_name}"))
        current.card.append(Markdown(f"- Test Accuracy: {self.best_ensemble_metrics['test_accuracy']:.3f}"))
        current.card.append(Markdown(f"- CV Score: {self.best_ensemble_metrics['cv_mean']:.3f} ± {self.best_ensemble_metrics['cv_std']:.3f}"))
        
        if self.best_ensemble_metrics['best_params']:
            current.card.append(Markdown("### Optimized Hyperparameters"))
            for param, value in self.best_ensemble_metrics['best_params'].items():
                current.card.append(Markdown(f"- {param}: {value}"))
        
        # Meta-ensemble results
        current.card.append(Markdown("## Meta-Ensemble Results"))
        current.card.append(Markdown(f"- Accuracy: {self.meta_accuracy:.3f}"))
        current.card.append(Markdown(f"- Improvement: {self.meta_accuracy - self.best_ensemble_metrics['test_accuracy']:+.3f}"))
        
        # Save models
        self.saved_models = {
            'best_single': self.ensemble_results[self.best_ensemble_name]['model'],
            'meta_ensemble': self.meta_ensemble,
            'all_models': {name: res['model'] for name, res in self.ensemble_results.items()}
        }
        
        print("✅ Report generated successfully!")
        print(f"🏆 Best ensemble: {self.best_ensemble_name} ({self.best_ensemble_metrics['test_accuracy']:.3f})")
        print(f"🎯 Meta-ensemble: {self.meta_accuracy:.3f}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Pipeline completion.
        """
        print("\n✅ Ensemble Pipeline Complete!")
        print("=" * 50)
        print(f"Best Single Model: {self.best_ensemble_name}")
        print(f"Best Accuracy: {self.best_ensemble_metrics['test_accuracy']:.3f}")
        print(f"Meta-Ensemble Accuracy: {self.meta_accuracy:.3f}")
        # print("\n📊 View detailed report with:")
        # print(f"   python ensemble_pipeline.py card view {current.run_id}")
    
    def _create_performance_plot(self):
        """Helper to create performance comparison plot."""
        fig = plt.figure(figsize=(12, 6))
        
        # Prepare data
        ensemble_names = list(self.ensemble_results.keys())
        train_accs = [self.ensemble_results[n]['train_accuracy'] for n in ensemble_names]
        test_accs = [self.ensemble_results[n]['test_accuracy'] for n in ensemble_names]
        
        # Create plot
        x = np.arange(len(ensemble_names))
        width = 0.35
        
        plt.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
        plt.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
        
        plt.xlabel('Ensemble Method')
        plt.ylabel('Accuracy')
        plt.title('Ensemble Performance Comparison')
        plt.xticks(x, ensemble_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def _create_feature_importance_plot(self):
        """Helper to create feature importance plot."""
        # Get models with feature importance
        models_with_importance = {
            name: res for name, res in self.ensemble_results.items()
            if res['feature_importances'] is not None
        }
        
        if models_with_importance:
            # Use best model with feature importance
            best_with_importance = max(
                models_with_importance.items(),
                key=lambda x: x[1]['test_accuracy']
            )
            
            name = best_with_importance[0]
            importances = best_with_importance[1]['feature_importances']
            
            # Create plot
            fig = plt.figure(figsize=(10, 8))
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features
            
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), 
                      [self.feature_names[i] for i in indices])
            plt.xlabel('Importance')
            plt.title(f'Feature Importance - {name}')
            plt.tight_layout()
            
            return fig

    def _create_confusion_matrix(self):
        """Helper to create confusion matrix for best model."""
        y_pred = self.ensemble_results[self.best_ensemble_name]['y_pred_test']
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.target_names,
                    yticklabels=self.target_names)
        plt.title(f'Confusion Matrix - {self.best_ensemble_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        return fig


if __name__ == '__main__':
    EnsemblePipeline()
