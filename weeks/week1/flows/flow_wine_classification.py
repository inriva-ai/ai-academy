"""
INRIVA AI Academy - Complete Wine Classification Pipeline
=========================================================

This is the complete ML pipeline from our workshop.
Demonstrates end-to-end MLOps with Metaflow.

Usage:
    python wine_classification_flow.py run
    python wine_classification_flow.py run --test_size 0.3
    python wine_classification_flow.py show
"""

from metaflow import FlowSpec, step, Parameter
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import json

class WineClassificationFlow(FlowSpec):
    """
    Complete ML pipeline for wine classification
    
    This flow demonstrates:
    - Parameter management
    - Data preprocessing
    - Multiple model training
    - Model comparison
    - Comprehensive evaluation
    - Production-ready structure
    """
    
    # Parameters - can be changed when running
    test_size = Parameter('test_size', 
                         help='Test set size ratio (0.1-0.5)',
                         default=0.2,
                         type=float)
    
    random_state = Parameter('random_state',
                           help='Random state for reproducibility', 
                           default=42,
                           type=int)
    
    models_to_compare = Parameter('models',
                                help='Comma-separated list of models to compare',
                                default='random_forest,logistic_regression,svm')
    
    @step
    def start(self):
        """
        Load and initial exploration of wine data
        """
        print("🍷 Starting Wine Classification Pipeline")
        print("=" * 50)
        print(f"Parameters:")
        print(f"   Test size: {self.test_size}")
        print(f"   Random state: {self.random_state}")
        print(f"   Models to compare: {self.models_to_compare}")
        
        # Load wine dataset
        wine_data = load_wine()
        
        # Store as artifacts
        self.feature_names = wine_data.feature_names.tolist()
        self.target_names = wine_data.target_names.tolist()
        self.X = wine_data.data
        self.y = wine_data.target
        
        # Basic data exploration
        self.data_info = {
            'n_samples': self.X.shape[0],
            'n_features': self.X.shape[1],
            'n_classes': len(np.unique(self.y)),
            'class_distribution': np.bincount(self.y).tolist(),
            'feature_ranges': {
                'min': self.X.min(axis=0).tolist(),
                'max': self.X.max(axis=0).tolist(),
                'mean': self.X.mean(axis=0).tolist()
            }
        }
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Samples: {self.data_info['n_samples']}")
        print(f"   Features: {self.data_info['n_features']}")
        print(f"   Classes: {self.data_info['n_classes']} ({self.target_names})")
        print(f"   Class distribution: {self.data_info['class_distribution']}")
        
        # Validate parameters
        if not (0.1 <= self.test_size <= 0.5):
            raise ValueError(f"test_size must be between 0.1 and 0.5, got {self.test_size}")
        
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        """
        Split and preprocess the data with comprehensive validation
        """
        print("\n🔧 Preprocessing pipeline...")
        
        # Split the data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.y
        )
        
        # Validate split maintains class distribution
        train_dist = np.bincount(self.y_train) / len(self.y_train)
        test_dist = np.bincount(self.y_test) / len(self.y_test)
        
        print(f"   Train set: {len(self.X_train)} samples")
        print(f"   Test set: {len(self.X_test)} samples")
        print(f"   Train distribution: {train_dist.round(3)}")
        print(f"   Test distribution: {test_dist.round(3)}")
        
        # Feature scaling with StandardScaler
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Store preprocessing info
        self.preprocessing_info = {
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'train_class_distribution': train_dist.tolist(),
            'test_class_distribution': test_dist.tolist()
        }
        
        print(f"   ✅ Features scaled using StandardScaler")
        print(f"   ✅ Preprocessing info stored")
        
        self.next(self.train_models)
    
    @step
    def train_models(self):
        """
        Train multiple models and compare performance
        """
        print("\n🎯 Training and comparing models...")
        
        # Parse model list
        model_names = [name.strip() for name in self.models_to_compare.split(',')]
        
        # Define available models
        available_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                multi_class='ovr'
            ),
            'svm': SVC(
                random_state=self.random_state,
                probability=True,
                kernel='rbf'
            )
        }
        
        # Validate requested models
        invalid_models = set(model_names) - set(available_models.keys())
        if invalid_models:
            raise ValueError(f"Invalid models requested: {invalid_models}")
        
        # Train and evaluate each model
        self.model_results = {}
        
        for name in model_names:
            print(f"\n   Training {name}...")
            
            model = available_models[name]
            
            # Train
            model.fit(self.X_train_scaled, self.y_train)
            
            # Evaluate on both train and test sets
            train_accuracy = model.score(self.X_train_scaled, self.y_train)
            test_accuracy = model.score(self.X_test_scaled, self.y_test)
            
            # Get predictions for detailed analysis
            train_pred = model.predict(self.X_train_scaled)
            test_pred = model.predict(self.X_test_scaled)
            
            # Store comprehensive results
            self.model_results[name] = {
                'model': model,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_predictions': train_pred,
                'test_predictions': test_pred,
                'overfitting': train_accuracy - test_accuracy
            }
            
            print(f"     Train accuracy: {train_accuracy:.4f}")
            print(f"     Test accuracy: {test_accuracy:.4f}")
            print(f"     Overfitting gap: {train_accuracy - test_accuracy:.4f}")
        
        print(f"\n   ✅ Trained {len(model_names)} models successfully")
        
        self.next(self.evaluate_best_model)
    
    @step
    def evaluate_best_model(self):
        """
        Select and thoroughly evaluate the best performing model
        """
        print("\n📊 Evaluating best model...")
        
        # Find best model by test accuracy
        best_model_name = max(self.model_results.keys(), 
                            key=lambda x: self.model_results[x]['test_accuracy'])
        
        self.best_model_name = best_model_name
        self.best_model = self.model_results[best_model_name]['model']
        
        print(f"   🏆 Best model: {best_model_name}")
        
        # Generate predictions
        self.y_pred = self.model_results[best_model_name]['test_predictions']
        
        # Detailed evaluation
        self.final_accuracy = accuracy_score(self.y_test, self.y_pred)
        self.classification_report = classification_report(
            self.y_test, self.y_pred, 
            target_names=self.target_names,
            output_dict=True
        )
        
        # Confusion matrix
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)
        
        print(f"   Final accuracy: {self.final_accuracy:.4f}")
        print(f"   Macro avg F1: {self.classification_report['macro avg']['f1-score']:.4f}")
        print(f"   Weighted avg F1: {self.classification_report['weighted avg']['f1-score']:.4f}")
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names, 
                self.best_model.feature_importances_
            ))
            
            # Top 5 features
            top_features = sorted(self.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            print(f"\n   🔍 Top 5 important features:")
            for feature, importance in top_features:
                print(f"      {feature}: {importance:.4f}")
        else:
            self.feature_importance = None
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Finalize pipeline with comprehensive summary and recommendations
        """
        print("\n🎉 Wine Classification Pipeline Complete!")
        print("=" * 50)
        
        # Create comprehensive summary
        self.pipeline_summary = {
            'dataset_info': self.data_info,
            'preprocessing_info': self.preprocessing_info,
            'models_compared': list(self.model_results.keys()),
            'model_performance': {
                name: {
                    'test_accuracy': results['test_accuracy'],
                    'overfitting_gap': results['overfitting']
                }
                for name, results in self.model_results.items()
            },
            'best_model': {
                'name': self.best_model_name,
                'test_accuracy': self.final_accuracy,
                'classification_metrics': self.classification_report
            },
            'feature_importance': self.feature_importance,
            'parameters_used': {
                'test_size': self.test_size,
                'random_state': self.random_state,
                'models_compared': self.models_to_compare
            }
        }
        
        print("📋 Final Results Summary:")
        print(f"   🏆 Best Model: {self.best_model_name}")
        print(f"   🎯 Accuracy: {self.final_accuracy:.4f}")
        print(f"   📊 Models Compared: {len(self.model_results)}")
        print(f"   🔧 Features Used: {len(self.feature_names)}")
        
        print(f"\n📈 Model Comparison:")
        for name, metrics in self.pipeline_summary['model_performance'].items():
            marker = "🏆" if name == self.best_model_name else "  "
            print(f"   {marker} {name}: {metrics['test_accuracy']:.4f} (gap: {metrics['overfitting_gap']:.4f})")
        
        print(f"\n💡 Recommendations:")
        
        # Performance recommendations
        if self.final_accuracy > 0.95:
            print("   ✅ Excellent performance! Model ready for production.")
        elif self.final_accuracy > 0.9:
            print("   ✅ Good performance. Consider hyperparameter tuning for improvement.")
        else:
            print("   ⚠️ Consider feature engineering or ensemble methods.")
        
        # Overfitting analysis
        overfitting_gap = self.model_results[self.best_model_name]['overfitting']
        if overfitting_gap > 0.1:
            print("   ⚠️ Significant overfitting detected. Consider regularization.")
        elif overfitting_gap < 0.02:
            print("   ✅ Well-balanced model with minimal overfitting.")
        
        print(f"\n🔗 Access Results:")
        print(f"   Use: from metaflow import Flow; Flow('WineClassificationFlow').latest_run")
        print(f"   All artifacts and metrics are automatically saved!")

if __name__ == '__main__':
    WineClassificationFlow()