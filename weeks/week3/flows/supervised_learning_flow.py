"""
Week 3: Supervised Learning Pipeline with Metaflow

This flow implements a comprehensive supervised learning pipeline with:
- Parallel model training using @foreach
- Multiple algorithm comparison
- Cross-validation and hyperparameter tuning
- Comprehensive evaluation and reporting

Usage:
    python supervised_learning_flow.py run
    python supervised_learning_flow.py run --dataset_type housing --test_size 0.3
"""

from metaflow import FlowSpec, step, Parameter, resources, catch
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SupervisedLearningFlow(FlowSpec):
    """
    Complete supervised learning pipeline with parallel model training.
    
    Features:
    - Data loading and preprocessing
    - Parallel training of multiple algorithms
    - Comprehensive evaluation and comparison
    - Model selection and interpretation
    """
    
    dataset_type = Parameter('dataset_type',
                            help='Type of dataset: wine or housing',
                            default='wine')
    
    test_size = Parameter('test_size',
                         help='Test set size (0.0-1.0)',
                         default=0.2)
    
    n_cv_folds = Parameter('n_cv_folds',
                          help='Number of cross-validation folds',
                          default=5)
    
    random_state = Parameter('random_state',
                            help='Random state for reproducibility',
                            default=42)
    
    @step
    def start(self):
        """
        Load and prepare the dataset for training.
        """
        print(f"🚀 Starting Supervised Learning Pipeline")
        print(f"   Dataset: {self.dataset_type}")
        print(f"   Test size: {self.test_size}")
        print(f"   CV folds: {self.n_cv_folds}")
        print(f"   Random state: {self.random_state}")
        
        if self.dataset_type == 'wine':
            # Load wine dataset for classification
            wine_data = load_wine()
            self.X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
            self.y = wine_data.target
            self.target_names = wine_data.target_names.tolist()
            self.problem_type = 'classification'
            
            # Define algorithms for classification
            self.algorithms = {
                'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
                'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
                'svm': SVC(random_state=self.random_state, probability=True),
                'naive_bayes': GaussianNB()
            }
            
        else:  # housing dataset
            # Create synthetic housing dataset for regression
            X_housing, y_housing = make_regression(
                n_samples=1000,
                n_features=10,
                n_informative=8,
                noise=0.1,
                random_state=self.random_state
            )
            
            # Create feature names
            feature_names = [f'feature_{i}' for i in range(X_housing.shape[1])]
            self.X = pd.DataFrame(X_housing, columns=feature_names)
            self.y = y_housing
            self.target_names = ['target']
            self.problem_type = 'regression'
            
            # Define algorithms for regression
            self.algorithms = {
                'linear_regression': LinearRegression(),
                'ridge_regression': Ridge(alpha=1.0, random_state=self.random_state),
                'lasso_regression': Lasso(alpha=1.0, random_state=self.random_state),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                'svr': SVR(kernel='rbf')
            }
        
        print(f"📊 Dataset shape: {self.X.shape}")
        print(f"🎯 Problem type: {self.problem_type}")
        print(f"🤖 Algorithms: {list(self.algorithms.keys())}")
        
        # Store dataset info for reporting
        self.dataset_info = {
            'shape': self.X.shape,
            'feature_names': list(self.X.columns),
            'target_names': self.target_names,
            'problem_type': self.problem_type
        }
        
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        """
        Preprocess the data: split and scale.
        """
        print("🔧 Preprocessing data...")
        
        # Split the data
        if self.problem_type == 'classification':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, 
                random_state=self.random_state, stratify=self.y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   📊 Training set: {self.X_train_scaled.shape}")
        print(f"   📊 Test set: {self.X_test_scaled.shape}")
        
        # Calculate class distribution for classification
        if self.problem_type == 'classification':
            train_dist = np.bincount(self.y_train)
            test_dist = np.bincount(self.y_test)
            print(f"   🎯 Train distribution: {train_dist}")
            print(f"   🎯 Test distribution: {test_dist}")
        
        # Prepare for parallel training
        self.algorithm_names = list(self.algorithms.keys())
        
        self.next(self.train_model, foreach='algorithm_names')
    
    @resources(memory=4000, cpu=2)
    @catch(var='training_error')
    @step
    def train_model(self):
        """
        Train individual models in parallel.
        """
        # Get current algorithm
        self.current_algorithm = self.input
        algorithm = self.algorithms[self.current_algorithm]
        
        print(f"🏋️ Training {self.current_algorithm}...")
        
        start_time = datetime.now()
        
        try:
            # Train the model
            algorithm.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = algorithm.predict(self.X_test_scaled)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics based on problem type
            if self.problem_type == 'classification':
                accuracy = accuracy_score(self.y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    algorithm, self.X_train_scaled, self.y_train, 
                    cv=self.n_cv_folds, scoring='accuracy'
                )
                
                # Additional metrics
                if hasattr(algorithm, 'predict_proba'):
                    y_pred_proba = algorithm.predict_proba(self.X_test_scaled)
                else:
                    y_pred_proba = None
                
                self.model_results = {
                    'algorithm': self.current_algorithm,
                    'model': algorithm,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist(),
                    'training_time': training_time,
                    'error': None
                }
                
                print(f"   ✅ {self.current_algorithm}: Accuracy={accuracy:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
            else:  # regression
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                # Cross-validation for regression
                cv_scores = cross_val_score(
                    algorithm, self.X_train_scaled, self.y_train, 
                    cv=self.n_cv_folds, scoring='r2'
                )
                
                self.model_results = {
                    'algorithm': self.current_algorithm,
                    'model': algorithm,
                    'predictions': y_pred,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist(),
                    'training_time': training_time,
                    'error': None
                }
                
                print(f"   ✅ {self.current_algorithm}: R²={r2:.3f}, RMSE={rmse:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        except Exception as e:
            print(f"   ❌ {self.current_algorithm}: Training failed - {str(e)}")
            self.model_results = {
                'algorithm': self.current_algorithm,
                'error': str(e),
                'training_time': (datetime.now() - start_time).total_seconds()
            }
        
        self.next(self.evaluate_models)
    
    @step
    def evaluate_models(self, inputs):
        """
        Collect and evaluate all trained models.
        """
        print("📊 Evaluating all models...")
        
        self.merge_artifacts(inputs, exclude=['algorithms', 'model_results', 'current_algorithm'])
        
        # Collect results from all parallel branches
        self.all_results = {}
        self.failed_models = []
        
        for input_data in inputs:
            result = input_data.model_results
            if result.get('error') is None:
                self.all_results[result['algorithm']] = result
            else:
                self.failed_models.append({
                    'algorithm': result['algorithm'],
                    'error': result['error']
                })
        
        print(f"   📈 Successfully trained: {len(self.all_results)} models")
        if self.failed_models:
            print(f"   ❌ Failed models: {len(self.failed_models)}")
            for failed in self.failed_models:
                print(f"      - {failed['algorithm']}: {failed['error']}")
        
        if not self.all_results:
            print("   ⚠️ No models trained successfully!")
            self.best_algorithm = None
            self.best_model = None
            self.best_metric_value = 0
        else:
            # Find best model based on problem type
            if self.problem_type == 'classification':
                best_algorithm = max(self.all_results.keys(), 
                                   key=lambda x: self.all_results[x]['accuracy'])
                best_metric_value = self.all_results[best_algorithm]['accuracy']
                metric_name = 'accuracy'
            else:
                best_algorithm = max(self.all_results.keys(), 
                                   key=lambda x: self.all_results[x]['r2'])
                best_metric_value = self.all_results[best_algorithm]['r2']
                metric_name = 'r2'
            
            self.best_algorithm = best_algorithm
            self.best_model = self.all_results[best_algorithm]['model']
            self.best_metric_value = best_metric_value
            
            print(f"🏆 Best model: {best_algorithm}")
            print(f"   📊 {metric_name.upper()}: {best_metric_value:.3f}")
        
        # Create comparison summary
        self.model_comparison = []
        for alg_name, results in self.all_results.items():
            comparison_entry = {
                'algorithm': alg_name,
                'cv_mean': results['cv_mean'],
                'cv_std': results['cv_std'],
                'training_time': results['training_time']
            }
            
            if self.problem_type == 'classification':
                comparison_entry.update({
                    'accuracy': results['accuracy'],
                    'primary_metric': results['accuracy']
                })
            else:
                comparison_entry.update({
                    'r2': results['r2'],
                    'mse': results['mse'],
                    'rmse': results['rmse'],
                    'mae': results['mae'],
                    'primary_metric': results['r2']
                })
            
            self.model_comparison.append(comparison_entry)
        
        # Sort by primary metric
        self.model_comparison.sort(key=lambda x: x['primary_metric'], reverse=True)
        
        self.next(self.generate_insights)
    
    @step
    def generate_insights(self):
        """
        Generate insights and model interpretation.
        """
        print("🧠 Generating model insights...")
        
        if not self.all_results:
            print("   ⚠️ No successful models to analyze")
            self.feature_importance = None
            self.model_insights = {}
            self.performance_assessment = "Failed"
            self.next(self.end)
            return
        
        # Feature importance analysis (for applicable models)
        self.feature_importance = None
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.X.columns if hasattr(self.X, 'columns') else [f'feature_{i}' for i in range(len(importances))]
            
            # Create feature importance dictionary
            importance_dict = dict(zip(feature_names, importances))
            # Sort by importance
            self.feature_importance = dict(sorted(importance_dict.items(), 
                                                key=lambda x: x[1], reverse=True))
            
            print(f"   🔍 Top 3 features: {list(self.feature_importance.keys())[:3]}")
        
        elif hasattr(self.best_model, 'coef_'):
            # For linear models, use coefficients
            coefficients = self.best_model.coef_
            if coefficients.ndim > 1:
                coefficients = np.abs(coefficients).mean(axis=0)  # For multi-class
            else:
                coefficients = np.abs(coefficients)
            
            feature_names = self.X.columns if hasattr(self.X, 'columns') else [f'feature_{i}' for i in range(len(coefficients))]
            importance_dict = dict(zip(feature_names, coefficients))
            self.feature_importance = dict(sorted(importance_dict.items(), 
                                                key=lambda x: x[1], reverse=True))
            
            print(f"   🔍 Top 3 features (by coefficient): {list(self.feature_importance.keys())[:3]}")
        
        # Model interpretation summary
        self.model_insights = {
            'best_algorithm': self.best_algorithm,
            'best_metric_value': self.best_metric_value,
            'problem_type': self.problem_type,
            'dataset_shape': self.X.shape,
            'feature_importance': self.feature_importance,
            'total_algorithms_tested': len(self.all_results),
            'failed_algorithms': len(self.failed_models),
            'cross_validation_folds': self.n_cv_folds
        }
        
        # Generate performance assessment
        if self.problem_type == 'classification':
            performance_level = (
                'Excellent' if self.best_metric_value > 0.95 else
                'Very Good' if self.best_metric_value > 0.90 else
                'Good' if self.best_metric_value > 0.80 else
                'Fair' if self.best_metric_value > 0.70 else
                'Poor'
            )
        else:
            performance_level = (
                'Excellent' if self.best_metric_value > 0.90 else
                'Very Good' if self.best_metric_value > 0.80 else
                'Good' if self.best_metric_value > 0.70 else
                'Fair' if self.best_metric_value > 0.50 else
                'Poor'
            )
        
        self.performance_assessment = performance_level
        
        print(f"   🎯 Performance level: {performance_level}")
        
        # Generate recommendations
        self.recommendations = []
        
        if performance_level in ['Excellent', 'Very Good']:
            self.recommendations.extend([
                'Model is ready for production deployment',
                'Implement monitoring and alerting systems',
                'Consider A/B testing framework'
            ])
        elif performance_level == 'Good':
            self.recommendations.extend([
                'Consider ensemble methods for improvement',
                'Collect more training data if possible',
                'Implement comprehensive testing before deployment'
            ])
        else:
            self.recommendations.extend([
                'Enhance feature engineering techniques',
                'Try advanced algorithms (XGBoost, Neural Networks)',
                'Review data quality and collection process'
            ])
        
        # Always add general recommendations
        self.recommendations.extend([
            'Implement hyperparameter tuning for top models',
            'Create model interpretation dashboard',
            'Plan for model monitoring and retraining'
        ])
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Generate final report and summary.
        """
        print("🎉 Supervised Learning Pipeline Complete!")
        print("=" * 45)
        
        if not self.all_results:
            print("❌ Pipeline failed - no models trained successfully")
            self.final_summary = {
                'status': 'failed',
                'error': 'No models trained successfully',
                'timestamp': datetime.now().isoformat()
            }
            return
        
        print(f"📊 Problem Type: {self.problem_type.title()}")
        print(f"📈 Dataset: {self.dataset_type} ({self.X.shape[0]} samples, {self.X.shape[1]} features)")
        print(f"🏆 Best Algorithm: {self.best_algorithm}")
        
        if self.problem_type == 'classification':
            print(f"🎯 Best Accuracy: {self.best_metric_value:.3f}")
        else:
            print(f"🎯 Best R² Score: {self.best_metric_value:.3f}")
        
        print(f"⭐ Performance Level: {self.performance_assessment}")
        print(f"🔬 Algorithms Tested: {len(self.all_results)}")
        
        if self.failed_models:
            print(f"❌ Failed Algorithms: {len(self.failed_models)}")
        
        if self.feature_importance:
            print(f"🔍 Most Important Features:")
            for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:5], 1):
                print(f"   {i}. {feature}: {importance:.3f}")
        
        # Model comparison summary
        print(f"\n📊 Model Performance Summary:")
        for i, model in enumerate(self.model_comparison[:5], 1):
            if self.problem_type == 'classification':
                print(f"   {i}. {model['algorithm']}: {model['accuracy']:.3f} (CV: {model['cv_mean']:.3f}±{model['cv_std']:.3f})")
            else:
                print(f"   {i}. {model['algorithm']}: R²={model['r2']:.3f} (CV: {model['cv_mean']:.3f}±{model['cv_std']:.3f})")
        
        print(f"\n🚀 Key Recommendations:")
        for i, rec in enumerate(self.recommendations[:3], 1):
            print(f"   {i}. {rec}")
        
        # Create final summary
        self.final_summary = {
            'status': 'success',
            'pipeline_type': 'supervised_learning',
            'dataset_type': self.dataset_type,
            'problem_type': self.problem_type,
            'dataset_info': self.dataset_info,
            'best_algorithm': self.best_algorithm,
            'best_metric_value': self.best_metric_value,
            'performance_assessment': self.performance_assessment,
            'model_comparison': self.model_comparison,
            'model_insights': self.model_insights,
            'recommendations': self.recommendations,
            'failed_models': self.failed_models,
            'parameters': {
                'test_size': self.test_size,
                'n_cv_folds': self.n_cv_folds,
                'random_state': self.random_state
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print("\n✨ All results automatically saved by Metaflow!")
        print("💡 Access results with:")
        print("   from metaflow import Flow")
        print("   run = Flow('SupervisedLearningFlow').latest_run")
        print("   print(run.data.final_summary)")


if __name__ == '__main__':
    SupervisedLearningFlow()
