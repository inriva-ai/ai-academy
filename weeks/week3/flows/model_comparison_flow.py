"""
Week 3: Model Comparison and Hyperparameter Tuning Flow

This flow focuses on advanced model comparison with hyperparameter tuning
and comprehensive evaluation metrics.

Usage:
    python model_comparison_flow.py run
    python model_comparison_flow.py run --tuning_method grid --n_jobs 4
"""

from metaflow import FlowSpec, step, Parameter, foreach, resources, catch
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, roc_auc_score
)
from scipy.stats import randint, uniform
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelComparisonFlow(FlowSpec):
    """
    Advanced model comparison flow with hyperparameter tuning.
    
    Features:
    - Multiple dataset support
    - Grid search and random search
    - Parallel hyperparameter tuning
    - Comprehensive model evaluation
    - Performance comparison and ranking
    """
    
    dataset = Parameter('dataset',
                       help='Dataset to use: wine, breast_cancer',
                       default='wine')
    
    tuning_method = Parameter('tuning_method',
                             help='Hyperparameter tuning method: grid, random, none',
                             default='grid')
    
    n_jobs = Parameter('n_jobs',
                      help='Number of parallel jobs for tuning',
                      default=2)
    
    cv_folds = Parameter('cv_folds',
                        help='Number of CV folds',
                        default=5)
    
    random_state = Parameter('random_state',
                            help='Random state for reproducibility',
                            default=42)
    
    @step
    def start(self):
        """
        Load dataset and define models for comparison.
        """
        print(f"🚀 Starting Model Comparison Flow")
        print(f"   Dataset: {self.dataset}")
        print(f"   Tuning method: {self.tuning_method}")
        print(f"   Parallel jobs: {self.n_jobs}")
        print(f"   CV folds: {self.cv_folds}")
        
        # Load dataset
        if self.dataset == 'wine':
            data = load_wine()
            self.X = pd.DataFrame(data.data, columns=data.feature_names)
            self.y = data.target
            self.target_names = data.target_names.tolist()
            self.dataset_name = 'Wine Classification'
        elif self.dataset == 'breast_cancer':
            data = load_breast_cancer()
            self.X = pd.DataFrame(data.data, columns=data.feature_names)
            self.y = data.target
            self.target_names = data.target_names.tolist()
            self.dataset_name = 'Breast Cancer Classification'
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        print(f"📊 Dataset: {self.dataset_name}")
        print(f"   Shape: {self.X.shape}")
        print(f"   Classes: {len(self.target_names)}")
        
        # Define models and their parameter grids
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'param_dist': {
                    'C': uniform(0.1, 100),
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'param_dist': {
                    'n_estimators': randint(50, 200),
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': randint(2, 11),
                    'min_samples_leaf': randint(1, 5)
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'param_dist': {
                    'n_estimators': randint(50, 200),
                    'learning_rate': uniform(0.01, 0.2),
                    'max_depth': randint(3, 8)
                }
            },
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                },
                'param_dist': {
                    'C': uniform(0.1, 10),
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
        
        self.model_names = list(self.model_configs.keys())
        print(f"🤖 Models to compare: {self.model_names}")
        
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        """
        Preprocess the data for model training.
        """
        print("🔧 Preprocessing data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, 
            random_state=self.random_state, stratify=self.y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   📊 Training set: {self.X_train_scaled.shape}")
        print(f"   📊 Test set: {self.X_test_scaled.shape}")
        
        # Class distribution
        train_dist = np.bincount(self.y_train)
        test_dist = np.bincount(self.y_test)
        print(f"   🎯 Train distribution: {train_dist}")
        print(f"   🎯 Test distribution: {test_dist}")
        
        self.next(self.tune_model, foreach='model_names')
    
    @resources(memory=6000, cpu=4)
    @catch(var='tuning_error')
    @step
    def tune_model(self):
        """
        Tune hyperparameters for each model in parallel.
        """
        self.current_model_name = self.input
        model_config = self.model_configs[self.current_model_name]
        
        print(f"🎛️ Tuning {self.current_model_name}...")
        
        start_time = datetime.now()
        
        try:
            base_model = model_config['model']
            
            if self.tuning_method == 'grid':
                # Grid Search
                param_grid = model_config['param_grid']
                search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=self.cv_folds,
                    scoring='accuracy',
                    n_jobs=self.n_jobs,
                    verbose=0
                )
                search_type = 'Grid Search'
                
            elif self.tuning_method == 'random':
                # Random Search
                param_dist = model_config['param_dist']
                search = RandomizedSearchCV(
                    base_model,
                    param_dist,
                    n_iter=20,  # Number of parameter combinations to try
                    cv=self.cv_folds,
                    scoring='accuracy',
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=0
                )
                search_type = 'Random Search'
                
            else:  # no tuning
                # Use default parameters
                base_model.fit(self.X_train_scaled, self.y_train)
                self.tuned_model = base_model
                best_params = {}
                best_cv_score = cross_val_score(
                    base_model, self.X_train_scaled, self.y_train,
                    cv=self.cv_folds, scoring='accuracy'
                ).mean()
                search_type = 'Default Parameters'
                
            if self.tuning_method != 'none':
                # Perform hyperparameter search
                search.fit(self.X_train_scaled, self.y_train)
                self.tuned_model = search.best_estimator_
                best_params = search.best_params_
                best_cv_score = search.best_score_
            
            # Evaluate on test set
            y_pred = self.tuned_model.predict(self.X_test_scaled)
            y_pred_proba = self.tuned_model.predict_proba(self.X_test_scaled)
            
            test_accuracy = accuracy_score(self.y_test, y_pred)
            
            # Calculate additional metrics
            test_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='macro')
            
            # Cross-validation on tuned model
            final_cv_scores = cross_val_score(
                self.tuned_model, self.X_train_scaled, self.y_train,
                cv=self.cv_folds, scoring='accuracy'
            )
            
            tuning_time = (datetime.now() - start_time).total_seconds()
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(self.tuned_model, 'feature_importances_'):
                importances = self.tuned_model.feature_importances_
                feature_names = list(self.X.columns)
                feature_importance = dict(zip(feature_names, importances))
                feature_importance = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True))
            elif hasattr(self.tuned_model, 'coef_'):
                coefficients = np.abs(self.tuned_model.coef_)
                if coefficients.ndim > 1:
                    coefficients = coefficients.mean(axis=0)
                feature_names = list(self.X.columns)
                feature_importance = dict(zip(feature_names, coefficients))
                feature_importance = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True))
            
            self.model_results = {
                'model_name': self.current_model_name,
                'search_type': search_type,
                'tuned_model': self.tuned_model,
                'best_params': best_params,
                'best_cv_score': best_cv_score,
                'test_accuracy': test_accuracy,
                'test_auc': test_auc,
                'final_cv_scores': final_cv_scores.tolist(),
                'final_cv_mean': final_cv_scores.mean(),
                'final_cv_std': final_cv_scores.std(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist(),
                'feature_importance': feature_importance,
                'tuning_time': tuning_time,
                'error': None
            }
            
            print(f"   ✅ {self.current_model_name} ({search_type}):")
            print(f"      Test Accuracy: {test_accuracy:.3f}")
            print(f"      CV Score: {best_cv_score:.3f}")
            print(f"      AUC: {test_auc:.3f}")
            print(f"      Tuning Time: {tuning_time:.1f}s")
            if best_params:
                print(f"      Best Params: {best_params}")
        
        except Exception as e:
            print(f"   ❌ {self.current_model_name}: Tuning failed - {str(e)}")
            self.model_results = {
                'model_name': self.current_model_name,
                'error': str(e),
                'tuning_time': (datetime.now() - start_time).total_seconds()
            }
        
        self.next(self.compare_models)
    
    @step
    def compare_models(self, inputs):
        """
        Compare all tuned models and generate comprehensive evaluation.
        """
        print("📊 Comparing all tuned models...")
        
        # Collect results from all parallel branches
        self.all_model_results = {}
        self.failed_models = []
        
        for input_data in inputs:
            result = input_data.model_results
            if result.get('error') is None:
                self.all_model_results[result['model_name']] = result
            else:
                self.failed_models.append({
                    'model_name': result['model_name'],
                    'error': result['error']
                })
        
        print(f"   📈 Successfully tuned: {len(self.all_model_results)} models")
        if self.failed_models:
            print(f"   ❌ Failed models: {len(self.failed_models)}")
        
        if not self.all_model_results:
            print("   ⚠️ No models tuned successfully!")
            self.next(self.end)
            return
        
        # Find best model
        best_model_name = max(self.all_model_results.keys(), 
                             key=lambda x: self.all_model_results[x]['test_accuracy'])
        self.best_model_results = self.all_model_results[best_model_name]
        
        print(f"🏆 Best model: {best_model_name}")
        print(f"   📊 Test Accuracy: {self.best_model_results['test_accuracy']:.3f}")
        print(f"   📈 AUC: {self.best_model_results['test_auc']:.3f}")
        
        # Create detailed comparison
        self.model_comparison = []
        for model_name, results in self.all_model_results.items():
            self.model_comparison.append({
                'model_name': model_name,
                'search_type': results['search_type'],
                'test_accuracy': results['test_accuracy'],
                'test_auc': results['test_auc'],
                'cv_mean': results['final_cv_mean'],
                'cv_std': results['final_cv_std'],
                'tuning_time': results['tuning_time'],
                'best_params': results['best_params']
            })
        
        # Sort by test accuracy
        self.model_comparison.sort(key=lambda x: x['test_accuracy'], reverse=True)
        
        # Generate detailed evaluation for best model
        self.generate_detailed_evaluation()
        
        self.next(self.generate_report)
    
    def generate_detailed_evaluation(self):
        """
        Generate detailed evaluation metrics for the best model.
        """
        print(f"🔍 Generating detailed evaluation for {self.best_model_results['model_name']}...")
        
        best_model = self.best_model_results['tuned_model']
        y_pred = np.array(self.best_model_results['predictions'])
        y_pred_proba = np.array(self.best_model_results['probabilities'])
        
        # Classification report
        class_report = classification_report(
            self.y_test, y_pred, 
            target_names=self.target_names,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        # Per-class metrics
        per_class_auc = []
        if len(self.target_names) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba[:, 1])
            auc_score = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            per_class_auc.append(auc_score)
        else:
            # Multi-class classification
            for i in range(len(self.target_names)):
                y_test_binary = (self.y_test == i).astype(int)
                auc_score = roc_auc_score(y_test_binary, y_pred_proba[:, i])
                per_class_auc.append(auc_score)
        
        self.detailed_evaluation = {
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_auc': per_class_auc,
            'feature_importance': self.best_model_results['feature_importance']
        }
        
        print(f"   ✅ Detailed evaluation complete")
    
    @step
    def generate_report(self):
        """
        Generate comprehensive comparison report.
        """
        print("📄 Generating comprehensive comparison report...")
        
        # Performance summary
        self.performance_summary = {
            'best_model': self.best_model_results['model_name'],
            'best_accuracy': self.best_model_results['test_accuracy'],
            'best_auc': self.best_model_results['test_auc'],
            'models_compared': len(self.all_model_results),
            'tuning_method': self.tuning_method,
            'dataset': self.dataset_name
        }
        
        # Statistical analysis
        accuracies = [r['test_accuracy'] for r in self.all_model_results.values()]
        self.statistical_summary = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'accuracy_range': np.max(accuracies) - np.min(accuracies)
        }
        
        # Generate insights
        self.insights = self.generate_insights()
        
        print(f"   📊 Report generated successfully")
        print(f"   🏆 Best model: {self.performance_summary['best_model']}")
        print(f"   📈 Best accuracy: {self.performance_summary['best_accuracy']:.3f}")
        
        self.next(self.end)
    
    def generate_insights(self):
        """
        Generate actionable insights from model comparison.
        """
        insights = []
        
        # Performance insights
        best_acc = self.performance_summary['best_accuracy']
        if best_acc > 0.95:
            insights.append("Excellent performance achieved - models are production-ready")
        elif best_acc > 0.90:
            insights.append("Very good performance - consider additional validation")
        elif best_acc > 0.80:
            insights.append("Good performance - may benefit from more data or feature engineering")
        else:
            insights.append("Performance needs improvement - review data quality and feature selection")
        
        # Model comparison insights
        acc_range = self.statistical_summary['accuracy_range']
        if acc_range < 0.05:
            insights.append("Models perform similarly - consider ensemble methods")
        else:
            insights.append("Significant performance differences between models")
        
        # Tuning insights
        if self.tuning_method != 'none':
            best_model_name = self.performance_summary['best_model']
            best_results = self.all_model_results[best_model_name]
            cv_improvement = best_results['test_accuracy'] - best_results['best_cv_score']
            
            if abs(cv_improvement) < 0.02:
                insights.append("Good generalization - CV scores align with test performance")
            elif cv_improvement < -0.05:
                insights.append("Possible overfitting - test performance lower than CV")
            else:
                insights.append("Test performance exceeds CV - consider more robust validation")
        
        # Feature importance insights
        if self.best_model_results['feature_importance']:
            top_features = list(self.best_model_results['feature_importance'].keys())[:3]
            insights.append(f"Key predictive features: {', '.join(top_features[:3])}")
        
        return insights
    
    @step
    def end(self):
        """
        Finalize model comparison and display summary.
        """
        print("🎉 Model Comparison Flow Complete!")
        print("=" * 40)
        
        if not self.all_model_results:
            print("❌ No models were successfully tuned")
            return
        
        print(f"📊 Dataset: {self.dataset_name}")
        print(f"🎛️ Tuning Method: {self.tuning_method}")
        print(f"🏆 Best Model: {self.performance_summary['best_model']}")
        print(f"📈 Best Accuracy: {self.performance_summary['best_accuracy']:.3f}")
        print(f"📊 Best AUC: {self.performance_summary['best_auc']:.3f}")
        
        print(f"\n📋 Model Rankings:")
        for i, model in enumerate(self.model_comparison, 1):
            print(f"   {i}. {model['model_name']}: {model['test_accuracy']:.3f} "
                  f"(AUC: {model['test_auc']:.3f}, Time: {model['tuning_time']:.1f}s)")
        
        print(f"\n🧠 Key Insights:")
        for insight in self.insights:
            print(f"   • {insight}")
        
        if self.best_model_results['feature_importance']:
            print(f"\n🔍 Top Features ({self.performance_summary['best_model']}):")
            top_features = list(self.best_model_results['feature_importance'].items())[:5]
            for feature, importance in top_features:
                print(f"   • {feature}: {importance:.3f}")
        
        # Create final summary
        self.final_summary = {
            'status': 'success',
            'performance_summary': self.performance_summary,
            'statistical_summary': self.statistical_summary,
            'model_comparison': self.model_comparison,
            'detailed_evaluation': self.detailed_evaluation,
            'insights': self.insights,
            'failed_models': self.failed_models,
            'parameters': {
                'dataset': self.dataset,
                'tuning_method': self.tuning_method,
                'n_jobs': self.n_jobs,
                'cv_folds': self.cv_folds,
                'random_state': self.random_state
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print("\n✨ Complete model comparison results saved!")
        print("💡 Access results with:")
        print("   from metaflow import Flow")
        print("   run = Flow('ModelComparisonFlow').latest_run")
        print("   print(run.data.final_summary)")


if __name__ == '__main__':
    ModelComparisonFlow()
