# Week 3: Complete Workshop Solutions
# Supervised Learning with Metaflow Pipelines - Full Implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import time
import joblib
from pathlib import Path

# Machine Learning Core
from sklearn.datasets import (
    load_wine, load_breast_cancer, load_diabetes, 
    make_classification, make_regression
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold,
    GridSearchCV, learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance
from scipy import stats
from scipy.stats import ttest_rel

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")
np.random.seed(42)

print("✅ Complete Workshop Environment Ready!")
print(f"📅 Session started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# DATASET MANAGER - Advanced Dataset Handling
# =============================================================================

class DatasetManager:
    """Complete dataset management for workshop exercises."""
    
    def __init__(self):
        self.datasets = {}
        self.metadata = {}
        print("🗃️ Initializing Dataset Manager")
    
    def load_classification_datasets(self):
        """Load and prepare all classification datasets."""
        print("\n📊 Loading Classification Datasets")
        print("=" * 35)
        
        # Wine Classification (Multi-class, balanced)
        wine_data = load_wine()
        self.datasets['wine'] = {
            'X': pd.DataFrame(wine_data.data, columns=wine_data.feature_names),
            'y': wine_data.target,
            'target_names': wine_data.target_names,
            'type': 'classification',
            'n_classes': len(wine_data.target_names),
            'balanced': True
        }
        
        # Breast Cancer (Binary, balanced)
        cancer_data = load_breast_cancer()
        self.datasets['cancer'] = {
            'X': pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names),
            'y': cancer_data.target,
            'target_names': cancer_data.target_names,
            'type': 'classification',
            'n_classes': 2,
            'balanced': True
        }
        
        # Synthetic Imbalanced (Multi-class, imbalanced)
        X_imb, y_imb = make_classification(
            n_samples=2000, n_features=20, n_informative=15,
            n_classes=3, weights=[0.7, 0.2, 0.1], random_state=42
        )
        
        self.datasets['imbalanced'] = {
            'X': pd.DataFrame(X_imb, columns=[f'feature_{i:02d}' for i in range(20)]),
            'y': y_imb,
            'target_names': ['Majority', 'Medium', 'Minority'],
            'type': 'classification',
            'n_classes': 3,
            'balanced': False
        }
        
        for name, data in self.datasets.items():
            if data['type'] == 'classification':
                X, y = data['X'], data['y']
                class_dist = np.bincount(y)
                print(f"\n📋 {name.title()} Dataset:")
                print(f"   Shape: {X.shape}")
                print(f"   Classes: {data['n_classes']} - {list(data['target_names'])}")
                print(f"   Distribution: {class_dist}")
    
    def load_regression_datasets(self):
        """Load and prepare all regression datasets."""
        print("\n📈 Loading Regression Datasets")
        print("=" * 31)
        
        # Diabetes (Real-world, low-dimensional)
        diabetes_data = load_diabetes()
        self.datasets['diabetes'] = {
            'X': pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names),
            'y': diabetes_data.target,
            'type': 'regression',
            'target_name': 'diabetes_progression'
        }
        
        # Synthetic Housing Prices
        np.random.seed(42)
        n_samples = 1000
        
        house_size = np.random.normal(2000, 500, n_samples)
        bedrooms = np.random.poisson(3, n_samples) + 1
        bathrooms = np.random.normal(2.5, 0.8, n_samples)
        age = np.random.exponential(15, n_samples)
        location_score = np.random.uniform(1, 10, n_samples)
        
        price = (
            house_size * 150 + bedrooms * 15000 + bathrooms * 8000 +
            location_score * 12000 - age * 800 +
            np.random.normal(0, 25000, n_samples) + 50000
        )
        
        housing_X = pd.DataFrame({
            'house_size': house_size, 'bedrooms': bedrooms,
            'bathrooms': bathrooms, 'age': age, 'location_score': location_score
        })
        
        self.datasets['housing'] = {
            'X': housing_X, 'y': price,
            'type': 'regression', 'target_name': 'price_usd'
        }
        
        for name, data in self.datasets.items():
            if data['type'] == 'regression':
                X, y = data['X'], data['y']
                print(f"\n📋 {name.title()} Dataset:")
                print(f"   Shape: {X.shape}")
                print(f"   Target: {data['target_name']}")
                print(f"   Range: [{y.min():.1f}, {y.max():.1f}]")
    
    def get_dataset(self, name, return_split=True, test_size=0.2, random_state=42):
        """Get dataset with optional train/test split."""
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found. Available: {list(self.datasets.keys())}")
        
        data = self.datasets[name]
        X, y = data['X'], data['y']
        
        if return_split:
            if data['type'] == 'classification':
                return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            else:
                return train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            return X, y
    
    def get_dataset_info(self, name):
        """Get detailed information about a dataset."""
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")
        
        data = self.datasets[name]
        X, y = data['X'], data['y']
        
        info = {
            'name': name, 'type': data['type'], 'shape': X.shape,
            'features': list(X.columns), 'n_features': X.shape[1], 'n_samples': X.shape[0]
        }
        
        if data['type'] == 'classification':
            info.update({
                'n_classes': data['n_classes'],
                'target_names': data['target_names'],
                'class_distribution': np.bincount(y).tolist(),
                'balanced': data['balanced']
            })
        else:
            info.update({
                'target_name': data['target_name'],
                'target_range': [float(y.min()), float(y.max())],
                'target_mean': float(y.mean()), 'target_std': float(y.std())
            })
        
        return info

# Initialize dataset manager
dataset_manager = DatasetManager()
dataset_manager.load_classification_datasets()
dataset_manager.load_regression_datasets()

# =============================================================================
# ADVANCED MODEL COMPARISON FRAMEWORK
# =============================================================================

class AdvancedModelComparison:
    """Comprehensive model comparison with statistical analysis."""
    
    def __init__(self, dataset_name='wine', test_size=0.2, cv_folds=5):
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare dataset for comparison."""
        self.X_train, self.X_test, self.y_train, self.y_test = dataset_manager.get_dataset(
            self.dataset_name, test_size=self.test_size
        )
        
        dataset_info = dataset_manager.get_dataset_info(self.dataset_name)
        self.problem_type = dataset_info['type']
        self.dataset_info = dataset_info
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Full dataset for CV
        self.X, self.y = dataset_manager.get_dataset(self.dataset_name, return_split=False)
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"🔬 Comparison initialized for {self.dataset_name} ({self.problem_type})")
    
    def define_models(self):
        """Define comprehensive model suite."""
        if self.problem_type == 'classification':
            self.models = {
                'Logistic Regression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000),
                    'scaled': True, 'category': 'Linear'
                },
                'Random Forest': {
                    'model': RandomForestClassifier(n_estimators=100, random_state=42),
                    'scaled': False, 'category': 'Ensemble'
                },
                'Gradient Boosting': {
                    'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'scaled': False, 'category': 'Ensemble'
                },
                'SVM (RBF)': {
                    'model': SVC(kernel='rbf', random_state=42, probability=True),
                    'scaled': True, 'category': 'Kernel'
                },
                'K-Nearest Neighbors': {
                    'model': KNeighborsClassifier(n_neighbors=5),
                    'scaled': True, 'category': 'Instance'
                },
                'Naive Bayes': {
                    'model': GaussianNB(), 'scaled': False, 'category': 'Probabilistic'
                }
            }
            self.primary_metric = 'f1_weighted'
        else:
            self.models = {
                'Linear Regression': {
                    'model': LinearRegression(), 'scaled': True, 'category': 'Linear'
                },
                'Ridge Regression': {
                    'model': Ridge(random_state=42), 'scaled': True, 'category': 'Linear'
                },
                'Random Forest': {
                    'model': RandomForestRegressor(n_estimators=100, random_state=42),
                    'scaled': False, 'category': 'Ensemble'
                },
                'Gradient Boosting': {
                    'model': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'scaled': False, 'category': 'Ensemble'
                },
                'SVR': {
                    'model': SVR(), 'scaled': True, 'category': 'Kernel'
                }
            }
            self.primary_metric = 'r2'
    
    def run_comparison(self):
        """Run comprehensive model comparison."""
        print(f"\n🔬 Running Model Comparison")
        print("=" * 28)
        
        self.define_models()
        self.results = {}
        self.cv_results = {}
        
        # CV setup
        if self.problem_type == 'classification':
            cv_splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        else:
            cv_splitter = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Train and evaluate each model
        for name, model_config in self.models.items():
            print(f"\n🔄 Evaluating {name}...")
            
            model = model_config['model']
            use_scaled = model_config['scaled']
            category = model_config['category']
            
            # Select data
            if use_scaled:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
                X_cv_use = self.X_scaled
            else:
                X_train_use = self.X_train.values
                X_test_use = self.X_test.values
                X_cv_use = self.X.values
            
            # Training
            start_time = time.time()
            model.fit(X_train_use, self.y_train)
            training_time = time.time() - start_time
            
            # Predictions
            y_pred = model.predict(X_test_use)
            
            # Metrics
            if self.problem_type == 'classification':
                test_metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'f1': f1_score(self.y_test, y_pred, average='weighted'),
                    'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(self.y_test, y_pred, average='weighted')
                }
            else:
                test_metrics = {
                    'r2': r2_score(self.y_test, y_pred),
                    'mse': mean_squared_error(self.y_test, y_pred),
                    'mae': mean_absolute_error(self.y_test, y_pred)
                }
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_cv_use, self.y, cv=cv_splitter, scoring=self.primary_metric)
            
            self.results[name] = {
                'model': model, 'category': category, 'test_metrics': test_metrics,
                'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
                'training_time': training_time, 'predictions': y_pred,
                'uses_scaling': use_scaled
            }
            
            self.cv_results[name] = cv_scores
            
            print(f"   ✅ CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"   ⏱️ Time: {training_time:.3f}s | Category: {category}")
        
        print(f"\n✅ Comparison complete!")
        return self.results
    
    def statistical_significance_testing(self):
        """Perform statistical significance testing."""
        print(f"\n📊 Statistical Significance Testing")
        print("=" * 36)
        
        # Rank models
        model_rankings = sorted(
            [(name, scores.mean()) for name, scores in self.cv_results.items()],
            key=lambda x: x[1], reverse=True
        )
        
        print(f"🏆 Model Rankings (by CV {self.primary_metric}):")
        for i, (name, score) in enumerate(model_rankings, 1):
            print(f"   {i:2d}. {name:20} | {score:.3f}")
        
        # Statistical tests (top 3 models)
        top_models = model_rankings[:3]
        significance_results = []
        
        print(f"\n🔬 Pairwise Tests (α=0.05):")
        for i in range(len(top_models)):
            for j in range(i + 1, len(top_models)):
                model1_name, model1_score = top_models[i]
                model2_name, model2_score = top_models[j]
                
                scores1 = self.cv_results[model1_name]
                scores2 = self.cv_results[model2_name]
                
                # Paired t-test
                t_stat, p_value = ttest_rel(scores1, scores2)
                
                significance = "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
                
                significance_results.append({
                    'model1': model1_name, 'model2': model2_name,
                    'p_value': p_value, 'significant': p_value < 0.05
                })
                
                print(f"   {model1_name} vs {model2_name}:")
                print(f"      p-value: {p_value:.4f} | {significance}")
        
        self.significance_results = significance_results
        
        # Summary
        significant_pairs = sum(1 for r in significance_results if r['significant'])
        total_pairs = len(significance_results)
        
        print(f"\n📋 Summary:")
        print(f"   Significant differences: {significant_pairs}/{total_pairs}")
        print(f"   Best model: {model_rankings[0][0]} ({model_rankings[0][1]:.3f})")
        
        return significance_results
    
    def generate_report(self):
        """Generate comprehensive comparison report."""
        print(f"\n📋 Comparison Report")
        print("=" * 19)
        
        # Overall summary
        print(f"\n🎯 Summary:")
        print(f"   Dataset: {self.dataset_name} ({self.problem_type})")
        print(f"   Models tested: {len(self.results)}")
        print(f"   CV folds: {self.cv_folds}")
        
        # Best performers by category
        categories = set(r['category'] for r in self.results.values())
        
        print(f"\n🏆 Best by Category:")
        for category in sorted(categories):
            category_models = {name: results for name, results in self.results.items() 
                             if results['category'] == category}
            
            if category_models:
                best_in_category = max(category_models.items(), key=lambda x: x[1]['cv_mean'])
                name, results = best_in_category
                score = results['cv_mean']
                print(f"   {category:12}: {name:20} | {score:.3f}")
        
        # Performance vs efficiency
        print(f"\n⚡ Performance vs Efficiency:")
        best_perf = max(self.results.items(), key=lambda x: x[1]['cv_mean'])
        fastest = min(self.results.items(), key=lambda x: x[1]['training_time'])
        
        print(f"   Best Performance: {best_perf[0]} ({best_perf[1]['cv_mean']:.3f})")
        print(f"   Fastest: {fastest[0]} ({fastest[1]['training_time']:.3f}s)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            print(f"   • {rec}")
        
        return {
            'best_model': best_perf[0], 'best_score': best_perf[1]['cv_mean'],
            'fastest_model': fastest[0], 'recommendations': recommendations
        }
    
    def _generate_recommendations(self):
        """Generate recommendations based on results."""
        recommendations = []
        
        # Get best performer
        best_model = max(self.results.items(), key=lambda x: x[1]['cv_mean'])
        best_score = best_model[1]['cv_mean']
        
        # Performance-based recommendations
        if self.problem_type == 'classification':
            if best_score > 0.95:
                recommendations.append("Excellent performance - ready for production")
            elif best_score > 0.90:
                recommendations.append("Very good performance - consider hyperparameter tuning")
            elif best_score > 0.80:
                recommendations.append("Good performance - try feature engineering")
            else:
                recommendations.append("Performance needs improvement - try more data")
        else:
            if best_score > 0.90:
                recommendations.append("Excellent predictive power - ready for production")
            elif best_score > 0.75:
                recommendations.append("Good performance - consider regularization tuning")
            else:
                recommendations.append("Consider polynomial features or more complex models")
        
        # Model-specific recommendations
        best_category = best_model[1]['category']
        if best_category == 'Ensemble':
            recommendations.append("Ensemble methods work well - consider stacking")
        elif best_category == 'Linear':
            recommendations.append("Linear models sufficient - prioritize interpretability")
        
        return recommendations

print("✅ Advanced Model Comparison Framework Ready!")

# =============================================================================
# MODEL INTERPRETATION SYSTEM
# =============================================================================

class ModelInterpreter:
    """Advanced model interpretation system."""
    
    def __init__(self):
        print("🧠 Model Interpreter initialized")
    
    def interpret_single_model(self, model_name, model_results, dataset_name, feature_names=None):
        """Interpret a single model's results."""
        # Prepare performance summary
        if 'test_metrics' in model_results:
            metrics = model_results['test_metrics']
            if 'accuracy' in metrics:
                performance = f"Accuracy: {metrics['accuracy']:.3f}, F1: {metrics.get('f1', 'N/A'):.3f}"
            else:
                performance = f"R²: {metrics.get('r2', 'N/A'):.3f}, MAE: {metrics.get('mae', 'N/A'):.3f}"
        else:
            performance = f"CV Score: {model_results.get('cv_mean', 'N/A'):.3f}"
        
        # Get feature importance if available
        features = "Feature importance not available"
        if hasattr(model_results.get('model'), 'feature_importances_'):
            importances = model_results['model'].feature_importances_
            if feature_names:
                top_features = sorted(zip(feature_names, importances), 
                                    key=lambda x: x[1], reverse=True)[:5]
                features = ", ".join([f"{name} ({imp:.3f})" for name, imp in top_features])
        
        interpretation = f"""
📊 Model Interpretation:

🤖 Model: {model_name}
📈 Performance: {performance}
🏷️ Category: {model_results.get('category', 'Unknown')}
📊 Dataset: {dataset_name}

💡 Key Insights:
• Performance level: {'Excellent' if model_results.get('cv_mean', 0) > 0.9 else 'Good' if model_results.get('cv_mean', 0) > 0.8 else 'Fair'}
• Training time: {model_results.get('training_time', 0):.3f} seconds
• Top features: {features}

🎯 Recommendations:
• {'Deploy with confidence' if model_results.get('cv_mean', 0) > 0.9 else 'Consider hyperparameter tuning'}
• Monitor performance on new data
• {'Ready for business use' if model_results.get('cv_mean', 0) > 0.85 else 'May need additional training data'}
        """
        
        return interpretation
    
    def interpret_model_comparison(self, comparison_results, dataset_name):
        """Interpret model comparison results."""
        # Find best model
        best_model = max(comparison_results.items(), key=lambda x: x[1].get('cv_mean', 0))
        best_name, best_results = best_model
        best_score = best_results.get('cv_mean', 0)
        
        # Category analysis
        categories = {}
        for name, results in comparison_results.items():
            cat = results.get('category', 'Unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, results.get('cv_mean', 0)))
        
        interpretation = f"""
📊 Model Comparison Analysis:

🏆 Best Performer: {best_name}
📈 Best Score: {best_score:.3f}
📊 Dataset: {dataset_name}
🤖 Models Tested: {len(comparison_results)}

📋 Category Performance:
"""
        
        for cat, models in categories.items():
            best_in_cat = max(models, key=lambda x: x[1])
            interpretation += f"• {cat}: {best_in_cat[0]} ({best_in_cat[1]:.3f})\n"
        
        interpretation += f"""
💡 Key Insights:
• {best_name} provides the best overall performance
• {'Excellent' if best_score > 0.9 else 'Good' if best_score > 0.8 else 'Moderate'} results achieved
• Consider ensemble methods if top models are close

🎯 Recommendations:
• Deploy {best_name} for production use
• Monitor performance and retrain as needed
• Consider hyperparameter tuning for optimization
        """
        
        return interpretation

# Initialize interpreter
model_interpreter = ModelInterpreter()

# =============================================================================
# WORKSHOP VISUALIZER
# =============================================================================

class WorkshopVisualizer:
    """Complete visualization suite for workshop results."""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_model_comparison(self, comparison_results, dataset_name, save_path=None):
        """Create comprehensive model comparison visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Comparison Analysis - {dataset_name}', fontsize=16, fontweight='bold')
        
        # Prepare data
        models = list(comparison_results.keys())
        cv_means = [r['cv_mean'] for r in comparison_results.values()]
        cv_stds = [r['cv_std'] for r in comparison_results.values()]
        training_times = [r['training_time'] for r in comparison_results.values()]
        categories = [r['category'] for r in comparison_results.values()]
        
        # 1. CV Performance with error bars
        x_pos = np.arange(len(models))
        bars = axes[0, 0].bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('CV Score')
        axes[0, 0].set_title('Cross-Validation Performance')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([m[:10] + '...' if len(m) > 10 else m for m in models], rotation=45)
        
        # Add value labels
        for bar, mean_val in zip(bars, cv_means):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Training Time Analysis
        colors = sns.color_palette("viridis", len(models))
        bars = axes[0, 1].bar(x_pos, training_times, color=colors, alpha=0.8)
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Training Time (seconds)')
        axes[0, 1].set_title('Training Efficiency')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([m[:10] + '...' if len(m) > 10 else m for m in models], rotation=45)
        
        # 3. Performance vs Time Scatter
        scatter = axes[1, 0].scatter(training_times, cv_means, s=100, alpha=0.7, 
                                   c=range(len(models)), cmap='viridis')
        axes[1, 0].set_xlabel('Training Time (seconds)')
        axes[1, 0].set_ylabel('CV Score')
        axes[1, 0].set_title('Performance vs Efficiency')
        
        # Add model labels
        for i, model in enumerate(models):
            axes[1, 0].annotate(model[:8], (training_times[i], cv_means[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Category Performance
        category_performance = {}
        for model, results in comparison_results.items():
            cat = results['category']
            if cat not in category_performance:
                category_performance[cat] = []
            category_performance[cat].append(results['cv_mean'])
        
        categories_list = list(category_performance.keys())
        category_means = [np.mean(scores) for scores in category_performance.values()]
        
        bars = axes[1, 1].bar(categories_list, category_means, alpha=0.8)
        axes[1, 1].set_xlabel('Model Category')
        axes[1, 1].set_ylabel('Average CV Score')
        axes[1, 1].set_title('Performance by Category')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        
        plt.show()

# Initialize visualizer
visualizer = WorkshopVisualizer()

# =============================================================================
# PRODUCTION ML PIPELINE
# =============================================================================

class ProductionMLPipeline:
    """Production-ready ML pipeline with monitoring and deployment features."""
    
    def __init__(self, model_name="production_model", version="1.0.0"):
        self.model_name = model_name
        self.version = version
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_metadata = {}
        self.performance_log = []
        
        print(f"🚀 Production Pipeline Initialized: {model_name} v{version}")
    
    def train_production_model(self, X_train, y_train, algorithm='random_forest', 
                              hyperparameters=None, validation_split=0.2):
        """Train production model with validation."""
        print(f"🏋️ Training Production Model: {algorithm}")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        # Split for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42,
            stratify=y_train if len(np.unique(y_train)) > 1 and len(np.unique(y_train)) < 20 else None
        )
        
        # Preprocessing
        self.scaler = StandardScaler()
        
        # Determine if scaling needed
        needs_scaling = algorithm not in ['random_forest', 'decision_tree', 'gradient_boosting']
        
        if needs_scaling:
            X_train_processed = self.scaler.fit_transform(X_train_split)
            X_val_processed = self.scaler.transform(X_val)
        else:
            X_train_processed = X_train_split.values if hasattr(X_train_split, 'values') else X_train_split
            X_val_processed = X_val.values if hasattr(X_val, 'values') else X_val
        
        # Define model
        model_configs = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True)
        }
        
        if algorithm not in model_configs:
            raise ValueError(f"Algorithm {algorithm} not supported")
        
        self.model = model_configs[algorithm]
        
        # Apply hyperparameters if provided
        if hyperparameters:
            self.model.set_params(**hyperparameters)
            print(f"   ⚙️ Applied hyperparameters: {hyperparameters}")
        
        # Train model
        start_time = time.time()
        self.model.fit(X_train_processed, y_train_split)
        training_time = time.time() - start_time
        
        # Validation
        y_val_pred = self.model.predict(X_val_processed)
        
        # Determine problem type and calculate metrics
        if len(np.unique(y_train)) > 10:  # Regression
            val_score = r2_score(y_val, y_val_pred)
            metric_name = 'R²'
            problem_type = 'regression'
        else:  # Classification
            val_score = accuracy_score(y_val, y_val_pred)
            metric_name = 'Accuracy'
            problem_type = 'classification'
        
        # Store metadata
        self.model_metadata = {
            'algorithm': algorithm, 'problem_type': problem_type,
            'training_samples': len(X_train_split), 'validation_samples': len(X_val),
            'features': len(X_train.columns) if hasattr(X_train, 'columns') else X_train.shape[1],
            'feature_names': self.feature_names, 'validation_score': val_score,
            'metric_name': metric_name, 'training_time': training_time,
            'hyperparameters': hyperparameters or {}, 'needs_scaling': needs_scaling,
            'trained_at': datetime.now().isoformat(), 'version': self.version
        }
        
        print(f"   ✅ Training complete: {metric_name} = {val_score:.3f}")
        print(f"   ⏱️ Training time: {training_time:.3f}s")
        print(f"   📊 Features: {self.model_metadata['features']}")
        
        return self.model_metadata
    
    def predict(self, X, log_prediction=True, confidence_threshold=None):
        """Make predictions with optional confidence filtering."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_production_model first.")
        
        # Preprocess input
        if self.model_metadata['needs_scaling']:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X.values if hasattr(X, 'values') else X
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Get prediction confidence if available
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_processed)
            confidence = np.max(proba, axis=1)
            
            # Apply confidence threshold if specified
            if confidence_threshold is not None:
                low_confidence_mask = confidence < confidence_threshold
                predictions[low_confidence_mask] = -1  # Flag low confidence predictions
        
        # Log prediction if enabled
        if log_prediction:
            self.performance_log.append({
                'timestamp': datetime.now().isoformat(),
                'n_predictions': len(predictions),
                'avg_confidence': confidence.mean() if confidence is not None else None,
                'low_confidence_count': np.sum(predictions == -1) if confidence_threshold else 0
            })
        
        result = {
            'predictions': predictions,
            'confidence': confidence,
            'model_version': self.version
        }
        
        return result
    
    def generate_model_card(self):
        """Generate model card for documentation."""
        if self.model is None:
            return "No model available for documentation."
        
        card = f"""
# Model Card: {self.model_name} v{self.version}

## Model Details
- **Algorithm**: {self.model_metadata['algorithm']}
- **Problem Type**: {self.model_metadata['problem_type']}
- **Trained**: {self.model_metadata['trained_at']}
- **Version**: {self.version}

## Performance
- **Validation {self.model_metadata['metric_name']}**: {self.model_metadata['validation_score']:.3f}
- **Training Time**: {self.model_metadata['training_time']:.3f} seconds
- **Training Samples**: {self.model_metadata['training_samples']:,}
- **Validation Samples**: {self.model_metadata['validation_samples']:,}

## Features
- **Feature Count**: {self.model_metadata['features']}
- **Requires Scaling**: {self.model_metadata['needs_scaling']}

## Usage
```python
# Make predictions
result = model.predict(X_new)
predictions = result['predictions']
confidence = result['confidence']
```

## Monitoring
- **Prediction Logs**: {len(self.performance_log)} entries
- **Last Used**: {self.performance_log[-1]['timestamp'] if self.performance_log else 'Never'}

## Considerations
- Monitor prediction confidence scores
- Retrain when performance degrades
- Validate input data before predictions
        """
        
        return card

print("✅ Production ML Pipeline Ready!")

# =============================================================================
# COMPLETE WORKSHOP DEMONSTRATION
# =============================================================================

def run_complete_workshop_demo(dataset_name='wine'):
    """Run complete workshop demonstration."""
    print("🎯 COMPLETE WORKSHOP DEMONSTRATION")
    print("=" * 35)
    print(f"Dataset: {dataset_name}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    
    # Step 1: Dataset Analysis
    print("\n📊 Step 1: Dataset Analysis")
    print("-" * 27)
    
    dataset_info = dataset_manager.get_dataset_info(dataset_name)
    print(f"✅ Dataset: {dataset_info['type']} problem")
    print(f"✅ Shape: {dataset_info['shape']}")
    print(f"✅ Features: {dataset_info['n_features']}")
    
    if dataset_info['type'] == 'classification':
        print(f"✅ Classes: {dataset_info['n_classes']}")
        print(f"✅ Balanced: {dataset_info['balanced']}")
    
    # Step 2: Advanced Model Comparison
    print("\n🔬 Step 2: Advanced Model Comparison")
    print("-" * 35)
    
    comparison = AdvancedModelComparison(dataset_name, test_size=0.2, cv_folds=5)
    results = comparison.run_comparison()
    
    # Step 3: Statistical Analysis
    print("\n📊 Step 3: Statistical Significance Testing")
    print("-" * 40)
    
    significance_results = comparison.statistical_significance_testing()
    
    # Step 4: Model Interpretation
    print("\n🧠 Step 4: Model Interpretation")
    print("-" * 30)
    
    # Get best model for detailed interpretation
    best_model = max(results.items(), key=lambda x: x[1]['cv_mean'])
    best_name, best_results = best_model
    
    print(f"🏆 Best Model: {best_name} (CV: {best_results['cv_mean']:.3f})")
    
    # Single model interpretation
    interpretation = model_interpreter.interpret_single_model(
        best_name, best_results, dataset_name, dataset_info['features']
    )
    print("\n" + interpretation)
    
    # Step 5: Comparison Analysis
    print("\n🔍 Step 5: Comparison Analysis")
    print("-" * 29)
    
    comparison_interpretation = model_interpreter.interpret_model_comparison(
        results, dataset_name
    )
    print("\n" + comparison_interpretation)
    
    # Step 6: Generate Final Report
    print("\n📋 Step 6: Final Report")
    print("-" * 21)
    
    report = comparison.generate_report()
    
    # Step 7: Visualization
    print("\n📊 Step 7: Visualizations")
    print("-" * 25)
    
    visualizer.plot_model_comparison(results, dataset_name)
    
    # Final Summary
    print("\n🎉 WORKSHOP DEMONSTRATION COMPLETE!")
    print("=" * 37)
    
    summary = {
        'dataset': dataset_name,
        'dataset_info': dataset_info,
        'best_model': best_name,
        'best_score': best_results['cv_mean'],
        'models_tested': len(results),
        'significant_differences': sum(1 for r in significance_results if r['significant']),
        'recommendations': report['recommendations'],
        'completed_at': datetime.now().strftime('%H:%M:%S')
    }
    
    print(f"\n📊 Final Summary:")
    print(f"   🗃️ Dataset: {summary['dataset']} ({summary['dataset_info']['type']})")
    print(f"   🏆 Best Model: {summary['best_model']} ({summary['best_score']:.3f})")
    print(f"   🤖 Models Tested: {summary['models_tested']}")
    print(f"   📊 Significant Differences: {summary['significant_differences']}")
    print(f"   💡 Recommendations: {len(summary['recommendations'])}")
    print(f"   ⏰ Completed: {summary['completed_at']}")
    
    return summary

print("✅ Complete Workshop Demo Ready!")

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

print("\n" + "="*60)
print("WEEK 3 COMPLETE WORKSHOP SOLUTIONS - READY FOR USE")
print("="*60)

print("\n💡 Usage Examples:")
print("# Run complete demo:")
print("summary = run_complete_workshop_demo('wine')")
print("")
print("# Advanced comparison:")
print("comparison = AdvancedModelComparison('cancer')")
print("results = comparison.run_comparison()")
print("significance = comparison.statistical_significance_testing()")
print("")
print("# Production pipeline:")
print("pipeline = ProductionMLPipeline('classifier', '1.0.0')")
print("X_train, X_test, y_train, y_test = dataset_manager.get_dataset('wine')")
print("metadata = pipeline.train_production_model(X_train, y_train)")
print("result = pipeline.predict(X_test)")

print("\n🚀 All systems ready for Week 3 workshop execution!")
