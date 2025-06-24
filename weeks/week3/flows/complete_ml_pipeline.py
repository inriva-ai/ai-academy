from metaflow import FlowSpec, step, Parameter, resources
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

# Complete Metaflow ML Pipeline
class SupervisedLearningFlow(FlowSpec):
    """
    Complete supervised learning pipeline with parallel model training.
    """
    
    dataset_type = Parameter('dataset_type', help='Type of dataset: wine or housing', default='wine')
    test_size = Parameter('test_size', help='Test set size (0.0-1.0)', default=0.2)
    n_cv_folds = Parameter('n_cv_folds', help='Number of cross-validation folds', default=5)
    
    @step
    def start(self):
        """Load and prepare the dataset for training."""
        print(f"🚀 Starting Supervised Learning Pipeline")
        print(f"   Dataset: {self.dataset_type}")
        
        if self.dataset_type == 'wine':
            wine_data = load_wine()
            self.X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
            self.y = wine_data.target
            self.target_names = wine_data.target_names
            self.problem_type = 'classification'
            
            self.algorithms = {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'svm': SVC(random_state=42, probability=True),
                'naive_bayes': GaussianNB()
            }
        else:  # housing dataset
            self.X = X_reg
            self.y = y_reg
            self.target_names = ['price']
            self.problem_type = 'regression'
            
            self.algorithms = {
                'linear_regression': LinearRegression(),
                'ridge_regression': Ridge(alpha=1.0, random_state=42),
                'lasso_regression': Lasso(alpha=1.0, random_state=42),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'svr': SVR(kernel='rbf')
            }
        
        print(f"📊 Dataset shape: {self.X.shape}")
        print(f"🎯 Problem type: {self.problem_type}")
        
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        """Preprocess the data: split and scale."""
        print("🔧 Preprocessing data...")
        
        if self.problem_type == 'classification':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, 
                random_state=42, stratify=self.y
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=42
            )
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   📊 Training set: {self.X_train_scaled.shape}")
        print(f"   📊 Test set: {self.X_test_scaled.shape}")
        
        self.algorithm_names = list(self.algorithms.keys())
        self.next(self.train_model, foreach='algorithm_names')
    
    @resources(memory=2000, cpu=2)
    @step
    def train_model(self):
        """Train individual models in parallel."""
        self.current_algorithm = self.input
        algorithm = self.algorithms[self.current_algorithm]
        
        print(f"🏋️ Training {self.current_algorithm}...")
        
        start_time = datetime.now()
        algorithm.fit(self.X_train_scaled, self.y_train)
        y_pred = algorithm.predict(self.X_test_scaled)
        training_time = (datetime.now() - start_time).total_seconds()
        
        if self.problem_type == 'classification':
            accuracy = accuracy_score(self.y_test, y_pred)
            cv_scores = cross_val_score(algorithm, self.X_train_scaled, self.y_train, 
                                       cv=self.n_cv_folds, scoring='accuracy')
            
            self.model_results = {
                'algorithm': self.current_algorithm,
                'model': algorithm,
                'predictions': y_pred,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time
            }
            
            print(f"   ✅ {self.current_algorithm}: Accuracy={accuracy:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            
        else:  # regression
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            cv_scores = cross_val_score(algorithm, self.X_train_scaled, self.y_train, 
                                       cv=self.n_cv_folds, scoring='r2')
            
            self.model_results = {
                'algorithm': self.current_algorithm,
                'model': algorithm,
                'predictions': y_pred,
                'mse': mse,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time
            }
            
            print(f"   ✅ {self.current_algorithm}: R²={r2:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        self.next(self.aggregate_results)
    
    @step
    def aggregate_results(self, inputs):
        """Aggregate results from all parallel training tasks."""
        print("🔄 Aggregating results from parallel training...")

        # Merge artifacts from all parallel steps
        merged_algorithms = {}
        for input in inputs:
            for key, model in input.algorithms.items():
                # If the key is new, add it
                if key not in merged_algorithms:
                    merged_algorithms[key] = model
                else:
                    # Optional: check that models are consistent across inputs
                    if str(merged_algorithms[key]) != str(model):
                        raise ValueError(f"Conflict in model definition for '{key}'")

        self.algorithms = merged_algorithms
        self.merge_artifacts(inputs, exclude=['algorithms', 'model_results', 'current_algorithm'])

        self.all_model_results = {}
        self.training_summary = []


        for input_flow in inputs:
            algorithm = input_flow.current_algorithm
            results = input_flow.model_results
            self.all_model_results[algorithm] = results
            self.training_summary.append(results)
        
        print(f"✅ Aggregated results from {len(self.all_model_results)} models")
        self.next(self.model_selection)
    
    @step
    def model_selection(self):
        """Select the best model based on cross-validation performance."""
        print("🏆 Selecting best model...")
        
        best_algorithm = max(self.all_model_results.keys(), 
                           key=lambda x: self.all_model_results[x]['cv_mean'])
        best_score = self.all_model_results[best_algorithm]['cv_mean']
        
        self.best_model = {
            'algorithm': best_algorithm,
            'results': self.all_model_results[best_algorithm],
            'score': best_score
        }
        
        print(f"🏆 Best Model: {best_algorithm}")
        print(f"📊 Best Score: {best_score:.3f}")
        
        self.next(self.hyperparameter_tuning)
    
    @step
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model."""
        print("⚙️ Hyperparameter tuning for best model...")
        
        best_algorithm = self.best_model['algorithm']
        print(f"🔧 Tuning {best_algorithm}...")
        
        param_grids = {
            'logistic_regression': {'C': [0.1, 1.0, 10.0], 'solver': ['liblinear', 'lbfgs']},
            'random_forest': {'n_estimators': [50, 100], 'max_depth': [None, 10], 'min_samples_split': [2, 5]},
            'gradient_boosting': {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2], 'max_depth': [3, 5]},
            'svm': {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']},
            'ridge_regression': {'alpha': [0.1, 1.0, 10.0]},
            'lasso_regression': {'alpha': [0.1, 1.0, 10.0]},
            'svr': {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']}
        }
        
        param_grid = param_grids.get(best_algorithm, {})
        
        if param_grid:
            base_model = self.algorithms[best_algorithm]
            scoring = 'accuracy' if self.problem_type == 'classification' else 'r2'
            
            grid_search = GridSearchCV(
                base_model, param_grid, cv=self.n_cv_folds, 
                scoring=scoring, n_jobs=-1
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            self.tuned_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.tuning_score = grid_search.best_score_
            
            print(f"✅ Best parameters: {self.best_params}")
            print(f"📈 Tuned CV score: {self.tuning_score:.3f}")
        else:
            print("ℹ️ No hyperparameters to tune for this algorithm")
            self.tuned_model = self.all_model_results[best_algorithm]['model']
            self.best_params = {}
            self.tuning_score = self.best_model['score']
        
        self.next(self.final_evaluation)
    
    @step
    def final_evaluation(self):
        """Final evaluation with tuned model."""
        print("📊 Final Model Evaluation")
        
        y_pred_final = self.tuned_model.predict(self.X_test_scaled)
        
        if self.problem_type == 'classification':
            final_accuracy = accuracy_score(self.y_test, y_pred_final)
            conf_matrix = confusion_matrix(self.y_test, y_pred_final)
            
            self.final_results = {
                'accuracy': final_accuracy,
                'confusion_matrix': conf_matrix.tolist(),
                'predictions': y_pred_final.tolist(),
                'actual': self.y_test.tolist()
            }
            
            print(f"🎯 Final Accuracy: {final_accuracy:.3f}")
        else:
            final_mse = mean_squared_error(self.y_test, y_pred_final)
            final_r2 = r2_score(self.y_test, y_pred_final)
            
            self.final_results = {
                'mse': final_mse,
                'r2': final_r2,
                'rmse': np.sqrt(final_mse),
                'predictions': y_pred_final.tolist(),
                'actual': self.y_test.tolist()
            }
            
            print(f"🎯 Final R²: {final_r2:.3f}")
        
        self.next(self.generate_insights)
    
    @step  
    def generate_insights(self):
        """Generate insights and model interpretation."""
        print("💡 Generating Model Insights")
        
        insights = []
        
        # Feature importance (if available)
        if hasattr(self.tuned_model, 'feature_importances_'):
            feature_importance = dict(zip(self.X.columns, self.tuned_model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            insights.append("🔍 Top 5 Most Important Features:")
            for i, (feature, importance) in enumerate(top_features, 1):
                insights.append(f"   {i}. {feature}: {importance:.3f}")
        
        # Performance insights
        if self.problem_type == 'classification':
            accuracy = self.final_results['accuracy']
            if accuracy > 0.95:
                insights.append("🏆 Excellent model performance - ready for production!")
            elif accuracy > 0.90:
                insights.append("✅ Very good performance - minor optimizations possible")
            elif accuracy > 0.80:
                insights.append("⚠️ Good performance - consider feature engineering")
            else:
                insights.append("❌ Performance needs improvement")
        else:
            r2 = self.final_results['r2']
            if r2 > 0.90:
                insights.append("🏆 Excellent predictive power!")
            elif r2 > 0.75:
                insights.append("✅ Good predictive performance")
            else:
                insights.append("⚠️ Moderate predictive power - room for improvement")
        
        # Training efficiency insights
        fastest_model = min(self.training_summary, key=lambda x: x['training_time'])
        insights.append(f"⚡ Fastest training: {fastest_model['algorithm']} ({fastest_model['training_time']:.2f}s)")
        
        best_cv = max(self.training_summary, key=lambda x: x['cv_mean'])
        insights.append(f"🎯 Best cross-validation: {best_cv['algorithm']} ({best_cv['cv_mean']:.3f})")
        
        self.model_insights = insights
        
        for insight in insights:
            print(insight)
        
        self.next(self.end)
    
    @step
    def end(self):
        """Complete the pipeline and generate final summary."""
        print("\n🎉 SUPERVISED LEARNING PIPELINE COMPLETE!")
        
        summary = {
            'dataset_info': {
                'type': self.dataset_type,
                'shape': f"{self.X.shape[0]} samples × {self.X.shape[1]} features",
                'problem_type': self.problem_type
            },
            'training_summary': {
                'algorithms_tested': len(self.all_model_results),
                'best_algorithm': self.best_model['algorithm'],
                'hyperparameter_tuning': bool(self.best_params),
                'total_training_time': sum(result['training_time'] for result in self.all_model_results.values())
            },
            'performance': self.final_results,
            'insights': self.model_insights
        }
        
        self.pipeline_summary = summary
        
        print("📊 Pipeline Summary:")
        print(f"   🗃️ Dataset: {summary['dataset_info']['shape']}")
        print(f"   🤖 Algorithms tested: {summary['training_summary']['algorithms_tested']}")
        print(f"   🏆 Best model: {summary['training_summary']['best_algorithm']}")
        print(f"   ⚙️ Hyperparameter tuning: {'Yes' if summary['training_summary']['hyperparameter_tuning'] else 'No'}")
        
        if self.problem_type == 'classification':
            print(f"   🎯 Final accuracy: {self.final_results['accuracy']:.3f}")
        else:
            print(f"   🎯 Final R²: {self.final_results['r2']:.3f}")
        
        print("\n✨ All results and models saved by Metaflow!")
        print("💾 Access results using: flow.run.data")
        print("🔄 Reproduce anytime: python flow.py run")

if __name__ == '__main__':
    SupervisedLearningFlow()