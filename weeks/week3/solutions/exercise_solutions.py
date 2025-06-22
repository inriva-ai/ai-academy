"""
Week 3: Exercise Solutions - Supervised Learning with Metaflow Pipelines

Complete solutions to all Week 3 exercises including:
1. Classification challenges
2. Regression practice
3. Pipeline optimization
4. Hybrid evaluation exercises

Usage:
    python exercise_solutions.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine, load_breast_cancer, make_regression
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LangChain imports with fallback
try:
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import Ollama
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class Week3Solutions:
    """
    Complete solutions for Week 3 supervised learning exercises.
    """
    
    def __init__(self):
        self.results = {}
        print("🎓 Week 3 Solutions Initialized")
        print(f"   LangChain Available: {LANGCHAIN_AVAILABLE}")
    
    def exercise_1_classification_challenges(self):
        """
        SOLUTION 1: Multi-class Classification with Advanced Techniques
        
        Challenge: Implement and compare multiple classification algorithms
        with advanced evaluation metrics and visualization.
        """
        print("\\n🎯 Exercise 1: Classification Challenges")
        print("=" * 45)
        
        # Load multiple datasets for comparison
        datasets = {
            'Wine': load_wine(),
            'Breast Cancer': load_breast_cancer()
        }
        
        # Advanced classification algorithms
        classifiers = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
            'SVM (Poly)': SVC(kernel='poly', degree=3, random_state=42, probability=True),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Bagging Classifier': BaggingClassifier(random_state=42),
        }
        
        results = {}
        
        for dataset_name, dataset in datasets.items():
            print(f"\\n📊 Dataset: {dataset_name}")
            print(f"   Shape: {dataset.data.shape}")
            print(f"   Classes: {len(dataset.target_names)}")
            
            X = dataset.data
            y = dataset.target
            
            # Split and scale
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            dataset_results = {}
            
            for clf_name, classifier in classifiers.items():
                start_time = datetime.now()
                
                # Train model
                classifier.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = classifier.predict(X_test_scaled)
                y_pred_proba = classifier.predict_proba(X_test_scaled)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=5)
                
                # AUC Score (multi-class)
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                dataset_results[clf_name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'auc_score': auc_score,
                    'training_time': training_time,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"   {clf_name:20}: Acc={accuracy:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}, AUC={auc_score:.3f}")
            
            results[dataset_name] = {
                'X_test': X_test_scaled,
                'y_test': y_test,
                'results': dataset_results,
                'target_names': dataset.target_names
            }
        
        # Advanced ensemble method
        print("\\n🤖 Creating Advanced Ensemble...")
        wine_data = datasets['Wine']
        X = wine_data.data
        y = wine_data.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create voting classifier with best performers
        voting_clf = VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(kernel='rbf', random_state=42, probability=True))
        ], voting='soft')
        
        voting_clf.fit(X_train_scaled, y_train)
        ensemble_pred = voting_clf.predict(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        print(f"   🏆 Ensemble Accuracy: {ensemble_accuracy:.3f}")
        
        self.results['exercise_1'] = results
        print("\\n✅ Exercise 1 Complete!")
        
        return results
    
    def exercise_2_regression_practice(self):
        """
        SOLUTION 2: Advanced Regression Techniques
        
        Challenge: Implement comprehensive regression pipeline with
        feature selection, polynomial features, and model comparison.
        """
        print("\\n📈 Exercise 2: Regression Practice")
        print("=" * 35)
        
        # Create complex regression dataset
        X_reg, y_reg = make_regression(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            noise=0.1,
            random_state=42
        )
        
        # Add feature names
        feature_names = [f'feature_{i:02d}' for i in range(X_reg.shape[1])]
        X_reg_df = pd.DataFrame(X_reg, columns=feature_names)
        
        print(f"📊 Regression Dataset: {X_reg_df.shape}")
        print(f"   Features: {X_reg_df.shape[1]}")
        print(f"   Target range: [{y_reg.min():.2f}, {y_reg.max():.2f}]")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg_df, y_reg, test_size=0.2, random_state=42
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Regression algorithms with hyperparameters
        regressors = {
            'Linear Regression': Ridge(alpha=1.0),
            'Ridge (α=0.1)': Ridge(alpha=0.1),
            'Ridge (α=10)': Ridge(alpha=10.0),
            'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=2000),
            'Lasso (α=1.0)': Lasso(alpha=1.0, max_iter=2000),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        regression_results = {}
        
        print("\\n🏋️ Training Regression Models:")
        for name, regressor in regressors.items():
            start_time = datetime.now()
            
            # Train model
            regressor.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = regressor.predict(X_test_scaled)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_r2_scores = cross_val_score(regressor, X_train_scaled, y_train, cv=5, scoring='r2')
            cv_mse_scores = -cross_val_score(regressor, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            regression_results[name] = {
                'model': regressor,
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'cv_r2_mean': cv_r2_scores.mean(),
                'cv_r2_std': cv_r2_scores.std(),
                'cv_mse_mean': cv_mse_scores.mean(),
                'training_time': training_time,
                'predictions': y_pred
            }
            
            print(f"   {name:20}: R²={r2:.3f}, RMSE={rmse:.2f}, CV-R²={cv_r2_scores.mean():.3f}±{cv_r2_scores.std():.3f}")
        
        # Feature selection analysis
        print("\\n🔍 Feature Selection Analysis:")
        
        # Univariate feature selection
        selector = SelectKBest(score_func=f_classif, k=10)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
        print(f"   📈 Top 10 features: {selected_features[:5]}...")
        
        # Train model on selected features
        rf_selected = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_selected.fit(X_train_selected, y_train)
        y_pred_selected = rf_selected.predict(X_test_selected)
        r2_selected = r2_score(y_test, y_pred_selected)
        
        print(f"   🎯 RF with feature selection: R²={r2_selected:.3f}")
        
        # Recursive Feature Elimination
        rf_base = RandomForestRegressor(n_estimators=50, random_state=42)
        rfe = RFE(rf_base, n_features_to_select=10, step=1)
        rfe.fit(X_train_scaled, y_train)
        
        rfe_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
        print(f"   🔄 RFE selected features: {rfe_features[:5]}...")
        
        # Learning curve analysis
        print("\\n📚 Learning Curve Analysis:")
        best_model = max(regression_results.items(), key=lambda x: x[1]['r2'])
        best_regressor = best_model[1]['model']
        
        train_sizes, train_scores, val_scores = learning_curve(
            best_regressor, X_train_scaled, y_train, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
        )
        
        print(f"   📊 Learning curve computed for {best_model[0]}")
        print(f"   📈 Training score at 100%: {train_scores[-1].mean():.3f}±{train_scores[-1].std():.3f}")
        print(f"   📉 Validation score at 100%: {val_scores[-1].mean():.3f}±{val_scores[-1].std():.3f}")
        
        self.results['exercise_2'] = {
            'regression_results': regression_results,
            'feature_selection': {
                'selected_features': selected_features,
                'rfe_features': rfe_features,
                'selected_r2': r2_selected
            },
            'learning_curve': {
                'train_sizes': train_sizes.tolist(),
                'train_scores': train_scores.tolist(),
                'val_scores': val_scores.tolist()
            }
        }
        
        print("\\n✅ Exercise 2 Complete!")
        return regression_results
    
    def exercise_3_pipeline_optimization(self):
        """
        SOLUTION 3: Metaflow Pipeline Optimization
        
        Challenge: Create optimized Metaflow pipeline with advanced
        parallel processing and resource management.
        """
        print("\\n⚙️ Exercise 3: Pipeline Optimization")
        print("=" * 38)
        
        # Demonstrate advanced Metaflow patterns
        print("🌊 Advanced Metaflow Pipeline Patterns:")
        
        # Pattern 1: Nested parallel execution
        print("\\n📋 Pattern 1: Nested Parallel Execution")
        pipeline_structure = {
            'datasets': ['wine', 'breast_cancer', 'iris'],
            'algorithms': ['rf', 'gb', 'svm', 'lr'],
            'hyperparameters': {
                'rf': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
                'gb': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2]},
                'svm': {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'poly']},
                'lr': {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}
            }
        }
        
        total_combinations = 0
        for dataset in pipeline_structure['datasets']:
            for algorithm in pipeline_structure['algorithms']:
                params = pipeline_structure['hyperparameters'][algorithm]
                combinations = np.prod([len(param_values) for param_values in params.values()])
                total_combinations += combinations
                print(f"   {dataset} + {algorithm}: {combinations} parameter combinations")
        
        print(f"   🔢 Total parameter combinations: {total_combinations}")
        print(f"   ⚡ Parallel speedup potential: {total_combinations // 4}x (4 cores)")
        
        # Pattern 2: Resource optimization
        print("\\n💾 Pattern 2: Resource Optimization")
        resource_requirements = {
            'light_models': {'memory': 2000, 'cpu': 1},  # Logistic Regression, Naive Bayes
            'medium_models': {'memory': 4000, 'cpu': 2},  # SVM, KNN
            'heavy_models': {'memory': 8000, 'cpu': 4},   # Random Forest, Gradient Boosting
            'ensemble_models': {'memory': 16000, 'cpu': 8}  # Voting, Stacking
        }
        
        for model_type, resources in resource_requirements.items():
            print(f"   {model_type:15}: Memory={resources['memory']}MB, CPU={resources['cpu']} cores")
        
        # Pattern 3: Error handling and recovery
        print("\\n🛡️ Pattern 3: Error Handling Strategies")
        error_strategies = {
            'Data Loading': 'Fallback to synthetic data generation',
            'Model Training': 'Skip failed models, continue with successful ones',
            'Hyperparameter Tuning': 'Use default parameters if tuning fails',
            'LLM Integration': 'Fallback to rule-based interpretation',
            'Result Aggregation': 'Handle partial results gracefully'
        }
        
        for stage, strategy in error_strategies.items():
            print(f"   {stage:20}: {strategy}")
        
        # Pattern 4: Caching and artifact management
        print("\\n💽 Pattern 4: Caching and Artifact Management")
        caching_strategies = {
            'Preprocessed Data': 'Cache scaled features to avoid recomputation',
            'Trained Models': 'Store model objects for reuse and comparison',
            'CV Results': 'Cache cross-validation scores for quick access',
            'Feature Importance': 'Store feature rankings for interpretation',
            'Evaluation Metrics': 'Cache detailed metrics for reporting'
        }
        
        for artifact_type, strategy in caching_strategies.items():
            print(f"   {artifact_type:20}: {strategy}")
        
        # Optimization recommendations
        print("\\n🚀 Pipeline Optimization Recommendations:")
        optimizations = [
            "Use @batch decorator for cloud scaling",
            "Implement @conda decorator for environment isolation",
            "Add @timeout decorator for long-running tasks",
            "Use @retry decorator for transient failures",
            "Implement custom @card decorators for rich reporting",
            "Add parameter validation in start() step",
            "Use foreach with dynamic branching for flexibility",
            "Implement progress tracking with custom logging",
            "Add memory profiling for resource optimization",
            "Use @environment decorator for reproducible runs"
        ]
        
        for i, optimization in enumerate(optimizations, 1):
            print(f"   {i:2d}. {optimization}")
        
        # Sample optimized flow structure
        print("\\n📝 Sample Optimized Flow Structure:")
        flow_structure = """
        class OptimizedMLFlow(FlowSpec):
            @step
            def start(self):
                # Parameter validation and environment setup
                
            @step  
            def load_and_validate_data(self):
                # Data loading with fallback strategies
                
            @step
            def preprocess_data(self):
                # Efficient preprocessing with caching
                
            @batch(memory=4000, cpu=2)
            @retry(times=3)
            @step
            def train_model(self):
                # Parallel model training with error handling
                
            @resources(memory=8000, cpu=4)
            @timeout(seconds=1800)
            @step
            def hyperparameter_tuning(self):
                # Resource-intensive tuning with timeout
                
            @step
            def evaluate_models(self, inputs):
                # Robust result aggregation
                
            @conda(libraries={'langchain': '0.1.0'})
            @catch(var='llm_error')
            @step
            def llm_interpretation(self):
                # LLM integration with fallback
                
            @step
            def generate_report(self):
                # Comprehensive reporting
        """
        
        print(flow_structure)
        
        self.results['exercise_3'] = {
            'pipeline_structure': pipeline_structure,
            'resource_requirements': resource_requirements,
            'error_strategies': error_strategies,
            'caching_strategies': caching_strategies,
            'optimizations': optimizations,
            'total_combinations': total_combinations
        }
        
        print("\\n✅ Exercise 3 Complete!")
        
    def exercise_4_hybrid_evaluation(self):
        """
        SOLUTION 4: Advanced Hybrid ML + LLM Evaluation
        
        Challenge: Create sophisticated hybrid evaluation system
        with multiple LLM integration patterns.
        """
        print("\\n🤖 Exercise 4: Hybrid Evaluation")
        print("=" * 35)
        
        # Load dataset for hybrid evaluation
        wine_data = load_wine()
        X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
        y = wine_data.target
        target_names = wine_data.target_names
        
        # Advanced model selection
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost-style GB': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'Optimized SVM': SVC(C=10.0, kernel='rbf', gamma='scale', random_state=42, probability=True),
            'Ensemble': VotingClassifier([
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                ('svm', SVC(C=1.0, kernel='rbf', probability=True, random_state=42))
            ], voting='soft')
        }
        
        # Train and evaluate models
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model_results = {}
        
        print("\\n🏋️ Training Advanced Models:")
        for name, model in models.items():
            start_time = datetime.now()
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': class_report,
                'training_time': training_time,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"   {name:15}: Acc={accuracy:.3f}, AUC={auc:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        # Multi-level LLM integration
        print("\\n🦜 Multi-Level LLM Integration:")
        
        llm_integration_levels = {
            'Level 1': 'Basic performance interpretation',
            'Level 2': 'Comparative model analysis', 
            'Level 3': 'Business impact assessment',
            'Level 4': 'Strategic recommendations',
            'Level 5': 'Risk analysis and mitigation'
        }
        
        for level, description in llm_integration_levels.items():
            print(f"   {level}: {description}")
        
        # Generate interpretations
        interpretations = {}
        
        if LANGCHAIN_AVAILABLE:
            print("\\n🔍 Generating LLM Interpretations:")
            try:
                llm = Ollama(model="llama3.2")
                
                # Level 1: Basic interpretation
                for model_name, results in model_results.items():
                    interpretation = self.generate_advanced_interpretation(
                        llm, model_name, results, target_names
                    )
                    interpretations[model_name] = interpretation
                    print(f"   ✅ {model_name} interpretation generated")
                
                # Level 2: Comparative analysis
                comparative_analysis = self.generate_comparative_analysis(llm, model_results)
                interpretations['comparative_analysis'] = comparative_analysis
                print("   ✅ Comparative analysis generated")
                
                # Level 3: Business impact
                business_impact = self.generate_business_impact_analysis(llm, model_results)
                interpretations['business_impact'] = business_impact
                print("   ✅ Business impact analysis generated")
                
            except Exception as e:
                print(f"   ⚠️ LLM integration failed: {e}")
                interpretations = self.generate_fallback_interpretations(model_results)
        else:
            print("   📝 Using fallback interpretations")
            interpretations = self.generate_fallback_interpretations(model_results)
        
        # Advanced evaluation metrics
        print("\\n📊 Advanced Evaluation Metrics:")
        advanced_metrics = self.calculate_advanced_metrics(model_results, y_test)
        
        for metric_name, metric_value in advanced_metrics.items():
            print(f"   {metric_name:25}: {metric_value}")
        
        # Generate comprehensive report
        comprehensive_report = self.generate_comprehensive_report(
            model_results, interpretations, advanced_metrics
        )
        
        self.results['exercise_4'] = {
            'model_results': {name: {k: v for k, v in results.items() if k != 'model'} 
                            for name, results in model_results.items()},
            'interpretations': interpretations,
            'advanced_metrics': advanced_metrics,
            'comprehensive_report': comprehensive_report,
            'llm_integration_levels': llm_integration_levels
        }
        
        print("\\n✅ Exercise 4 Complete!")
        return model_results, interpretations
    
    def generate_advanced_interpretation(self, llm, model_name, results, target_names):
        """Generate advanced LLM interpretation for a model."""
        
        # Prepare detailed context
        accuracy = results['accuracy']
        auc = results['auc']
        cv_mean = results['cv_mean']
        cv_std = results['cv_std']
        class_report = results['classification_report']
        
        # Per-class performance
        class_performance = []
        for i, class_name in enumerate(target_names):
            if str(i) in class_report:
                precision = class_report[str(i)]['precision']
                recall = class_report[str(i)]['recall']
                f1 = class_report[str(i)]['f1-score']
                class_performance.append(f"{class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        class_performance_text = "; ".join(class_performance)
        
        prompt = PromptTemplate(
            input_variables=["model_name", "accuracy", "auc", "cv_mean", "cv_std", "class_performance"],
            template="""
            Provide an advanced analysis of this wine classification model:
            
            Model: {model_name}
            Overall Accuracy: {accuracy:.3f}
            AUC Score: {auc:.3f}
            Cross-validation: {cv_mean:.3f} ± {cv_std:.3f}
            Per-class Performance: {class_performance}
            
            Provide analysis covering:
            1. Technical performance assessment
            2. Model reliability and generalization
            3. Per-class performance insights
            4. Practical deployment considerations
            5. Potential limitations and risks
            
            Focus on actionable insights for ML practitioners.
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        
        interpretation = chain.invoke({
            "model_name": model_name,
            "accuracy": accuracy,
            "auc": auc,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "class_performance": class_performance_text
        })
        
        return interpretation.strip()
    
    def generate_comparative_analysis(self, llm, model_results):
        """Generate comparative analysis across all models."""
        
        # Prepare comparison data
        model_summary = []
        for name, results in model_results.items():
            model_summary.append(
                f"{name}: Accuracy={results['accuracy']:.3f}, "
                f"AUC={results['auc']:.3f}, "
                f"CV={results['cv_mean']:.3f}±{results['cv_std']:.3f}, "
                f"Time={results['training_time']:.2f}s"
            )
        
        models_text = "; ".join(model_summary)
        
        # Find best and worst performers
        best_model = max(model_results.items(), key=lambda x: x[1]['accuracy'])
        worst_model = min(model_results.items(), key=lambda x: x[1]['accuracy'])
        
        prompt = PromptTemplate(
            input_variables=["models_summary", "best_model", "worst_model"],
            template="""
            Compare these wine classification models:
            
            {models_summary}
            
            Best performer: {best_model}
            Lowest performer: {worst_model}
            
            Provide strategic comparison covering:
            1. Performance landscape and gaps
            2. Trade-offs between models (accuracy vs speed vs interpretability)
            3. Model selection recommendations for different scenarios
            4. Ensemble opportunities
            5. Resource efficiency considerations
            
            Focus on strategic decision-making for model deployment.
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        
        analysis = chain.invoke({
            "models_summary": models_text,
            "best_model": f"{best_model[0]} ({best_model[1]['accuracy']:.3f})",
            "worst_model": f"{worst_model[0]} ({worst_model[1]['accuracy']:.3f})"
        })
        
        return analysis.strip()
    
    def generate_business_impact_analysis(self, llm, model_results):
        """Generate business impact analysis."""
        
        best_model = max(model_results.items(), key=lambda x: x[1]['accuracy'])
        best_accuracy = best_model[1]['accuracy']
        
        prompt = PromptTemplate(
            input_variables=["best_model", "best_accuracy", "model_count"],
            template="""
            Analyze the business impact of this wine classification system:
            
            Best Model: {best_model} with {best_accuracy:.1%} accuracy
            Total Models Evaluated: {model_count}
            
            Provide business impact analysis covering:
            1. ROI potential and cost-benefit analysis
            2. Risk assessment and mitigation strategies
            3. Implementation timeline and resource requirements
            4. Success metrics and KPIs for monitoring
            5. Competitive advantages and market positioning
            
            Focus on executive-level strategic insights and business value.
            """
        )
        
        chain = prompt | llm | StrOutputParser()
        
        analysis = chain.invoke({
            "best_model": best_model[0],
            "best_accuracy": best_accuracy,
            "model_count": len(model_results)
        })
        
        return analysis.strip()
    
    def generate_fallback_interpretations(self, model_results):
        """Generate rule-based interpretations when LLM is not available."""
        
        interpretations = {}
        
        for model_name, results in model_results.items():
            accuracy = results['accuracy']
            cv_std = results['cv_std']
            
            # Performance level
            if accuracy > 0.95:
                perf_level = "Excellent"
            elif accuracy > 0.90:
                perf_level = "Very Good"
            elif accuracy > 0.80:
                perf_level = "Good"
            else:
                perf_level = "Needs Improvement"
            
            # Reliability
            reliability = "High" if cv_std < 0.02 else "Moderate" if cv_std < 0.05 else "Low"
            
            interpretation = f"""
            Performance: {perf_level} ({accuracy:.3f} accuracy)
            Reliability: {reliability} (CV std: {cv_std:.3f})
            Recommendation: {'Production ready' if accuracy > 0.90 and cv_std < 0.03 else 'Needs optimization'}
            """
            
            interpretations[model_name] = interpretation.strip()
        
        return interpretations
    
    def calculate_advanced_metrics(self, model_results, y_test):
        """Calculate advanced evaluation metrics."""
        
        best_model = max(model_results.items(), key=lambda x: x[1]['accuracy'])
        best_results = best_model[1]
        
        metrics = {}
        
        # Model diversity
        predictions = [results['predictions'] for results in model_results.values()]
        prediction_matrix = np.array(predictions)
        
        # Calculate prediction disagreement
        disagreement_rate = np.mean(np.std(prediction_matrix, axis=0) > 0)
        metrics['Prediction Disagreement Rate'] = f"{disagreement_rate:.3f}"
        
        # Performance spread
        accuracies = [results['accuracy'] for results in model_results.values()]
        metrics['Accuracy Range'] = f"{np.max(accuracies) - np.min(accuracies):.3f}"
        metrics['Accuracy Std Dev'] = f"{np.std(accuracies):.3f}"
        
        # Best model confidence
        best_probabilities = best_results['probabilities']
        avg_confidence = np.mean(np.max(best_probabilities, axis=1))
        metrics['Best Model Avg Confidence'] = f"{avg_confidence:.3f}"
        
        # Training efficiency
        training_times = [results['training_time'] for results in model_results.values()]
        metrics['Total Training Time'] = f"{sum(training_times):.2f}s"
        metrics['Avg Training Time'] = f"{np.mean(training_times):.2f}s"
        
        return metrics
    
    def generate_comprehensive_report(self, model_results, interpretations, advanced_metrics):
        """Generate comprehensive evaluation report."""
        
        report_sections = []
        
        # Executive Summary
        best_model = max(model_results.items(), key=lambda x: x[1]['accuracy'])
        best_name, best_results = best_model
        
        report_sections.append("🏆 EXECUTIVE SUMMARY")
        report_sections.append("=" * 20)
        report_sections.append(f"Best Model: {best_name}")
        report_sections.append(f"Accuracy: {best_results['accuracy']:.3f}")
        report_sections.append(f"AUC Score: {best_results['auc']:.3f}")
        report_sections.append(f"Models Evaluated: {len(model_results)}")
        report_sections.append("")
        
        # Performance Rankings
        report_sections.append("📊 PERFORMANCE RANKINGS")
        report_sections.append("-" * 25)
        
        ranked_models = sorted(model_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for i, (name, results) in enumerate(ranked_models, 1):
            report_sections.append(
                f"{i}. {name}: {results['accuracy']:.3f} "
                f"(AUC: {results['auc']:.3f}, Time: {results['training_time']:.2f}s)"
            )
        
        report_sections.append("")
        
        # Advanced Metrics
        report_sections.append("📈 ADVANCED METRICS")
        report_sections.append("-" * 20)
        for metric, value in advanced_metrics.items():
            report_sections.append(f"{metric}: {value}")
        
        report_sections.append("")
        
        # Key Insights
        if 'comparative_analysis' in interpretations:
            report_sections.append("🧠 COMPARATIVE ANALYSIS")
            report_sections.append("-" * 25)
            report_sections.append(interpretations['comparative_analysis'])
            report_sections.append("")
        
        # Business Impact
        if 'business_impact' in interpretations:
            report_sections.append("💼 BUSINESS IMPACT")
            report_sections.append("-" * 18)
            report_sections.append(interpretations['business_impact'])
        
        return "\\n".join(report_sections)
    
    def run_all_exercises(self):
        """Run all Week 3 exercises in sequence."""
        print("🚀 Running All Week 3 Exercises")
        print("=" * 35)
        
        # Run all exercises
        self.exercise_1_classification_challenges()
        self.exercise_2_regression_practice()
        self.exercise_3_pipeline_optimization()
        self.exercise_4_hybrid_evaluation()
        
        # Generate final summary
        print("\\n🎓 WEEK 3 SOLUTIONS SUMMARY")
        print("=" * 30)
        
        print("✅ Exercise 1: Multi-class classification with 8 algorithms")
        print("✅ Exercise 2: Advanced regression with feature selection")
        print("✅ Exercise 3: Optimized Metaflow pipeline patterns")
        print("✅ Exercise 4: Hybrid ML + LLM evaluation system")
        
        print("\\n🏆 Key Accomplishments:")
        accomplishments = [
            "Implemented comprehensive algorithm comparison",
            "Mastered feature selection and engineering techniques",
            "Designed scalable Metaflow pipeline architectures",
            "Created hybrid quantitative + qualitative evaluation",
            "Generated production-ready ML workflows",
            "Integrated LLM-powered model interpretation"
        ]
        
        for accomplishment in accomplishments:
            print(f"   • {accomplishment}")
        
        print("\\n📚 Skills Demonstrated:")
        skills = [
            "Advanced scikit-learn usage and optimization",
            "Metaflow parallel processing and resource management", 
            "LangChain integration for model interpretation",
            "Comprehensive evaluation metrics and reporting",
            "Production-ready ML pipeline design",
            "Hybrid AI system architecture"
        ]
        
        for skill in skills:
            print(f"   🎯 {skill}")
        
        print("\\n🚀 Ready for Week 4: Advanced ML and LangGraph!")
        
        return self.results


# Usage example
if __name__ == "__main__":
    solutions = Week3Solutions()
    
    # Run individual exercises
    # solutions.exercise_1_classification_challenges()
    # solutions.exercise_2_regression_practice()
    # solutions.exercise_3_pipeline_optimization()
    # solutions.exercise_4_hybrid_evaluation()
    
    # Or run all exercises
    all_results = solutions.run_all_exercises()
    
    print("\\n💾 All results saved to solutions.results")
    print("🎯 Solutions complete - ready for practical application!")
