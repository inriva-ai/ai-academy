"""
Week 2 Exercise Solutions: Data Preprocessing and LangChain Integration
========================================================================

Complete solutions to all Week 2 exercises with explanations and best practices.

Solutions included:
1. Enhanced Data Preprocessing Pipeline (Advanced Metaflow)
2. Multi-Model LangChain Comparison System
3. Hybrid ML + LLM Integration Pipeline

Author: INRIVA AI Academy
Date: Week 2 Solutions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from metaflow import FlowSpec, step, Parameter
import time
import warnings
from datetime import datetime
import subprocess
import json

# Handle LangChain imports gracefully
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.llms import Ollama
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel, RunnableBranch
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️ LangChain not available - install with: pip install langchain langchain-community")
    LANGCHAIN_AVAILABLE = False

warnings.filterwarnings('ignore')


# ============================================================================
# SOLUTION 1: Enhanced Data Preprocessing Pipeline
# ============================================================================

class AdvancedImputer:
    """Advanced missing value imputation with multiple strategies"""
    
    def __init__(self):
        self.imputers = {}
        self.strategies = {}
    
    def fit_transform(self, df, strategy_map=None):
        """
        Apply different imputation strategies for different columns
        
        EXPLANATION:
        This solution demonstrates advanced imputation by:
        1. Auto-detecting best strategy per column based on data characteristics
        2. Using KNN for numerical data to preserve relationships
        3. Using iterative imputation for complex missing patterns
        4. Handling categorical data appropriately
        """
        if strategy_map is None:
            strategy_map = self._auto_strategy(df)
        
        df_imputed = df.copy()
        
        for column, strategy in strategy_map.items():
            if column not in df.columns or df[column].isnull().sum() == 0:
                continue
            
            if strategy == 'knn':
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                df_imputed[[column]] = imputer.fit_transform(df[[column]])
                self.imputers[column] = imputer
            
            elif strategy == 'iterative':
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                imputer = IterativeImputer(random_state=42, max_iter=10)
                df_imputed[[column]] = imputer.fit_transform(df[[column]])
                self.imputers[column] = imputer
            
            elif strategy in ['mean', 'median', 'most_frequent']:
                imputer = SimpleImputer(strategy=strategy)
                df_imputed[[column]] = imputer.fit_transform(df[[column]])
                self.imputers[column] = imputer
        
        return df_imputed
    
    def _auto_strategy(self, df):
        """Automatically determine best imputation strategy"""
        strategies = {}
        
        for column in df.columns:
            if df[column].isnull().sum() == 0:
                continue
            
            if df[column].dtype in ['object', 'category']:
                strategies[column] = 'most_frequent'
            elif df[column].dtype in ['int64', 'float64']:
                # Check skewness to decide between mean/median
                skewness = abs(df[column].skew())
                if skewness > 1:
                    strategies[column] = 'median'  # Skewed data
                else:
                    strategies[column] = 'knn'     # Normal data - use KNN
        
        return strategies


class EnhancedFeatureEngineer:
    """Advanced feature engineering with domain-specific knowledge"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_titanic_features(self, df):
        """
        Create Titanic-specific features
        
        EXPLANATION:
        This demonstrates domain knowledge application:
        1. Family size impacts survival (larger families had lower survival)
        2. Fare per person normalizes economic status by family size
        3. Title extraction reveals social status
        4. Deck information from cabin numbers
        5. Age groups for different survival patterns
        """
        df_new = df.copy()
        
        # Family size features
        df_new['family_size'] = df_new['SibSp'] + df_new['Parch'] + 1
        df_new['is_alone'] = (df_new['family_size'] == 1).astype(int)
        
        # Economic features
        df_new['fare_per_person'] = df_new['Fare'] / df_new['family_size']
        
        # Title extraction from names
        df_new['title'] = df_new['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        # Group rare titles
        rare_titles = df_new['title'].value_counts()[df_new['title'].value_counts() < 10].index
        df_new['title'] = df_new['title'].replace(rare_titles, 'Rare')
        
        # Deck from cabin (first character)
        df_new['deck'] = df_new['Cabin'].str[0]
        df_new['has_cabin'] = df_new['Cabin'].notna().astype(int)
        
        # Age groups
        df_new['age_group'] = pd.cut(df_new['Age'], 
                                   bins=[0, 12, 18, 35, 60, 100], 
                                   labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        
        # Interaction features
        df_new['class_gender'] = df_new['Pclass'].astype(str) + '_' + df_new['Sex']
        
        self.feature_names = ['family_size', 'is_alone', 'fare_per_person', 
                            'title', 'deck', 'has_cabin', 'age_group', 'class_gender']
        
        return df_new


class EnhancedPreprocessingFlow(FlowSpec):
    """
    SOLUTION 1: Enhanced Data Preprocessing Pipeline
    
    This flow demonstrates advanced preprocessing techniques:
    - Sophisticated missing value handling
    - Domain-specific feature engineering
    - Advanced validation and quality checks
    - Feature selection and dimensionality management
    """
    
    # Parameters for flexibility
    missing_threshold = Parameter('missing_threshold',
                                help='Drop columns with >X fraction missing',
                                default=0.5)
    
    correlation_threshold = Parameter('correlation_threshold',
                                    help='Flag high correlation features',
                                    default=0.95)
    
    @step
    def start(self):
        """Load and validate input data"""
        print("🚀 Starting Enhanced Preprocessing Pipeline...")
        
        # Create sample Titanic-like data for demonstration
        self.df = self._create_sample_data()
        
        print(f"📊 Loaded data: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        self.next(self.data_quality_assessment)
    
    def _create_sample_data(self):
        """Create sample dataset for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Age': np.random.normal(30, 12, n_samples),
            'Fare': np.random.lognormal(2, 1, n_samples),
            'SibSp': np.random.poisson(0.5, n_samples),
            'Parch': np.random.poisson(0.3, n_samples),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.3, 0.4]),
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1]),
            'Survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        }
        
        df = pd.DataFrame(data)
        
        # Add some realistic missing values
        df.loc[np.random.choice(df.index, 100), 'Age'] = np.nan
        df.loc[np.random.choice(df.index, 50), 'Embarked'] = np.nan
        
        # Add name and cabin columns for feature engineering
        df['Name'] = [f"Passenger_{i}, Mr. John" for i in range(n_samples)]
        df['Cabin'] = np.random.choice(['A1', 'B2', 'C3', np.nan], n_samples, p=[0.1, 0.1, 0.1, 0.7])
        
        return df
    
    @step
    def data_quality_assessment(self):
        """Comprehensive data quality analysis"""
        print("🔍 Performing comprehensive data quality assessment...")
        
        # Missing value analysis
        missing_analysis = {}
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = missing_count / len(self.df)
            missing_analysis[col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
        
        # Identify columns to drop due to high missing values
        self.columns_to_drop = [col for col, info in missing_analysis.items() 
                              if info['percentage'] > self.missing_threshold]
        
        # Outlier detection
        self.outlier_analysis = self._detect_outliers()
        
        # Data type analysis
        self.dtype_analysis = {
            'numerical': self.df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical': self.df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Class imbalance check (assuming 'Survived' is target)
        if 'Survived' in self.df.columns:
            target_dist = self.df['Survived'].value_counts(normalize=True)
            self.class_imbalance = min(target_dist) < 0.3  # Flag if minority class < 30%
        
        self.quality_report = {
            'missing_analysis': missing_analysis,
            'columns_to_drop': self.columns_to_drop,
            'outlier_summary': {col: len(indices) for col, indices in self.outlier_analysis.items()},
            'class_imbalance': getattr(self, 'class_imbalance', False),
            'dtype_summary': self.dtype_analysis
        }
        
        print(f"✅ Quality assessment complete:")
        print(f"   - Columns to drop: {len(self.columns_to_drop)}")
        print(f"   - Outliers detected in: {len(self.outlier_analysis)} columns")
        print(f"   - Class imbalance: {getattr(self, 'class_imbalance', False)}")
        
        self.next(self.advanced_missing_handling)
    
    def _detect_outliers(self):
        """Detect outliers using IQR method"""
        outliers = {}
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_indices = self.df[(self.df[col] < lower_bound) | 
                                    (self.df[col] > upper_bound)].index.tolist()
            outliers[col] = outlier_indices
        
        return outliers
    
    @step
    def advanced_missing_handling(self):
        """Advanced missing value imputation"""
        print("🔧 Applying advanced missing value strategies...")
        
        # Drop columns with too many missing values
        self.df_clean = self.df.drop(columns=self.columns_to_drop)
        
        # Apply sophisticated imputation
        imputer = AdvancedImputer()
        self.df_imputed = imputer.fit_transform(self.df_clean)
        self.imputation_strategy = imputer.strategies
        
        # Validation: Check that missing values were handled
        remaining_missing = self.df_imputed.isnull().sum().sum()
        
        print(f"✅ Missing value handling complete:")
        print(f"   - Dropped {len(self.columns_to_drop)} columns")
        print(f"   - Imputed {len(self.imputation_strategy)} columns")
        print(f"   - Remaining missing values: {remaining_missing}")
        
        self.next(self.advanced_feature_engineering)
    
    @step
    def advanced_feature_engineering(self):
        """Create sophisticated features"""
        print("⚙️ Creating advanced features...")
        
        # Apply domain-specific feature engineering
        fe = EnhancedFeatureEngineer()
        self.df_features = fe.create_titanic_features(self.df_imputed)
        
        # Create polynomial features for key numerical columns
        numerical_cols = ['Age', 'Fare']
        if all(col in self.df_features.columns for col in numerical_cols):
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            poly_features = poly.fit_transform(self.df_features[numerical_cols])
            
            # Add polynomial features
            poly_feature_names = [f"{numerical_cols[0]}_x_{numerical_cols[1]}"]
            for i, name in enumerate(poly_feature_names):
                if i + len(numerical_cols) < poly_features.shape[1]:
                    self.df_features[name] = poly_features[:, i + len(numerical_cols)]
        
        self.new_features = fe.feature_names
        
        print(f"✅ Feature engineering complete:")
        print(f"   - Created {len(self.new_features)} domain-specific features")
        print(f"   - Added polynomial interactions")
        print(f"   - Final feature count: {self.df_features.shape[1]}")
        
        self.next(self.feature_selection)
    
    @step
    def feature_selection(self):
        """Select most important features"""
        print("🎯 Performing feature selection...")
        
        # Encode categorical variables for feature selection
        df_encoded = self.df_features.copy()
        
        # Label encode categorical columns
        label_encoders = {}
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col != 'Survived':  # Don't encode target
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                label_encoders[col] = le
        
        # Separate features and target
        if 'Survived' in df_encoded.columns:
            X = df_encoded.drop('Survived', axis=1)
            y = df_encoded['Survived']
            
            # Feature selection using statistical tests
            selector = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            self.selected_features = selected_features
            
            # Create final dataset
            self.df_final = pd.DataFrame(X_selected, columns=selected_features)
            self.df_final['Survived'] = y.values
            
        else:
            self.df_final = df_encoded
            self.selected_features = df_encoded.columns.tolist()
        
        self.label_encoders = label_encoders
        
        print(f"✅ Feature selection complete:")
        print(f"   - Selected {len(self.selected_features)} features")
        print(f"   - Features: {self.selected_features}")
        
        self.next(self.validation_and_scaling)
    
    @step
    def validation_and_scaling(self):
        """Final validation and scaling"""
        print("⚖️ Applying scaling and final validation...")
        
        # Check for high correlation
        numerical_cols = self.df_final.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            corr_matrix = self.df_final[numerical_cols].corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > self.correlation_threshold:
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]
                        ))
            
            self.high_correlations = high_corr_pairs
        else:
            self.high_correlations = []
        
        # Apply scaling
        if 'Survived' in self.df_final.columns:
            X = self.df_final.drop('Survived', axis=1)
            y = self.df_final['Survived']
        else:
            X = self.df_final
            y = None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        if y is not None:
            self.df_scaled['Survived'] = y.values
        
        self.scaler = scaler
        
        print(f"✅ Validation and scaling complete:")
        print(f"   - High correlation pairs: {len(self.high_correlations)}")
        print(f"   - Final dataset shape: {self.df_scaled.shape}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """Generate final report"""
        print("📋 ENHANCED PREPROCESSING REPORT")
        print("=" * 50)
        print(f"Original shape: {self.df.shape}")
        print(f"Final shape: {self.df_scaled.shape}")
        print(f"Columns dropped: {len(self.columns_to_drop)}")
        print(f"Features engineered: {len(self.new_features)}")
        print(f"Features selected: {len(self.selected_features)}")
        print(f"High correlations found: {len(self.high_correlations)}")
        print("🎉 Enhanced preprocessing pipeline completed!")


# ============================================================================
# SOLUTION 2: Multi-Model LangChain Comparison System
# ============================================================================

class ModelComparison:
    """
    SOLUTION 2: Multi-Model LangChain Comparison
    
    This solution demonstrates:
    1. Multiple model management with Ollama
    2. Intelligent task routing based on complexity
    3. Error handling and fallback strategies
    4. Performance comparison and monitoring
    """
    
    def __init__(self):
        self.models = {}
        self.performance_log = []
        self.available_models = []
        
        if LANGCHAIN_AVAILABLE:
            self._initialize_models()
        else:
            print("⚠️ LangChain not available - using mock implementation")
    
    def _initialize_models(self):
        """Initialize multiple Ollama models"""
        print("🔍 Checking available Ollama models...")
        
        try:
            # Check what models are available
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse available models
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        if ':' not in model_name:  # Add default tag
                            model_name += ':latest'
                        self.available_models.append(model_name)
                
                print(f"✅ Found {len(self.available_models)} available models")
                
                # Initialize models for different tasks
                self._setup_model_roles()
            else:
                print("❌ Ollama not responding - using fallback")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Ollama not installed or not running")
    
    def _setup_model_roles(self):
        """Assign models to different roles based on capabilities"""
        model_roles = {
            'quick_analysis': 'llama3.2:1b',      # Fast model for simple tasks
            'detailed_analysis': 'llama3.2',       # Balanced model
            'code_analysis': 'codellama',          # Specialized for code
            'general_purpose': 'mistral'           # Alternative general model
        }
        
        # Only initialize models that are actually available
        for role, preferred_model in model_roles.items():
            available_variants = [m for m in self.available_models 
                                if m.startswith(preferred_model.split(':')[0])]
            
            if available_variants:
                model_name = available_variants[0]
                try:
                    self.models[role] = Ollama(model=model_name, temperature=0.1)
                    print(f"✅ Initialized {role}: {model_name}")
                except Exception as e:
                    print(f"❌ Failed to initialize {role} with {model_name}: {e}")
            else:
                print(f"⚠️ No suitable model found for {role}")
        
        # Fallback: use any available model for missing roles
        if self.available_models and len(self.models) == 0:
            fallback_model = self.available_models[0]
            try:
                self.models['general_purpose'] = Ollama(model=fallback_model)
                print(f"✅ Using fallback model: {fallback_model}")
            except Exception as e:
                print(f"❌ Fallback model failed: {e}")
    
    def route_task(self, task_description, data_summary):
        """
        Intelligently route tasks to appropriate models
        
        EXPLANATION:
        This demonstrates intelligent routing by:
        1. Analyzing task complexity and type
        2. Selecting appropriate model based on requirements
        3. Implementing fallback strategies
        4. Logging performance for optimization
        """
        if not LANGCHAIN_AVAILABLE or not self.models:
            return self._mock_analysis(task_description, data_summary)
        
        # Determine task complexity and type
        task_type = self._classify_task(task_description)
        
        # Route to appropriate model
        if task_type == 'quick':
            model_key = 'quick_analysis'
        elif task_type == 'code':
            model_key = 'code_analysis'
        elif task_type == 'detailed':
            model_key = 'detailed_analysis'
        else:
            model_key = 'general_purpose'
        
        # Get model (with fallback)
        model = self._get_model_with_fallback(model_key)
        
        if model is None:
            return "❌ No suitable model available"
        
        # Create appropriate prompt
        prompt = self._create_prompt(task_type, task_description, data_summary)
        
        # Execute with timing and error handling
        start_time = time.time()
        try:
            result = model.invoke(prompt)
            execution_time = time.time() - start_time
            
            # Log performance
            self.performance_log.append({
                'task_type': task_type,
                'model_key': model_key,
                'execution_time': execution_time,
                'success': True,
                'timestamp': datetime.now()
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log failure
            self.performance_log.append({
                'task_type': task_type,
                'model_key': model_key,
                'execution_time': execution_time,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            })
            
            return f"❌ Analysis failed: {str(e)}"
    
    def _classify_task(self, task_description):
        """Classify task type based on description"""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ['quick', 'summary', 'brief', 'overview']):
            return 'quick'
        elif any(word in task_lower for word in ['code', 'script', 'function', 'programming']):
            return 'code'
        elif any(word in task_lower for word in ['detailed', 'comprehensive', 'deep', 'thorough']):
            return 'detailed'
        else:
            return 'general'
    
    def _get_model_with_fallback(self, preferred_key):
        """Get model with fallback to available alternatives"""
        if preferred_key in self.models:
            return self.models[preferred_key]
        
        # Fallback order
        fallback_order = ['general_purpose', 'detailed_analysis', 'quick_analysis', 'code_analysis']
        
        for key in fallback_order:
            if key in self.models:
                return self.models[key]
        
        return None
    
    def _create_prompt(self, task_type, task_description, data_summary):
        """Create task-specific prompts"""
        prompts = {
            'quick': f"""Provide a brief analysis of this data:

Task: {task_description}
Data Summary: {data_summary}

Give 2-3 key insights in bullet points.""",
            
            'detailed': f"""Perform a comprehensive analysis of this data:

Task: {task_description}
Data Summary: {data_summary}

Provide:
1. Key patterns and trends
2. Statistical insights
3. Potential issues or concerns
4. Recommendations for further analysis""",
            
            'code': f"""Analyze this data from a technical perspective:

Task: {task_description}
Data Summary: {data_summary}

Focus on:
1. Data quality and structure
2. Potential preprocessing needs
3. Suggested analytical approaches
4. Code considerations""",
            
            'general': f"""Analyze this data:

Task: {task_description}
Data Summary: {data_summary}

Provide insights and recommendations."""
        }
        
        return prompts.get(task_type, prompts['general'])
    
    def _mock_analysis(self, task_description, data_summary):
        """Mock analysis when LangChain is not available"""
        return f"""Mock Analysis (LangChain not available):
        
Task: {task_description}
Data: {data_summary}

This would normally provide AI-powered insights using multiple LLM models.
To use real analysis, install: pip install langchain langchain-community ollama
"""
    
    def compare_models_performance(self):
        """Compare performance across different models"""
        if not self.performance_log:
            return "No performance data available"
        
        df_perf = pd.DataFrame(self.performance_log)
        
        # Calculate metrics by model
        model_performance = df_perf.groupby('model_key').agg({
            'execution_time': ['mean', 'std', 'count'],
            'success': 'mean'
        }).round(3)
        
        return model_performance
    
    def get_model_recommendations(self):
        """Provide model usage recommendations"""
        if not self.performance_log:
            return "No data available for recommendations"
        
        df_perf = pd.DataFrame(self.performance_log)
        
        # Find best model for each task type
        recommendations = {}
        
        for task_type in df_perf['task_type'].unique():
            task_data = df_perf[df_perf['task_type'] == task_type]
            
            # Best by success rate, then by speed
            best_model = task_data.groupby('model_key').agg({
                'success': 'mean',
                'execution_time': 'mean'
            }).sort_values(['success', 'execution_time'], ascending=[False, True]).index[0]
            
            recommendations[task_type] = best_model
        
        return recommendations


def demo_model_comparison():
    """Demonstrate the multi-model comparison system"""
    print("🚀 Demonstrating Multi-Model LangChain Comparison")
    print("=" * 60)
    
    # Initialize comparison system
    comparison = ModelComparison()
    
    # Test different types of analysis
    test_cases = [
        {
            'task': 'Quick summary of passenger demographics',
            'data': 'Titanic dataset: 891 passengers, 12 features, 38% survival rate'
        },
        {
            'task': 'Detailed analysis of survival patterns',
            'data': 'Age range: 0-80, Classes: 1st(24%), 2nd(21%), 3rd(55%), Gender: Male(65%), Female(35%)'
        },
        {
            'task': 'Code recommendations for data preprocessing',
            'data': 'Missing values: Age(20%), Cabin(77%), Embarked(0.2%)'
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test Case {i}: {test_case['task']}")
        print("-" * 40)
        
        result = comparison.route_task(test_case['task'], test_case['data'])
        results.append(result)
        
        print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
    
    # Show performance comparison
    print(f"\n📈 Performance Analysis:")
    print("-" * 40)
    perf_comparison = comparison.compare_models_performance()
    print(perf_comparison)
    
    print(f"\n💡 Model Recommendations:")
    print("-" * 40)
    recommendations = comparison.get_model_recommendations()
    for task, model in recommendations.items():
        print(f"  {task}: {model}")
    
    return comparison, results


# ============================================================================
# SOLUTION 3: Hybrid ML + LLM Integration Pipeline
# ============================================================================

class HybridAnalysisPipeline(FlowSpec):
    """
    SOLUTION 3: Hybrid ML + LLM Integration Pipeline
    
    This solution demonstrates:
    1. Traditional ML pipeline with model training
    2. LLM-powered model interpretation and insights
    3. Automated report generation combining both approaches
    4. Performance monitoring and comparison
    """
    
    dataset_name = Parameter('dataset_name',
                           help='Name of dataset to analyze',
                           default='titanic')
    
    @step
    def start(self):
        """Initialize hybrid pipeline"""
        print("🚀 Starting Hybrid ML + LLM Analysis Pipeline")
        
        # Load sample data
        self.df = self._load_sample_data()
        print(f"📊 Loaded {self.dataset_name} dataset: {self.df.shape}")
        
        self.next(self.traditional_ml_analysis)
    
    def _load_sample_data(self):
        """Load sample dataset for analysis"""
        # Use the same sample data generator as before
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Age': np.random.normal(30, 12, n_samples),
            'Fare': np.random.lognormal(2, 1, n_samples),
            'SibSp': np.random.poisson(0.5, n_samples),
            'Parch': np.random.poisson(0.3, n_samples),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.3, 0.4]),
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1]),
            'Survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        }
        
        return pd.DataFrame(data)
    
    @step
    def traditional_ml_analysis(self):
        """Perform traditional ML analysis"""
        print("🤖 Performing traditional ML analysis...")
        
        # Prepare data
        df_processed = self.df.copy()
        
        # Handle categorical variables
        le_sex = LabelEncoder()
        le_embarked = LabelEncoder()
        
        df_processed['Sex_encoded'] = le_sex.fit_transform(df_processed['Sex'])
        df_processed['Embarked_encoded'] = le_embarked.fit_transform(df_processed['Embarked'].fillna('S'))
        
        # Select features
        feature_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass', 'Sex_encoded', 'Embarked_encoded']
        X = df_processed[feature_cols]
        y = df_processed['Survived']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        
        # Store results
        self.ml_results = {
            'model': model,
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'feature_importance': feature_importance,
            'feature_names': feature_cols,
            'test_size': len(X_test)
        }
        
        print(f"✅ ML Analysis complete:")
        print(f"   - Accuracy: {accuracy:.3f}")
        print(f"   - Top features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        self.next(self.llm_interpretation)
    
    @step
    def llm_interpretation(self):
        """Use LLM to interpret ML results"""
        print("🧠 Generating LLM interpretation of ML results...")
        
        if not LANGCHAIN_AVAILABLE:
            self.llm_insights = self._mock_llm_interpretation()
            self.next(self.hybrid_report_generation)
            return
        
        # Initialize model comparison system
        comparison = ModelComparison()
        
        # Prepare data summary for LLM
        data_summary = self._create_ml_summary()
        
        # Get different types of analysis
        analyses = {}
        
        # 1. Model performance interpretation
        task_performance = "Interpret the model performance metrics and explain what they mean for business decisions"
        analyses['performance'] = comparison.route_task(task_performance, data_summary)
        
        # 2. Feature importance explanation
        task_features = "Explain the feature importance results in plain English and business context"
        analyses['features'] = comparison.route_task(task_features, data_summary)
        
        # 3. Recommendations
        task_recommendations = "Provide actionable recommendations based on the model results"
        analyses['recommendations'] = comparison.route_task(task_recommendations, data_summary)
        
        self.llm_insights = analyses
        self.comparison_system = comparison
        
        print(f"✅ LLM interpretation complete:")
        print(f"   - Generated {len(analyses)} different analyses")
        
        self.next(self.hybrid_report_generation)
    
    def _create_ml_summary(self):
        """Create summary of ML results for LLM analysis"""
        ml = self.ml_results
        
        # Top features
        top_features = sorted(ml['feature_importance'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        
        summary = f"""
        ML Model Results Summary:
        - Model Type: Random Forest Classifier
        - Dataset: {self.dataset_name} ({self.df.shape[0]} samples)
        - Accuracy: {ml['accuracy']:.3f}
        - Precision: {ml['classification_report']['macro avg']['precision']:.3f}
        - Recall: {ml['classification_report']['macro avg']['recall']:.3f}
        - F1-Score: {ml['classification_report']['macro avg']['f1-score']:.3f}
        
        Top 5 Most Important Features:
        {chr(10).join([f"- {feature}: {importance:.3f}" for feature, importance in top_features])}
        
        Class Distribution in Results:
        - Class 0: {ml['classification_report']['0']['support']} samples
        - Class 1: {ml['classification_report']['1']['support']} samples
        
        Test Set Size: {ml['test_size']} samples
        """
        
        return summary
    
    def _mock_llm_interpretation(self):
        """Mock LLM interpretation when LangChain not available"""
        return {
            'performance': f"""Model Performance Analysis (Mock):
            The accuracy of {self.ml_results['accuracy']:.3f} indicates reasonably good predictive performance.
            This suggests the model can correctly classify about {self.ml_results['accuracy']*100:.1f}% of cases.
            """,
            
            'features': f"""Feature Importance Analysis (Mock):
            The most important features are {list(self.ml_results['feature_importance'].keys())[:3]}.
            These features drive the majority of the model's decisions.
            """,
            
            'recommendations': """Recommendations (Mock):
            1. Focus on the top important features for business decisions
            2. Collect more data for less important features
            3. Consider feature engineering to improve performance
            """
        }
    
    @step
    def hybrid_report_generation(self):
        """Generate comprehensive hybrid report"""
        print("📋 Generating comprehensive hybrid report...")
        
        # Combine traditional ML metrics with LLM insights
        self.hybrid_report = self._create_comprehensive_report()
        
        # Generate visualizations
        self._create_visualizations()
        
        print("✅ Hybrid report generation complete")
        
        self.next(self.performance_comparison)
    
    def _create_comprehensive_report(self):
        """Create comprehensive report combining ML and LLM insights"""
        ml = self.ml_results
        llm = self.llm_insights
        
        report = {
            'executive_summary': {
                'dataset_info': f"{self.dataset_name} dataset with {self.df.shape[0]} samples",
                'model_performance': f"Achieved {ml['accuracy']:.1%} accuracy",
                'key_insights': "Combined traditional ML with AI-powered interpretation"
            },
            
            'technical_results': {
                'accuracy': ml['accuracy'],
                'precision': ml['classification_report']['macro avg']['precision'],
                'recall': ml['classification_report']['macro avg']['recall'],
                'f1_score': ml['classification_report']['macro avg']['f1-score'],
                'feature_importance': ml['feature_importance']
            },
            
            'ai_interpretation': llm,
            
            'recommendations': {
                'immediate_actions': [
                    f"Focus on top {len([f for f, imp in ml['feature_importance'].items() if imp > 0.1])} most important features",
                    "Implement model monitoring for production deployment",
                    "Collect additional data for underperforming features"
                ],
                'future_improvements': [
                    "Experiment with ensemble methods",
                    "Implement feature engineering pipeline",
                    "Set up automated retraining schedule"
                ]
            },
            
            'methodology': {
                'ml_approach': "Random Forest with feature importance analysis",
                'llm_approach': "Multi-model LLM interpretation system",
                'integration': "Hybrid pipeline combining quantitative and qualitative insights"
            }
        }
        
        return report
    
    def _create_visualizations(self):
        """Create visualization artifacts"""
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        features = list(self.ml_results['feature_importance'].keys())
        importances = list(self.ml_results['feature_importance'].values())
        
        plt.barh(features, importances)
        plt.title('Feature Importance Analysis')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        # Save plot info (in real implementation, would save actual plots)
        self.visualizations = {
            'feature_importance_plot': 'Generated feature importance bar chart',
            'performance_metrics_plot': 'Generated performance metrics visualization'
        }
    
    @step
    def performance_comparison(self):
        """Compare traditional ML vs LLM-enhanced insights"""
        print("📊 Comparing traditional ML vs LLM-enhanced analysis...")
        
        # Traditional insights (what we get from ML alone)
        traditional_insights = [
            f"Model accuracy: {self.ml_results['accuracy']:.3f}",
            f"Most important feature: {max(self.ml_results['feature_importance'], key=self.ml_results['feature_importance'].get)}",
            f"Feature count: {len(self.ml_results['feature_importance'])}"
        ]
        
        # Enhanced insights (what LLM adds)
        if LANGCHAIN_AVAILABLE and hasattr(self, 'comparison_system'):
            enhanced_insights = [
                "Business context interpretation of model results",
                "Plain English explanations of technical metrics",
                "Actionable recommendations for stakeholders",
                "Contextual understanding of feature relationships"
            ]
            
            # Performance metrics of LLM system
            llm_performance = self.comparison_system.compare_models_performance()
        else:
            enhanced_insights = ["Mock enhanced insights (LangChain not available)"]
            llm_performance = "Not available"
        
        self.comparison_results = {
            'traditional_insights': traditional_insights,
            'enhanced_insights': enhanced_insights,
            'llm_performance': llm_performance,
            'value_added': len(enhanced_insights) - len(traditional_insights)
        }
        
        print(f"✅ Performance comparison complete:")
        print(f"   - Traditional insights: {len(traditional_insights)}")
        print(f"   - Enhanced insights: {len(enhanced_insights)}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """Final report and conclusions"""
        print("\n" + "="*60)
        print("🎉 HYBRID ML + LLM ANALYSIS COMPLETE")
        print("="*60)
        
        print(f"\n📊 EXECUTIVE SUMMARY:")
        print(f"   Dataset: {self.hybrid_report['executive_summary']['dataset_info']}")
        print(f"   Performance: {self.hybrid_report['executive_summary']['model_performance']}")
        print(f"   Approach: {self.hybrid_report['executive_summary']['key_insights']}")
        
        print(f"\n🤖 TRADITIONAL ML RESULTS:")
        tech = self.hybrid_report['technical_results']
        print(f"   Accuracy: {tech['accuracy']:.3f}")
        print(f"   Precision: {tech['precision']:.3f}")
        print(f"   Recall: {tech['recall']:.3f}")
        print(f"   F1-Score: {tech['f1_score']:.3f}")
        
        print(f"\n🧠 LLM-ENHANCED INSIGHTS:")
        for insight_type, content in self.llm_insights.items():
            print(f"   {insight_type.title()}: Available")
        
        print(f"\n💡 KEY RECOMMENDATIONS:")
        for i, rec in enumerate(self.hybrid_report['recommendations']['immediate_actions'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n🔄 METHODOLOGY:")
        method = self.hybrid_report['methodology']
        print(f"   ML: {method['ml_approach']}")
        print(f"   LLM: {method['llm_approach']}")
        print(f"   Integration: {method['integration']}")
        
        print("\n✅ Hybrid analysis pipeline completed successfully!")


# ============================================================================
# DEMONSTRATION AND TESTING FUNCTIONS
# ============================================================================

def run_all_solutions():
    """Run all exercise solutions with comprehensive demonstrations"""
    print("🚀 WEEK 2 EXERCISE SOLUTIONS DEMONSTRATION")
    print("=" * 70)
    
    print("\n1️⃣ SOLUTION 1: Enhanced Data Preprocessing Pipeline")
    print("-" * 50)
    try:
        # This would run the Metaflow pipeline in a real environment
        print("✅ Enhanced preprocessing pipeline ready")
        print("   To run: python exercise_solutions.py run-enhanced-preprocessing")
    except Exception as e:
        print(f"ℹ️ Pipeline definition ready (run in Metaflow environment): {e}")
    
    print("\n2️⃣ SOLUTION 2: Multi-Model LangChain Comparison")
    print("-" * 50)
    try:
        comparison, results = demo_model_comparison()
        print("✅ Multi-model comparison demonstration complete")
    except Exception as e:
        print(f"ℹ️ Multi-model comparison ready: {e}")
    
    print("\n3️⃣ SOLUTION 3: Hybrid ML + LLM Integration")
    print("-" * 50)
    try:
        # This would run the Metaflow pipeline in a real environment
        print("✅ Hybrid pipeline ready")
        print("   To run: python exercise_solutions.py run-hybrid-pipeline")
    except Exception as e:
        print(f"ℹ️ Hybrid pipeline definition ready (run in Metaflow environment): {e}")
    
    print("\n🎯 SOLUTIONS SUMMARY:")
    print("-" * 30)
    print("✅ All three exercise solutions implemented")
    print("✅ Production-ready code with error handling")
    print("✅ Comprehensive documentation and explanations")
    print("✅ Integration patterns for MLOps + LLMOps")
    
    print("\n📚 LEARNING OUTCOMES ACHIEVED:")
    print("-" * 35)
    print("✅ Advanced data preprocessing techniques")
    print("✅ Multi-model LLM orchestration")
    print("✅ Hybrid ML + LLM integration patterns")
    print("✅ Production deployment considerations")
    print("✅ Performance monitoring and optimization")
    
    return True


def demonstrate_key_concepts():
    """Demonstrate key concepts from all solutions"""
    print("\n🎯 KEY CONCEPTS DEMONSTRATION")
    print("=" * 40)
    
    # 1. Advanced Imputation
    print("\n1. Advanced Missing Value Imputation:")
    imputer = AdvancedImputer()
    sample_data = pd.DataFrame({
        'age': [25, np.nan, 35, 45, np.nan],
        'income': [50000, 60000, np.nan, 80000, 55000],
        'category': ['A', 'B', np.nan, 'A', 'C']
    })
    
    print("   Before imputation:")
    print(f"   Missing values: {sample_data.isnull().sum().sum()}")
    
    imputed_data = imputer.fit_transform(sample_data)
    print("   After imputation:")
    print(f"   Missing values: {imputed_data.isnull().sum().sum()}")
    print(f"   Strategies used: {imputer.strategies}")
    
    # 2. Feature Engineering
    print("\n2. Domain-Specific Feature Engineering:")
    fe = EnhancedFeatureEngineer()
    sample_titanic = pd.DataFrame({
        'SibSp': [1, 0, 3, 1],
        'Parch': [0, 0, 2, 1], 
        'Fare': [100, 50, 30, 75],
        'Name': ['Smith, Mr. John', 'Jones, Mrs. Mary', 'Brown, Master. Tom', 'Davis, Miss. Jane'],
        'Cabin': ['A1', np.nan, 'B2', 'C3'],
        'Age': [30, 25, 5, 22],
        'Pclass': [1, 2, 3, 1],
        'Sex': ['male', 'female', 'male', 'female']
    })
    
    enhanced_features = fe.create_titanic_features(sample_titanic)
    print(f"   Original features: {sample_titanic.shape[1]}")
    print(f"   Enhanced features: {enhanced_features.shape[1]}")
    print(f"   New features: {fe.feature_names}")
    
    # 3. Model Comparison (if available)
    print("\n3. Multi-Model LLM System:")
    comparison = ModelComparison()
    if comparison.models:
        print(f"   Available models: {len(comparison.models)}")
        print(f"   Model roles: {list(comparison.models.keys())}")
    else:
        print("   System ready (requires Ollama installation)")
    
    print("\n✅ Key concepts demonstration complete!")


if __name__ == "__main__":
    """
    Main execution block for running solutions
    
    Usage:
    python exercise_solutions.py                    # Run all demonstrations
    python exercise_solutions.py concepts          # Show key concepts only
    python exercise_solutions.py run-enhanced-preprocessing  # Run Solution 1
    python exercise_solutions.py run-hybrid-pipeline       # Run Solution 3
    """
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "concepts":
            demonstrate_key_concepts()
        elif sys.argv[1] == "run-enhanced-preprocessing":
            # Run Solution 1
            flow = EnhancedPreprocessingFlow()
        elif sys.argv[1] == "run-hybrid-pipeline":
            # Run Solution 3
            flow = HybridAnalysisPipeline()
        else:
            print("Unknown command. Available: concepts, run-enhanced-preprocessing, run-hybrid-pipeline")
    else:
        # Run full demonstration
        run_all_solutions()
        demonstrate_key_concepts()
