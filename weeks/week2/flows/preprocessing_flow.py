"""
Week 2: Advanced Data Preprocessing Flow with Metaflow
=====================================================

This flow demonstrates advanced data preprocessing techniques including:
- Intelligent missing value handling
- Feature engineering and creation
- Data validation and quality assessment
- Categorical encoding strategies
- Integration with LangChain for analysis

Usage:
    python preprocessing_flow.py run
    python preprocessing_flow.py run --test_size 0.3 --scaling_method robust
"""

from metaflow import FlowSpec, step, Parameter, IncludeFile, catch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class AdvancedPreprocessingFlow(FlowSpec):
    """
    Advanced data preprocessing pipeline demonstrating best practices
    for feature engineering, validation, and scaling in production ML systems.
    """
    
    # Configuration parameters
    data_file = Parameter('data_file',
                         help='Path to input data file',
                         default='../data/titanic.csv')
    
    test_size = Parameter('test_size',
                         help='Proportion of data for testing',
                         default=0.2)
    
    random_state = Parameter('random_state',
                           help='Random seed for reproducibility',
                           default=42)
    
    scaling_method = Parameter('scaling_method',
                             help='Scaling method: standard, minmax, robust',
                             default='standard')
    
    missing_strategy = Parameter('missing_strategy',
                               help='Missing value strategy: median, mean, knn',
                               default='median')
    
    feature_engineering = Parameter('feature_engineering',
                                  help='Enable advanced feature engineering',
                                  default=True)
    
    @step
    def start(self):
        """
        Initialize preprocessing pipeline and load data
        """
        print("🚀 Starting Advanced Data Preprocessing Pipeline")
        print(f"📋 Configuration:")
        print(f"   Data file: {self.data_file}")
        print(f"   Test size: {self.test_size}")
        print(f"   Random state: {self.random_state}")
        print(f"   Scaling method: {self.scaling_method}")
        print(f"   Missing strategy: {self.missing_strategy}")
        print(f"   Feature engineering: {self.feature_engineering}")
        
        # Load data with error handling
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Data loaded successfully: {self.df.shape}")
        except FileNotFoundError:
            print(f"⚠️  Data file not found: {self.data_file}")
            print("   Creating synthetic dataset for demonstration...")
            self.df = self._create_sample_dataset()
            print(f"✅ Synthetic data created: {self.df.shape}")
        
        # Store original data info
        self.original_shape = self.df.shape
        self.original_columns = self.df.columns.tolist()
        
        self.next(self.explore_data)
    
    def _create_sample_dataset(self):
        """Create sample Titanic-like dataset for demonstration"""
        np.random.seed(self.random_state)
        n_samples = 891
        
        # Demographics with realistic patterns
        ages = np.random.normal(30, 15, n_samples)
        ages = np.clip(ages, 0, 80)
        
        # Introduce missing values strategically
        missing_age_mask = np.random.random(n_samples) < 0.20
        ages[missing_age_mask] = np.nan
        
        # Categorical features
        sexes = np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35])
        pclasses = np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55])
        embarked = np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
        
        # Strategic missing values
        missing_embarked_mask = np.random.random(n_samples) < 0.002
        embarked[missing_embarked_mask] = None
        
        # Numerical features with correlations
        sibsp = np.random.poisson(0.5, n_samples)
        parch = np.random.poisson(0.4, n_samples)
        
        # Fare correlated with class
        fare_base = {1: 80, 2: 20, 3: 10}
        fares = [np.random.lognormal(np.log(fare_base[pc]), 0.5) for pc in pclasses]
        
        # Names with extractable titles
        titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.', 'Rev.']
        name_titles = np.random.choice(titles, n_samples, 
                                      p=[0.5, 0.2, 0.15, 0.05, 0.05, 0.05])
        first_names = ['John', 'Mary', 'James', 'Patricia', 'Robert', 'Jennifer']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia']
        names = [f"{np.random.choice(last_names)}, {title} {np.random.choice(first_names)}" 
                for title in name_titles]
        
        # Realistic survival patterns
        survival_prob = 0.3  # Base rate
        survival_prob += (sexes == 'female') * 0.4  # Women first
        survival_prob += (ages < 16) * 0.3  # Children first
        survival_prob += (pclasses == 1) * 0.3  # Class privilege
        survival_prob += (pclasses == 2) * 0.15
        survived = np.random.binomial(1, survival_prob)
        
        return pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Survived': survived,
            'Pclass': pclasses,
            'Name': names,
            'Sex': sexes,
            'Age': ages,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fares,
            'Embarked': embarked
        })
    
    @step
    def explore_data(self):
        """
        Comprehensive data exploration and quality assessment
        """
        print("🔍 Conducting comprehensive data exploration...")
        
        exploration_results = {}
        
        # Basic information
        exploration_results['shape'] = self.df.shape
        exploration_results['memory_usage'] = self.df.memory_usage(deep=True).sum()
        exploration_results['dtypes'] = self.df.dtypes.to_dict()
        
        # Missing values analysis
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        exploration_results['missing_values'] = {
            col: {'count': count, 'percentage': pct}
            for col, count, pct in zip(missing_counts.index, missing_counts.values, missing_percentages.values)
            if count > 0
        }
        
        # Numerical columns analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        exploration_results['numeric_columns'] = numeric_cols.tolist()
        
        # Outlier detection using IQR method
        outliers_summary = {}
        for col in numeric_cols:
            if col not in ['PassengerId', 'Survived']:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.df[col] < lower_bound) | 
                           (self.df[col] > upper_bound)).sum()
                outliers_summary[col] = {
                    'count': outliers,
                    'percentage': (outliers / len(self.df)) * 100,
                    'bounds': (lower_bound, upper_bound)
                }
        
        exploration_results['outliers'] = outliers_summary
        
        # Categorical columns analysis
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        exploration_results['categorical_columns'] = categorical_cols.tolist()
        
        # Cardinality analysis
        cardinality = {}
        for col in categorical_cols:
            unique_values = self.df[col].nunique()
            cardinality[col] = {
                'unique_count': unique_values,
                'cardinality_ratio': unique_values / len(self.df)
            }
        
        exploration_results['cardinality'] = cardinality
        
        # Data quality score
        quality_score = self._calculate_quality_score(exploration_results)
        exploration_results['quality_score'] = quality_score
        
        self.exploration_results = exploration_results
        
        print(f"📊 Exploration complete:")
        print(f"   Shape: {exploration_results['shape']}")
        print(f"   Missing values in {len(exploration_results['missing_values'])} columns")
        print(f"   Outliers detected in {len([k for k, v in outliers_summary.items() if v['count'] > 0])} columns")
        print(f"   Data quality score: {quality_score:.2f}/100")
        
        self.next(self.validate_data)
    
    def _calculate_quality_score(self, exploration_results):
        """Calculate overall data quality score"""
        score = 100.0
        
        # Penalize missing values
        for col_info in exploration_results['missing_values'].values():
            score -= col_info['percentage'] * 0.5
        
        # Penalize high cardinality categorical variables
        for col_info in exploration_results['cardinality'].values():
            if col_info['cardinality_ratio'] > 0.95:  # Nearly unique
                score -= 10
        
        # Penalize excessive outliers
        for col_info in exploration_results['outliers'].values():
            if col_info['percentage'] > 10:  # More than 10% outliers
                score -= 5
        
        return max(0, score)
    
    @step
    def validate_data(self):
        """
        Data validation and integrity checks
        """
        print("✅ Validating data integrity...")
        
        validation_results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates == 0:
            validation_results['passed'].append("No duplicate rows found")
        else:
            validation_results['warnings'].append(f"Found {duplicates} duplicate rows")
        
        # Check for completely empty columns
        empty_cols = self.df.columns[self.df.isnull().all()].tolist()
        if not empty_cols:
            validation_results['passed'].append("No completely empty columns")
        else:
            validation_results['failed'].append(f"Empty columns found: {empty_cols}")
        
        # Check for single-value columns (except ID columns)
        single_value_cols = []
        for col in self.df.columns:
            if not col.lower().endswith('id') and self.df[col].nunique() == 1:
                single_value_cols.append(col)
        
        if not single_value_cols:
            validation_results['passed'].append("No single-value columns")
        else:
            validation_results['warnings'].append(f"Single-value columns: {single_value_cols}")
        
        # Check for reasonable value ranges
        if 'Age' in self.df.columns:
            age_range = self.df['Age'].dropna()
            if age_range.min() >= 0 and age_range.max() <= 120:
                validation_results['passed'].append("Age values in reasonable range")
            else:
                validation_results['failed'].append(f"Age values out of range: {age_range.min()}-{age_range.max()}")
        
        self.validation_results = validation_results
        
        print(f"   Passed: {len(validation_results['passed'])} checks")
        print(f"   Failed: {len(validation_results['failed'])} checks")
        print(f"   Warnings: {len(validation_results['warnings'])} checks")
        
        self.next(self.handle_missing_values)
    
    @step
    def handle_missing_values(self):
        """
        Intelligent missing value handling with multiple strategies
        """
        print(f"🔧 Handling missing values using '{self.missing_strategy}' strategy...")
        
        df_clean = self.df.copy()
        missing_handling_log = {}
        
        # Age: Advanced imputation based on strategy
        if 'Age' in df_clean.columns and df_clean['Age'].isnull().any():
            original_missing = df_clean['Age'].isnull().sum()
            
            if self.missing_strategy == 'median':
                # Group-based median (more sophisticated)
                if 'Sex' in df_clean.columns and 'Pclass' in df_clean.columns:
                    age_imputer = df_clean.groupby(['Sex', 'Pclass'])['Age'].transform('median')
                    df_clean['Age'].fillna(age_imputer, inplace=True)
                    strategy_used = "group-based median by Sex and Pclass"
                else:
                    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
                    strategy_used = "overall median"
            
            elif self.missing_strategy == 'mean':
                df_clean['Age'].fillna(df_clean['Age'].mean(), inplace=True)
                strategy_used = "mean imputation"
            
            elif self.missing_strategy == 'knn':
                # KNN imputation (more advanced)
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    imputer = KNNImputer(n_neighbors=5)
                    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
                    strategy_used = "KNN imputation (k=5)"
                else:
                    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
                    strategy_used = "median (fallback from KNN)"
            
            missing_handling_log['Age'] = {
                'original_missing': original_missing,
                'strategy': strategy_used,
                'remaining_missing': df_clean['Age'].isnull().sum()
            }
        
        # Categorical variables: Mode or custom logic
        categorical_cols = ['Embarked', 'Cabin'] if 'Cabin' in df_clean.columns else ['Embarked']
        for col in categorical_cols:
            if col in df_clean.columns and df_clean[col].isnull().any():
                original_missing = df_clean[col].isnull().sum()
                
                if col == 'Embarked':
                    # Most frequent port
                    most_frequent = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'S'
                    df_clean[col].fillna(most_frequent, inplace=True)
                    strategy_used = f"most frequent value ({most_frequent})"
                
                missing_handling_log[col] = {
                    'original_missing': original_missing,
                    'strategy': strategy_used,
                    'remaining_missing': df_clean[col].isnull().sum()
                }
        
        # Numerical variables: Median or mean
        for col in ['Fare']:
            if col in df_clean.columns and df_clean[col].isnull().any():
                original_missing = df_clean[col].isnull().sum()
                
                if 'Pclass' in df_clean.columns:
                    # Class-based median
                    fare_imputer = df_clean.groupby('Pclass')[col].transform('median')
                    df_clean[col].fillna(fare_imputer, inplace=True)
                    strategy_used = "class-based median"
                else:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    strategy_used = "overall median"
                
                missing_handling_log[col] = {
                    'original_missing': original_missing,
                    'strategy': strategy_used,
                    'remaining_missing': df_clean[col].isnull().sum()
                }
        
        self.df_clean = df_clean
        self.missing_handling_log = missing_handling_log
        
        total_missing_before = self.df.isnull().sum().sum()
        total_missing_after = df_clean.isnull().sum().sum()
        
        print(f"   Missing values: {total_missing_before} → {total_missing_after}")
        for col, log in missing_handling_log.items():
            print(f"   {col}: {log['original_missing']} → {log['remaining_missing']} ({log['strategy']})")
        
        self.next(self.feature_engineering)
    
    @step
    def feature_engineering(self):
        """
        Advanced feature engineering and creation
        """
        print("⚙️ Engineering advanced features...")
        
        if not self.feature_engineering:
            print("   Feature engineering disabled - skipping")
            self.df_features = self.df_clean.copy()
            self.feature_engineering_log = {"message": "Feature engineering disabled"}
            self.next(self.encode_categorical)
            return
        
        df_features = self.df_clean.copy()
        feature_engineering_log = {}
        
        # 1. Family-related features
        if 'SibSp' in df_features.columns and 'Parch' in df_features.columns:
            df_features['FamilySize'] = df_features['SibSp'] + df_features['Parch'] + 1
            df_features['IsAlone'] = (df_features['FamilySize'] == 1).astype(int)
            df_features['FamilyType'] = pd.cut(df_features['FamilySize'], 
                                             bins=[0, 1, 4, 20], 
                                             labels=['Solo', 'Small', 'Large'])
            feature_engineering_log['family_features'] = ['FamilySize', 'IsAlone', 'FamilyType']
        
        # 2. Title extraction from names
        if 'Name' in df_features.columns:
            df_features['Title'] = df_features['Name'].str.extract(' ([A-Za-z]+)\\.')
            
            # Group rare titles
            title_counts = df_features['Title'].value_counts()
            rare_titles = title_counts[title_counts < 10].index
            df_features['Title'] = df_features['Title'].replace(rare_titles, 'Other')
            
            # Create title categories
            noble_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
            df_features['IsNoble'] = df_features['Title'].isin(noble_titles).astype(int)
            
            feature_engineering_log['title_features'] = ['Title', 'IsNoble']
        
        # 3. Age-related features
        if 'Age' in df_features.columns:
            df_features['AgeGroup'] = pd.cut(df_features['Age'], 
                                           bins=[0, 12, 18, 35, 60, 100], 
                                           labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
            
            df_features['IsChild'] = (df_features['Age'] < 18).astype(int)
            df_features['IsElderly'] = (df_features['Age'] > 60).astype(int)
            
            feature_engineering_log['age_features'] = ['AgeGroup', 'IsChild', 'IsElderly']
        
        # 4. Fare-related features
        if 'Fare' in df_features.columns:
            if 'FamilySize' in df_features.columns:
                df_features['FarePerPerson'] = df_features['Fare'] / df_features['FamilySize']
                feature_engineering_log['fare_features'] = ['FarePerPerson']
            
            # Fare quantiles
            df_features['FareGroup'] = pd.qcut(df_features['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
        
        # 5. Interaction features
        if 'Age' in df_features.columns and 'Pclass' in df_features.columns:
            df_features['Age_Pclass'] = df_features['Age'] * df_features['Pclass']
            feature_engineering_log['interaction_features'] = ['Age_Pclass']
        
        # 6. Text-based features
        if 'Name' in df_features.columns:
            df_features['NameLength'] = df_features['Name'].str.len()
            df_features['NameWordCount'] = df_features['Name'].str.split().str.len()
            feature_engineering_log['text_features'] = ['NameLength', 'NameWordCount']
        
        # 7. Statistical features
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['PassengerId', 'Survived']]
        
        if len(numeric_cols) > 1:
            # Add some polynomial features for key numeric variables
            key_numeric = ['Age', 'Fare']
            for col in key_numeric:
                if col in df_features.columns:
                    df_features[f'{col}_squared'] = df_features[col] ** 2
                    df_features[f'{col}_log'] = np.log1p(df_features[col])  # log(1+x) to handle zeros
            
            feature_engineering_log['polynomial_features'] = [f'{col}_squared' for col in key_numeric if col in df_features.columns]
            feature_engineering_log['log_features'] = [f'{col}_log' for col in key_numeric if col in df_features.columns]
        
        self.df_features = df_features
        self.feature_engineering_log = feature_engineering_log
        
        original_features = len(self.df_clean.columns)
        new_features = len(df_features.columns)
        features_added = new_features - original_features
        
        print(f"   Features: {original_features} → {new_features} (+{features_added})")
        print(f"   Categories created: {len(feature_engineering_log)} types")
        
        self.next(self.encode_categorical)
    
    @step
    def encode_categorical(self):
        """
        Advanced categorical encoding with multiple strategies
        """
        print("🏷️ Encoding categorical variables...")
        
        df_encoded = self.df_features.copy()
        encoding_log = {}
        
        # Identify categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove ID and name columns from encoding
        exclude_from_encoding = ['PassengerId', 'Name']
        categorical_cols = [col for col in categorical_cols if col not in exclude_from_encoding]
        
        # Label encoding for binary/ordinal categories
        label_encode_cols = ['Sex']  # Binary: male/female
        self.label_encoders = {}
        
        for col in label_encode_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                encoding_log[col] = {'method': 'label_encoding', 'classes': le.classes_.tolist()}
        
        # One-hot encoding for nominal categories
        onehot_cols = [col for col in categorical_cols if col not in label_encode_cols]
        
        for col in onehot_cols:
            if col in df_encoded.columns:
                # Handle missing values by converting to string
                df_encoded[col] = df_encoded[col].astype(str)
                
                # Get dummies with prefix
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                
                # Add to dataframe
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                encoding_log[col] = {
                    'method': 'one_hot_encoding',
                    'categories': df_encoded[col].unique().tolist(),
                    'dummy_columns': dummies.columns.tolist()
                }
        
        # Drop original categorical columns (keep PassengerId for tracking)
        cols_to_drop = [col for col in categorical_cols if col != 'PassengerId']
        cols_to_drop.extend(['Name'])  # Always drop name
        cols_to_drop = [col for col in cols_to_drop if col in df_encoded.columns]
        
        # Also drop the original Sex column since we have Sex_encoded
        if 'Sex' in df_encoded.columns and 'Sex_encoded' in df_encoded.columns:
            cols_to_drop.append('Sex')
        
        df_encoded = df_encoded.drop(columns=cols_to_drop, errors='ignore')
        
        self.df_encoded = df_encoded
        self.encoding_log = encoding_log
        
        print(f"   Categorical columns processed: {len(encoding_log)}")
        print(f"   Final columns: {len(df_encoded.columns)}")
        
        self.next(self.split_and_scale)
    
    @step
    def split_and_scale(self):
        """
        Data splitting and feature scaling
        """
        print("📊 Splitting data and applying scaling...")
        
        # Prepare features and target
        feature_cols = [col for col in self.df_encoded.columns 
                       if col not in ['Survived', 'PassengerId']]
        
        if 'Survived' not in self.df_encoded.columns:
            raise ValueError("Target variable 'Survived' not found in dataset")
        
        X = self.df_encoded[feature_cols]
        y = self.df_encoded['Survived']
        
        # Ensure all features are numeric
        X = X.select_dtypes(include=[np.number])
        
        print(f"   Features selected: {len(X.columns)}")
        print(f"   Target distribution: {y.value_counts().to_dict()}")
        
        # Train/test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        # Choose and apply scaler
        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if self.scaling_method not in scaler_map:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        self.scaler = scaler_map[self.scaling_method]
        
        # Fit scaler on training data only
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrames for easier handling
        self.X_train_scaled = pd.DataFrame(
            self.X_train_scaled, 
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.X_test_scaled, 
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        print(f"   Train set: {len(self.X_train)} samples")
        print(f"   Test set: {len(self.X_test)} samples")
        print(f"   Scaling method: {self.scaling_method}")
        
        self.next(self.quality_assessment)
    
    @step
    def quality_assessment(self):
        """
        Final data quality assessment
        """
        print("🔍 Conducting final quality assessment...")
        
        quality_metrics = {}
        
        # Feature distribution analysis
        feature_stats = {
            'mean': self.X_train_scaled.mean().to_dict(),
            'std': self.X_train_scaled.std().to_dict(),
            'skewness': self.X_train_scaled.skew().to_dict()
        }
        quality_metrics['feature_statistics'] = feature_stats
        
        # Class balance
        class_distribution = self.y_train.value_counts(normalize=True).to_dict()
        quality_metrics['class_balance'] = class_distribution
        
        # Feature correlation analysis
        correlation_matrix = self.X_train_scaled.corr()
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        quality_metrics['high_correlations'] = high_correlations
        
        # Missing values check (should be zero after preprocessing)
        missing_values = self.X_train_scaled.isnull().sum().sum()
        quality_metrics['remaining_missing'] = missing_values
        
        # Data leakage check (basic)
        if 'PassengerId' in self.X_train_scaled.columns:
            quality_metrics['data_leakage_warning'] = "PassengerId found in features"
        
        self.quality_metrics = quality_metrics
        
        print(f"   Class balance: {class_distribution}")
        print(f"   High correlations found: {len(high_correlations)}")
        print(f"   Remaining missing values: {missing_values}")
        
        self.next(self.generate_report)
    
    @step
    def generate_report(self):
        """
        Generate comprehensive preprocessing report
        """
        print("📋 Generating comprehensive preprocessing report...")
        
        # Compile complete report
        report = {
            'configuration': {
                'test_size': self.test_size,
                'random_state': self.random_state,
                'scaling_method': self.scaling_method,
                'missing_strategy': self.missing_strategy,
                'feature_engineering_enabled': self.feature_engineering
            },
            'data_overview': {
                'original_shape': self.original_shape,
                'final_shape': (len(self.X_train) + len(self.X_test), len(self.X_train_scaled.columns)),
                'features_created': len(self.X_train_scaled.columns) - len(self.original_columns)
            },
            'data_exploration': self.exploration_results,
            'validation_results': self.validation_results,
            'missing_value_handling': self.missing_handling_log,
            'feature_engineering': self.feature_engineering_log,
            'categorical_encoding': self.encoding_log,
            'quality_assessment': self.quality_metrics,
            'final_datasets': {
                'train_size': len(self.X_train),
                'test_size': len(self.X_test),
                'feature_count': len(self.X_train_scaled.columns),
                'target_distribution': self.y_train.value_counts().to_dict()
            }
        }
        
        # self.preprocessing_report = report
        
        # Print summary
        # print("\n📊 PREPROCESSING SUMMARY")
        # print("=" * 30)
        # print(f"Original shape: {report['data_overview']['original_shape']}")
        # print(f"Final shape: {report['data_overview']['final_shape']}")
        # print(f"Features created: {report['data_overview']['features_created']}")
        # print(f"Train/test split: {report['final_datasets']['train_size']}/{report['final_datasets']['test_size']}")
        # print(f"Data quality score: {report['data_exploration']['quality_score']:.1f}/100")

        self.next(self.end)
    
    @step
    def end(self):
        """
        Pipeline completion
        """
        print("\n🎉 Advanced Preprocessing Pipeline Complete!")
        print("=" * 45)
        
        print("\n🎯 Pipeline Outputs:")
        print(f"   ✅ Clean training data: {self.X_train_scaled.shape}")
        print(f"   ✅ Clean test data: {self.X_test_scaled.shape}")
        print(f"   ✅ Training labels: {len(self.y_train)}")
        print(f"   ✅ Test labels: {len(self.y_test)}")
        print(f"   ✅ Fitted scaler: {type(self.scaler).__name__}")
        
        print("\n📋 Key Artifacts Available:")
        print("   - self.X_train_scaled: Scaled training features")
        print("   - self.X_test_scaled: Scaled test features") 
        print("   - self.y_train, self.y_test: Target variables")
        print("   - self.scaler: Fitted scaling transformer")
        print("   - self.preprocessing_report: Complete analysis report")
        print("   - self.label_encoders: Categorical encoders for inference")
        
        print("\n🚀 Ready for model training!")


if __name__ == '__main__':
    AdvancedPreprocessingFlow()