# Advanced Data Preprocessing Guide

## 🎯 Overview

This guide covers advanced data preprocessing techniques essential for building robust machine learning pipelines with Metaflow. You'll learn how to handle complex data scenarios, implement sophisticated feature engineering, and ensure data quality at scale.

## 📊 Data Quality Assessment

### Initial Data Exploration
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def comprehensive_data_summary(df):
    """Generate comprehensive data quality report"""
    print("=" * 50)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 50)
    
    # Basic info
    print(f"📏 Shape: {df.shape}")
    print(f"💾 Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Missing values analysis
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    
    if missing_data.sum() > 0:
        print("\n🚨 MISSING VALUES:")
        missing_summary = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing %': missing_pct
        }).sort_values('Missing %', ascending=False)
        print(missing_summary[missing_summary['Missing Count'] > 0])
    
    # Data types
    print(f"\n📋 DATA TYPES:")
    type_summary = df.dtypes.value_counts()
    for dtype, count in type_summary.items():
        print(f"  {dtype}: {count} columns")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\n🔄 DUPLICATES: {duplicates} rows ({duplicates/len(df)*100:.2f}%)")
    
    # Numerical column analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(f"\n📊 NUMERICAL COLUMNS: {len(numerical_cols)}")
        for col in numerical_cols:
            skewness = df[col].skew()
            print(f"  {col}: skew={skewness:.2f}", end="")
            if abs(skewness) > 1:
                print(" ⚠️ Highly skewed")
            elif abs(skewness) > 0.5:
                print(" ⚠️ Moderately skewed")
            else:
                print(" ✅ Normal distribution")
    
    # Categorical column analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"\n📝 CATEGORICAL COLUMNS: {len(categorical_cols)}")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count == len(df):
                print("    ⚠️ All unique values (possible ID column)")
            elif unique_count / len(df) > 0.5:
                print("    ⚠️ High cardinality")

# Usage
# comprehensive_data_summary(df)
```

### Outlier Detection
```python
def detect_outliers(df, method='iqr', factor=1.5):
    """
    Detect outliers using multiple methods
    
    Parameters:
    - method: 'iqr', 'zscore', 'isolation_forest'
    - factor: multiplier for IQR method
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = {}
    
    for col in numerical_cols:
        outliers = []
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df.iloc[np.where(z_scores > 3)].index
            
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(df[[col]].dropna())
            outliers = df[col].dropna().iloc[np.where(outlier_labels == -1)].index
        
        outlier_summary[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'indices': outliers
        }
    
    return outlier_summary

# Usage
# outliers = detect_outliers(df, method='iqr')
# for col, info in outliers.items():
#     print(f"{col}: {info['count']} outliers ({info['percentage']:.2f}%)")
```

## 🔧 Advanced Missing Value Handling

### Intelligent Imputation Strategies
```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class AdvancedImputer:
    """Advanced missing value imputation with multiple strategies"""
    
    def __init__(self):
        self.imputers = {}
        self.strategies = {}
    
    def fit_transform(self, df, strategy_map=None):
        """
        Apply different imputation strategies for different columns
        
        strategy_map example:
        {
            'age': 'knn',
            'income': 'iterative', 
            'category': 'mode',
            'score': 'median'
        }
        """
        if strategy_map is None:
            strategy_map = self._auto_strategy(df)
        
        df_imputed = df.copy()
        
        for column, strategy in strategy_map.items():
            if column not in df.columns:
                continue
                
            if df[column].isnull().sum() == 0:
                continue
            
            if strategy == 'knn':
                # KNN imputation for numerical data
                imputer = KNNImputer(n_neighbors=5)
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                if column in numerical_cols:
                    df_imputed[[column]] = imputer.fit_transform(df[[column]])
                    self.imputers[column] = imputer
            
            elif strategy == 'iterative':
                # Iterative imputation (mice-like)
                imputer = IterativeImputer(random_state=42, max_iter=10)
                df_imputed[[column]] = imputer.fit_transform(df[[column]])
                self.imputers[column] = imputer
            
            elif strategy in ['mean', 'median', 'most_frequent', 'constant']:
                imputer = SimpleImputer(strategy=strategy)
                df_imputed[[column]] = imputer.fit_transform(df[[column]])
                self.imputers[column] = imputer
            
            elif strategy == 'forward_fill':
                df_imputed[column] = df_imputed[column].fillna(method='ffill')
            
            elif strategy == 'backward_fill':
                df_imputed[column] = df_imputed[column].fillna(method='bfill')
        
        return df_imputed
    
    def _auto_strategy(self, df):
        """Automatically determine best imputation strategy for each column"""
        strategies = {}
        
        for column in df.columns:
            if df[column].isnull().sum() == 0:
                continue
            
            if df[column].dtype in ['object', 'category']:
                strategies[column] = 'most_frequent'
            elif df[column].dtype in ['int64', 'float64']:
                # Check distribution for numerical columns
                skewness = abs(df[column].skew())
                if skewness > 1:
                    strategies[column] = 'median'  # Skewed data
                else:
                    strategies[column] = 'mean'    # Normal data
        
        return strategies

# Usage
# imputer = AdvancedImputer()
# df_imputed = imputer.fit_transform(df, {
#     'age': 'knn',
#     'income': 'median',
#     'category': 'most_frequent'
# })
```

### Missing Value Pattern Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing_patterns(df):
    """Analyze patterns in missing data"""
    
    # Missing value heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Value Patterns')
    plt.show()
    
    # Missing value correlation
    missing_df = df.isnull().astype(int)
    missing_corr = missing_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Missing Value Correlations')
    plt.show()
    
    # Missing value summary by column
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    }).sort_values('Missing_Percentage', ascending=False)
    
    return missing_summary

# Usage
# missing_analysis = analyze_missing_patterns(df)
# print(missing_analysis)
```

## 🎯 Feature Engineering Techniques

### Automated Feature Creation
```python
class FeatureEngineer:
    """Comprehensive feature engineering toolkit"""
    
    def __init__(self):
        self.feature_names = []
        self.transformations = {}
    
    def create_polynomial_features(self, df, columns, degree=2):
        """Create polynomial features for numerical columns"""
        from sklearn.preprocessing import PolynomialFeatures
        
        df_new = df.copy()
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_features = poly.fit_transform(df[[col]])
                
                # Create feature names
                feature_names = [f"{col}_degree_{i}" for i in range(1, degree+1)]
                
                for i, name in enumerate(feature_names[1:], 1):  # Skip original feature
                    df_new[name] = poly_features[:, i]
                    self.feature_names.append(name)
        
        return df_new
    
    def create_interaction_features(self, df, column_pairs):
        """Create interaction features between column pairs"""
        df_new = df.copy()
        
        for col1, col2 in column_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Multiplication interaction
                interaction_name = f"{col1}_x_{col2}"
                df_new[interaction_name] = df[col1] * df[col2]
                self.feature_names.append(interaction_name)
                
                # Division interaction (avoid division by zero)
                if (df[col2] != 0).all():
                    division_name = f"{col1}_div_{col2}"
                    df_new[division_name] = df[col1] / df[col2]
                    self.feature_names.append(division_name)
        
        return df_new
    
    def create_aggregation_features(self, df, group_col, target_cols, agg_funcs=['mean', 'std', 'count']):
        """Create aggregation features grouped by a categorical column"""
        df_new = df.copy()
        
        for target_col in target_cols:
            if target_col in df.columns and group_col in df.columns:
                for func in agg_funcs:
                    agg_values = df.groupby(group_col)[target_col].transform(func)
                    feature_name = f"{target_col}_{func}_by_{group_col}"
                    df_new[feature_name] = agg_values
                    self.feature_names.append(feature_name)
        
        return df_new
    
    def create_binning_features(self, df, columns, bins=5, strategy='quantile'):
        """Create binned versions of numerical features"""
        df_new = df.copy()
        
        for col in columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                if strategy == 'quantile':
                    df_new[f"{col}_binned"] = pd.qcut(df[col], bins, labels=False, duplicates='drop')
                elif strategy == 'uniform':
                    df_new[f"{col}_binned"] = pd.cut(df[col], bins, labels=False)
                
                self.feature_names.append(f"{col}_binned")
        
        return df_new
    
    def create_date_features(self, df, date_columns):
        """Extract features from datetime columns"""
        df_new = df.copy()
        
        for col in date_columns:
            if col in df.columns:
                # Ensure datetime format
                df_new[col] = pd.to_datetime(df[col])
                
                # Extract features
                df_new[f"{col}_year"] = df_new[col].dt.year
                df_new[f"{col}_month"] = df_new[col].dt.month
                df_new[f"{col}_day"] = df_new[col].dt.day
                df_new[f"{col}_dayofweek"] = df_new[col].dt.dayofweek
                df_new[f"{col}_quarter"] = df_new[col].dt.quarter
                df_new[f"{col}_is_weekend"] = (df_new[col].dt.dayofweek >= 5).astype(int)
                
                # Add to feature names
                date_features = [f"{col}_year", f"{col}_month", f"{col}_day", 
                               f"{col}_dayofweek", f"{col}_quarter", f"{col}_is_weekend"]
                self.feature_names.extend(date_features)
        
        return df_new

# Usage
# fe = FeatureEngineer()
# df_enhanced = fe.create_polynomial_features(df, ['age', 'income'], degree=2)
# df_enhanced = fe.create_interaction_features(df_enhanced, [('age', 'income')])
# print(f"Created {len(fe.feature_names)} new features")
```

### Text Feature Engineering
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import string

class TextFeatureEngineer:
    """Advanced text preprocessing and feature extraction"""
    
    def __init__(self):
        self.vectorizers = {}
    
    def clean_text(self, text):
        """Comprehensive text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (optional)
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_text_features(self, df, text_column):
        """Extract statistical features from text"""
        df_new = df.copy()
        
        # Basic text statistics
        df_new[f"{text_column}_length"] = df[text_column].astype(str).str.len()
        df_new[f"{text_column}_word_count"] = df[text_column].astype(str).str.split().str.len()
        df_new[f"{text_column}_char_count"] = df[text_column].astype(str).str.len()
        df_new[f"{text_column}_sentence_count"] = df[text_column].astype(str).str.count(r'[.!?]') + 1
        
        # Advanced features
        df_new[f"{text_column}_avg_word_length"] = (
            df_new[f"{text_column}_char_count"] / df_new[f"{text_column}_word_count"]
        )
        
        df_new[f"{text_column}_uppercase_ratio"] = (
            df[text_column].astype(str).str.count(r'[A-Z]') / df_new[f"{text_column}_char_count"]
        )
        
        df_new[f"{text_column}_punctuation_ratio"] = (
            df[text_column].astype(str).str.count(r'[^\w\s]') / df_new[f"{text_column}_char_count"]
        )
        
        return df_new
    
    def create_tfidf_features(self, df, text_column, max_features=100):
        """Create TF-IDF features from text"""
        # Clean text
        clean_texts = df[text_column].astype(str).apply(self.clean_text)
        
        # Create TF-IDF features
        tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(clean_texts)
        
        # Convert to DataFrame
        feature_names = [f"tfidf_{word}" for word in tfidf.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df.index)
        
        # Store vectorizer for future use
        self.vectorizers[text_column] = tfidf
        
        return pd.concat([df, tfidf_df], axis=1)

# Usage
# text_fe = TextFeatureEngineer()
# df_text_features = text_fe.extract_text_features(df, 'description')
# df_tfidf = text_fe.create_tfidf_features(df, 'description', max_features=50)
```

## ⚖️ Data Scaling and Normalization

### Comprehensive Scaling Strategies
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

class AdvancedScaler:
    """Intelligent scaling with automatic method selection"""
    
    def __init__(self):
        self.scalers = {}
        self.scaling_methods = {}
    
    def fit_transform(self, df, method_map=None):
        """
        Apply different scaling methods to different columns
        
        method_map example:
        {
            'age': 'standard',
            'income': 'robust',  # for outliers
            'score': 'minmax'
        }
        """
        if method_map is None:
            method_map = self._auto_select_scaling(df)
        
        df_scaled = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for column in numerical_cols:
            if column not in method_map:
                continue
            
            method = method_map[column]
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'power':
                scaler = PowerTransformer(method='yeo-johnson')
            else:
                continue
            
            df_scaled[[column]] = scaler.fit_transform(df[[column]])
            self.scalers[column] = scaler
            self.scaling_methods[column] = method
        
        return df_scaled
    
    def _auto_select_scaling(self, df):
        """Automatically select best scaling method for each column"""
        methods = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for column in numerical_cols:
            # Check for outliers using IQR
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            outlier_ratio = outliers / len(df)
            
            # Check skewness
            skewness = abs(df[column].skew())
            
            if outlier_ratio > 0.1:  # High outlier presence
                methods[column] = 'robust'
            elif skewness > 2:  # Highly skewed
                methods[column] = 'power'
            elif df[column].min() >= 0:  # All positive values
                methods[column] = 'minmax'
            else:
                methods[column] = 'standard'
        
        return methods
    
    def transform(self, df):
        """Transform new data using fitted scalers"""
        df_scaled = df.copy()
        
        for column, scaler in self.scalers.items():
            if column in df.columns:
                df_scaled[[column]] = scaler.transform(df[[column]])
        
        return df_scaled

# Usage
# scaler = AdvancedScaler()
# df_scaled = scaler.fit_transform(df)
# print("Scaling methods used:", scaler.scaling_methods)
```

## 🔄 Metaflow Pipeline Integration

### Complete Preprocessing Pipeline
```python
from metaflow import FlowSpec, step, Parameter, IncludeFile, catch

class AdvancedPreprocessingFlow(FlowSpec):
    """
    Complete data preprocessing pipeline with advanced techniques
    """
    
    # Parameters
    input_file = Parameter('input_file', 
                          help='Path to input CSV file',
                          default='data.csv')
    
    test_size = Parameter('test_size',
                         help='Test set size ratio',
                         default=0.2)
    
    @step
    def start(self):
        """Load and validate input data"""
        import pandas as pd
        
        print("🚀 Starting advanced preprocessing pipeline...")
        
        # Load data
        self.df = pd.read_csv(self.input_file)
        
        print(f"📊 Loaded data: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        self.next(self.data_quality_check)
    
    @step
    def data_quality_check(self):
        """Comprehensive data quality assessment"""
        
        # Run quality checks
        self.quality_report = self._generate_quality_report()
        
        # Detect outliers
        self.outliers = detect_outliers(self.df, method='iqr')
        
        # Check for high cardinality categorical variables
        self.high_cardinality_cols = self._identify_high_cardinality()
        
        print("✅ Data quality check complete")
        self.next(self.handle_missing_values)
    
    @step
    def handle_missing_values(self):
        """Advanced missing value handling"""
        
        # Create imputation strategy
        strategy_map = {}
        
        for column in self.df.columns:
            missing_pct = (self.df[column].isnull().sum() / len(self.df)) * 100
            
            if missing_pct == 0:
                continue
            elif missing_pct > 50:
                # Too many missing values - consider dropping
                print(f"⚠️ {column}: {missing_pct:.1f}% missing - consider dropping")
                continue
            elif self.df[column].dtype in ['object', 'category']:
                strategy_map[column] = 'most_frequent'
            elif self.df[column].dtype in ['int64', 'float64']:
                if abs(self.df[column].skew()) > 1:
                    strategy_map[column] = 'median'
                else:
                    strategy_map[column] = 'knn'
        
        # Apply imputation
        imputer = AdvancedImputer()
        self.df_imputed = imputer.fit_transform(self.df, strategy_map)
        self.imputation_strategy = strategy_map
        
        print(f"✅ Handled missing values: {len(strategy_map)} columns processed")
        self.next(self.feature_engineering)
    
    @step 
    def feature_engineering(self):
        """Create advanced features"""
        
        fe = FeatureEngineer()
        self.df_features = self.df_imputed.copy()
        
        # Identify numerical and categorical columns
        numerical_cols = self.df_features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df_features.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create polynomial features for key numerical columns
        if len(numerical_cols) >= 2:
            key_numericals = numerical_cols[:3]  # Limit to prevent explosion
            self.df_features = fe.create_polynomial_features(
                self.df_features, key_numericals, degree=2
            )
        
        # Create interaction features for selected pairs
        if len(numerical_cols) >= 2:
            interaction_pairs = [(numerical_cols[0], numerical_cols[1])]
            self.df_features = fe.create_interaction_features(
                self.df_features, interaction_pairs
            )
        
        # Create aggregation features if categorical columns exist
        if len(categorical_cols) > 0 and len(numerical_cols) > 0:
            self.df_features = fe.create_aggregation_features(
                self.df_features, 
                categorical_cols[0], 
                numerical_cols[:2],
                agg_funcs=['mean', 'std']
            )
        
        self.new_features = fe.feature_names
        print(f"✅ Created {len(self.new_features)} new features")
        
        self.next(self.handle_outliers)
    
    @step
    def handle_outliers(self):
        """Handle outliers in the data"""
        
        # For each numerical column with outliers
        self.df_clean = self.df_features.copy()
        outlier_handling_report = {}
        
        for column, outlier_info in self.outliers.items():
            if outlier_info['percentage'] > 5:  # More than 5% outliers
                # Cap outliers using IQR method
                Q1 = self.df_clean[column].quantile(0.25)
                Q3 = self.df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap values
                original_outliers = outlier_info['count']
                self.df_clean[column] = np.clip(self.df_clean[column], lower_bound, upper_bound)
                
                outlier_handling_report[column] = {
                    'original_outliers': original_outliers,
                    'method': 'capping',
                    'bounds': (lower_bound, upper_bound)
                }
        
        self.outlier_handling = outlier_handling_report
        print(f"✅ Handled outliers in {len(outlier_handling_report)} columns")
        
        self.next(self.scale_features)
    
    @step
    def scale_features(self):
        """Apply intelligent scaling"""
        
        # Apply scaling
        scaler = AdvancedScaler()
        self.df_scaled = scaler.fit_transform(self.df_clean)
        self.scaling_methods = scaler.scaling_methods
        self.fitted_scalers = scaler.scalers
        
        print(f"✅ Scaled features using: {set(self.scaling_methods.values())}")
        
        self.next(self.encode_categorical)
    
    @step
    def encode_categorical(self):
        """Encode categorical variables"""
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        
        self.df_encoded = self.df_scaled.copy()
        self.encoders = {}
        
        categorical_cols = self.df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_cols:
            unique_count = self.df_encoded[column].nunique()
            
            if unique_count <= 10:  # Low cardinality - use one-hot encoding
                encoded_cols = pd.get_dummies(self.df_encoded[column], prefix=column)
                self.df_encoded = pd.concat([self.df_encoded, encoded_cols], axis=1)
                self.df_encoded.drop(column, axis=1, inplace=True)
                self.encoders[column] = 'onehot'
            else:  # High cardinality - use label encoding
                le = LabelEncoder()
                self.df_encoded[column] = le.fit_transform(self.df_encoded[column].astype(str))
                self.encoders[column] = le
        
        print(f"✅ Encoded {len(categorical_cols)} categorical columns")
        
        self.next(self.split_data)
    
    @step
    def split_data(self):
        """Split data into train/test sets"""
        from sklearn.model_selection import train_test_split
        
        # Assume last column is target (adjust as needed)
        X = self.df_encoded.iloc[:, :-1]
        y = self.df_encoded.iloc[:, -1]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        
        print(f"✅ Split data: Train={len(self.X_train)}, Test={len(self.X_test)}")
        
        self.next(self.generate_report)
    
    @step
    def generate_report(self):
        """Generate comprehensive preprocessing report"""
        
        self.preprocessing_report = {
            'original_shape': self.df.shape,
            'final_shape': self.df_encoded.shape,
            'features_created': len(self.new_features),
            'missing_value_strategy': self.imputation_strategy,
            'outlier_handling': self.outlier_handling,
            'scaling_methods': self.scaling_methods,
            'encoding_methods': self.encoders,
            'train_test_split': {
                'train_size': len(self.X_train),
                'test_size': len(self.X_test),
                'test_ratio': self.test_size
            }
        }
        
        print("📋 PREPROCESSING REPORT")
        print("=" * 40)
        print(f"Original shape: {self.preprocessing_report['original_shape']}")
        print(f"Final shape: {self.preprocessing_report['final_shape']}")
        print(f"New features created: {self.preprocessing_report['features_created']}")
        print(f"Columns with missing values handled: {len(self.imputation_strategy)}")
        print(f"Columns with outliers handled: {len(self.outlier_handling)}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """Pipeline completion"""
        print("🎉 Advanced preprocessing pipeline completed successfully!")
        print(f"Final dataset ready for modeling: {self.df_encoded.shape}")
    
    # Helper methods
    def _generate_quality_report(self):
        """Generate data quality report"""
        return {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'duplicates': self.df.duplicated().sum()
        }
    
    def _identify_high_cardinality(self):
        """Identify high cardinality categorical columns"""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = []
        
        for col in categorical_cols:
            if self.df[col].nunique() > 50:
                high_cardinality.append(col)
        
        return high_cardinality

if __name__ == '__main__':
    AdvancedPreprocessingFlow()
```

## 💡 Best Practices

### Pipeline Design Principles
1. **Modularity**: Each preprocessing step should be independent and reusable
2. **Reproducibility**: Use fixed random seeds and version control
3. **Validation**: Always validate data quality before and after transformations
4. **Documentation**: Log all transformations and their parameters
5. **Efficiency**: Use vectorized operations and appropriate data types

### Common Pitfalls to Avoid
- **Data leakage**: Don't use future information in feature engineering
- **Overfitting**: Be careful with too many engineered features
- **Scale mismatch**: Always scale features consistently
- **Missing value propagation**: Handle missing values before feature engineering
- **Categorical explosion**: Be cautious with high-cardinality one-hot encoding

### Performance Optimization
```python
# Memory-efficient data types
def optimize_dtypes(df):
    """Optimize data types to reduce memory usage"""
    
    for col in df.columns:
        if df[col].dtype == 'int64':
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() > -128 and df[col].max() < 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype('int32')
        
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        
        elif df[col].dtype == 'object':
            # Try to convert to category if reasonable number of unique values
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
    
    return df
```

---

**🎯 Quick Checklist for Data Preprocessing:**
- [ ] Assess data quality and missing value patterns
- [ ] Handle missing values with appropriate strategies
- [ ] Detect and handle outliers
- [ ] Create meaningful features through engineering
- [ ] Scale numerical features appropriately
- [ ] Encode categorical variables
- [ ] Validate final dataset quality
- [ ] Document all transformations for reproducibility