# Scikit-learn Primer for AI/ML Interns

## What is Scikit-learn?

Scikit-learn (sklearn) is the most popular machine learning library for Python, providing simple and efficient tools for data analysis and modeling. It's built on NumPy, SciPy, and matplotlib, making it seamlessly integrate with the scientific Python ecosystem.

**Key Benefits:**
- **Consistent API**: All algorithms follow the same interface pattern
- **Comprehensive Coverage**: Classification, regression, clustering, dimensionality reduction
- **Production Ready**: Well-tested, optimized algorithms suitable for real applications
- **Excellent Documentation**: Clear examples and extensive user guides
- **Active Community**: Large community with extensive tutorials and support

**Why Scikit-learn for This Program:**
- Foundation for traditional machine learning (complements LangChain for LLMs)
- Industry standard for classical ML algorithms
- Essential for comparing traditional ML vs. generative AI approaches
- Required knowledge for most data science positions

---

## Installation and Setup

### Basic Installation

```bash
# Install scikit-learn
pip install scikit-learn

# Or with conda
conda install scikit-learn

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

### Complete Setup for ML Development

```bash
# Install full ML stack
pip install scikit-learn pandas numpy matplotlib seaborn jupyter

# Optional: Install additional tools
pip install scikit-plot  # For advanced plotting
pip install imbalanced-learn  # For handling imbalanced datasets
pip install yellowbrick  # For ML visualizations
```

### Environment Verification

```python
# verify_sklearn_setup.py
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def verify_setup():
    """Verify scikit-learn setup with a simple example."""
    print(f"✅ Scikit-learn version: {sklearn.__version__}")
    
    # Load sample data
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✅ Sample model accuracy: {accuracy:.3f}")
    print("✅ Scikit-learn setup complete!")
    
    return True

if __name__ == "__main__":
    verify_setup()
```

---

## Core Concepts and API Design

### The Scikit-learn API Pattern

Scikit-learn follows a consistent API pattern across all algorithms:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Import the algorithm
from sklearn.linear_model import LogisticRegression

# 2. Create an instance (estimator)
model = LogisticRegression(random_state=42)

# 3. Fit the model to training data
model.fit(X_train, y_train)

# 4. Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 5. Evaluate performance
score = model.score(X_test, y_test)
```

### Key Concepts

**Estimators**: Objects that learn from data (all ML algorithms)
**Transformers**: Objects that transform data (preprocessing, feature selection)
**Predictors**: Estimators that can make predictions
**Meta-estimators**: Estimators that take other estimators as parameters (pipelines, ensembles)

---

## Data Preprocessing

### 1. Loading and Exploring Data

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_iris, make_classification
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Comprehensive data preprocessing examples."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
    
    def load_sample_data(self):
        """Load and examine sample datasets."""
        
        # Built-in datasets
        iris = load_iris()
        print("Iris dataset:")
        print(f"Features: {iris.feature_names}")
        print(f"Target: {iris.target_names}")
        print(f"Shape: {iris.data.shape}")
        
        # Create synthetic dataset
        X_synthetic, y_synthetic = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42
        )
        
        print(f"\nSynthetic dataset shape: {X_synthetic.shape}")
        
        return iris, (X_synthetic, y_synthetic)
    
    def handle_missing_values(self, X):
        """Demonstrate missing value handling strategies."""
        from sklearn.impute import SimpleImputer, KNNImputer
        
        # Create dataset with missing values
        X_missing = X.copy()
        np.random.seed(42)
        missing_mask = np.random.random(X.shape) < 0.1  # 10% missing
        X_missing[missing_mask] = np.nan
        
        print(f"Created {np.sum(missing_mask)} missing values")
        
        # Strategy 1: Simple imputation
        simple_imputer = SimpleImputer(strategy='mean')
        X_simple = simple_imputer.fit_transform(X_missing)
        
        # Strategy 2: KNN imputation
        knn_imputer = KNNImputer(n_neighbors=5)
        X_knn = knn_imputer.fit_transform(X_missing)
        
        print("Missing value imputation completed")
        
        return {
            'original_with_missing': X_missing,
            'simple_imputed': X_simple,
            'knn_imputed': X_knn
        }
    
    def scale_features(self, X_train, X_test):
        """Demonstrate different scaling techniques."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        scaled_data = {}
        
        for name, scaler in scalers.items():
            # Fit on training data only
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            scaled_data[name] = {
                'train': X_train_scaled,
                'test': X_test_scaled,
                'scaler': scaler
            }
            
            print(f"{name.capitalize()} scaling:")
            print(f"  Train mean: {X_train_scaled.mean():.3f}, std: {X_train_scaled.std():.3f}")
        
        return scaled_data
    
    def encode_categorical_features(self):
        """Demonstrate categorical encoding techniques."""
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
        
        # Create sample categorical data
        data = pd.DataFrame({
            'color': ['red', 'blue', 'green', 'red', 'blue', 'green'],
            'size': ['small', 'medium', 'large', 'medium', 'small', 'large'],
            'quality': ['poor', 'fair', 'good', 'excellent', 'fair', 'good']
        })
        
        print("Original categorical data:")
        print(data)
        
        # Label Encoding
        label_encoder = LabelEncoder()
        data['color_label'] = label_encoder.fit_transform(data['color'])
        
        # One-Hot Encoding
        onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        color_onehot = onehot_encoder.fit_transform(data[['color']])
        
        # Ordinal Encoding (when order matters)
        quality_order = [['poor', 'fair', 'good', 'excellent']]
        ordinal_encoder = OrdinalEncoder(categories=quality_order)
        data['quality_ordinal'] = ordinal_encoder.fit_transform(data[['quality']])
        
        print("\nAfter encoding:")
        print(data[['color', 'color_label', 'quality', 'quality_ordinal']])
        
        return data, {'label': label_encoder, 'onehot': onehot_encoder, 'ordinal': ordinal_encoder}

# Demonstrate preprocessing
preprocessor = DataPreprocessor()
iris_data, synthetic_data = preprocessor.load_sample_data()

# Handle missing values
X, y = synthetic_data
missing_results = preprocessor.handle_missing_values(X)

# Feature scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaling_results = preprocessor.scale_features(X_train, X_test)

# Categorical encoding
categorical_data, encoders = preprocessor.encode_categorical_features()
```

### 2. Feature Selection and Engineering

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

class FeatureEngineer:
    """Feature selection and engineering techniques."""
    
    def __init__(self):
        pass
    
    def statistical_feature_selection(self, X, y, k=5):
        """Select features using statistical tests."""
        
        # Univariate feature selection
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature indices
        selected_features = selector.get_support(indices=True)
        feature_scores = selector.scores_
        
        print(f"Selected {k} best features from {X.shape[1]} total features")
        print(f"Selected feature indices: {selected_features}")
        print(f"Feature scores: {feature_scores[selected_features]}")
        
        return X_selected, selected_features, feature_scores
    
    def recursive_feature_elimination(self, X, y, n_features=5):
        """Use RFE with a model to select features."""
        
        # Use Random Forest as the estimator
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Recursive Feature Elimination
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        X_rfe = rfe.fit_transform(X, y)
        
        selected_features = rfe.get_support(indices=True)
        feature_ranking = rfe.ranking_
        
        print(f"RFE selected features: {selected_features}")
        print(f"Feature ranking: {feature_ranking}")
        
        return X_rfe, selected_features, feature_ranking
    
    def polynomial_features(self, X, degree=2):
        """Create polynomial features."""
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        feature_names = poly.get_feature_names_out()
        
        print(f"Original features: {X.shape[1]}")
        print(f"Polynomial features (degree {degree}): {X_poly.shape[1]}")
        print(f"Sample feature names: {feature_names[:10]}")
        
        return X_poly, feature_names
    
    def feature_importance_analysis(self, X, y):
        """Analyze feature importance using tree-based models."""
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': range(X.shape[1]),
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("Top 10 most important features:")
        print(feature_importance.head(10))
        
        return feature_importance

# Demonstrate feature engineering
feature_eng = FeatureEngineer()

# Use synthetic data from previous example
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                          n_redundant=5, random_state=42)

# Statistical feature selection
X_stat, selected_stat, scores_stat = feature_eng.statistical_feature_selection(X, y, k=10)

# Recursive feature elimination
X_rfe, selected_rfe, ranking_rfe = feature_eng.recursive_feature_elimination(X, y, n_features=8)

# Polynomial features
X_poly, poly_names = feature_eng.polynomial_features(X[:, :5], degree=2)  # Use subset for demo

# Feature importance analysis
importance_df = feature_eng.feature_importance_analysis(X, y)
```

---

## Supervised Learning

### 1. Classification Algorithms

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ClassificationAlgorithms:
    """Comprehensive classification algorithm examples."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def setup_algorithms(self):
        """Initialize different classification algorithms."""
        
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'naive_bayes': GaussianNB()
        }
        
        print(f"Initialized {len(self.models)} classification algorithms")
        return self.models
    
    def train_and_evaluate_all(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all classification algorithms."""
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
        
        self.results = results
        return results
    
    def detailed_evaluation(self, X_test, y_test, model_name='random_forest'):
        """Provide detailed evaluation for a specific model."""
        
        if model_name not in self.results:
            print(f"Model {model_name} not found in results")
            return
        
        model_results = self.results[model_name]
        y_pred = model_results['predictions']
        
        print(f"\nDetailed evaluation for {model_name}:")
        print("=" * 50)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm
        }
    
    def compare_algorithms(self):
        """Compare all algorithms and rank by performance."""
        
        comparison = []
        for name, results in self.results.items():
            comparison.append({
                'algorithm': name,
                'accuracy': results['accuracy']
            })
        
        comparison_df = pd.DataFrame(comparison).sort_values('accuracy', ascending=False)
        
        print("\nAlgorithm Comparison (by accuracy):")
        print(comparison_df)
        
        return comparison_df
    
    def hyperparameter_tuning_example(self, X_train, y_train):
        """Demonstrate hyperparameter tuning with GridSearch."""
        from sklearn.model_selection import GridSearchCV
        
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create Random Forest classifier
        rf = RandomForestClassifier(random_state=42)
        
        # Grid Search with Cross-Validation
        grid_search = GridSearchCV(
            rf, 
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=1
        )
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_

# Demonstrate classification algorithms
# Load a real dataset for demonstration
from sklearn.datasets import load_wine

wine = load_wine()
X, y = wine.data, wine.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train classifiers
classifier = ClassificationAlgorithms()
models = classifier.setup_algorithms()
results = classifier.train_and_evaluate_all(X_train_scaled, X_test_scaled, y_train, y_test)

# Detailed evaluation
detailed_results = classifier.detailed_evaluation(X_test_scaled, y_test, 'random_forest')

# Compare all algorithms
comparison = classifier.compare_algorithms()

# Hyperparameter tuning example
best_rf, best_params = classifier.hyperparameter_tuning_example(X_train_scaled, y_train)
```

### 2. Regression Algorithms

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionAlgorithms:
    """Comprehensive regression algorithm examples."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def setup_algorithms(self):
        """Initialize different regression algorithms."""
        
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=1.0, random_state=42),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
            'decision_tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf'),
            'knn': KNeighborsRegressor(n_neighbors=5)
        }
        
        print(f"Initialized {len(self.models)} regression algorithms")
        return self.models
    
    def train_and_evaluate_all(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all regression algorithms."""
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Store results
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"  RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        self.results = results
        return results
    
    def compare_algorithms(self):
        """Compare all algorithms and rank by performance."""
        
        comparison = []
        for name, results in self.results.items():
            comparison.append({
                'algorithm': name,
                'rmse': results['rmse'],
                'mae': results['mae'],
                'r2': results['r2']
            })
        
        comparison_df = pd.DataFrame(comparison).sort_values('r2', ascending=False)
        
        print("\nAlgorithm Comparison (by R² score):")
        print(comparison_df)
        
        return comparison_df
    
    def regularization_comparison(self, X_train, X_test, y_train, y_test):
        """Compare different regularization techniques."""
        
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        ridge_scores = []
        lasso_scores = []
        
        for alpha in alphas:
            # Ridge regression
            ridge = Ridge(alpha=alpha, random_state=42)
            ridge.fit(X_train, y_train)
            ridge_score = ridge.score(X_test, y_test)
            ridge_scores.append(ridge_score)
            
            # Lasso regression
            lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
            lasso.fit(X_train, y_train)
            lasso_score = lasso.score(X_test, y_test)
            lasso_scores.append(lasso_score)
        
        regularization_results = pd.DataFrame({
            'alpha': alphas,
            'ridge_r2': ridge_scores,
            'lasso_r2': lasso_scores
        })
        
        print("Regularization Comparison:")
        print(regularization_results)
        
        return regularization_results
    
    def feature_importance_regression(self, model_name='random_forest'):
        """Analyze feature importance for tree-based models."""
        
        if model_name not in self.results:
            print(f"Model {model_name} not found")
            return
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': range(len(importance)),
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"Feature importance for {model_name}:")
            print(feature_importance.head(10))
            
            return feature_importance
        else:
            print(f"Model {model_name} does not support feature importance")
            return None

# Demonstrate regression algorithms
from sklearn.datasets import fetch_california_housing

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train regressors
regressor = RegressionAlgorithms()
models = regressor.setup_algorithms()
results = regressor.train_and_evaluate_all(X_train_scaled, X_test_scaled, y_train, y_test)

# Compare algorithms
comparison = regressor.compare_algorithms()

# Regularization comparison
reg_comparison = regressor.regularization_comparison(X_train_scaled, X_test_scaled, y_train, y_test)

# Feature importance
feature_importance = regressor.feature_importance_regression('random_forest')
```

---

## Unsupervised Learning

### 1. Clustering Algorithms

```python
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

class ClusteringAlgorithms:
    """Comprehensive clustering algorithm examples."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def setup_algorithms(self, n_clusters=3):
        """Initialize different clustering algorithms."""
        
        self.models = {
            'kmeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            'hierarchical': AgglomerativeClustering(n_clusters=n_clusters),
            'dbscan': DBSCAN(eps=0.5, min_samples=5),
            'gaussian_mixture': GaussianMixture(n_components=n_clusters, random_state=42)
        }
        
        print(f"Initialized {len(self.models)} clustering algorithms")
        return self.models
    
    def optimal_clusters_analysis(self, X, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        
        # Elbow method for K-means
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Create results DataFrame
        cluster_analysis = pd.DataFrame({
            'n_clusters': k_range,
            'inertia': inertias,
            'silhouette_score': silhouette_scores
        })
        
        print("Optimal Clusters Analysis:")
        print(cluster_analysis)
        
        # Find optimal k (highest silhouette score)
        optimal_k = cluster_analysis.loc[cluster_analysis['silhouette_score'].idxmax(), 'n_clusters']
        print(f"\nOptimal number of clusters (by silhouette score): {optimal_k}")
        
        return cluster_analysis, optimal_k
    
    def perform_clustering(self, X):
        """Perform clustering with all algorithms."""
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nPerforming {name} clustering...")
            
            # Fit the model and get cluster labels
            if hasattr(model, 'fit_predict'):
                cluster_labels = model.fit_predict(X)
            else:
                model.fit(X)
                cluster_labels = model.predict(X)
            
            # Calculate silhouette score (if more than 1 cluster)
            n_clusters = len(np.unique(cluster_labels))
            if n_clusters > 1:
                silhouette_avg = silhouette_score(X, cluster_labels)
            else:
                silhouette_avg = -1
            
            # Store results
            results[name] = {
                'model': model,
                'labels': cluster_labels,
                'n_clusters': n_clusters,
                'silhouette_score': silhouette_avg
            }
            
            print(f"  Clusters found: {n_clusters}")
            print(f"  Silhouette score: {silhouette_avg:.4f}")
        
        self.results = results
        return results
    
    def visualize_clusters_2d(self, X, model_name='kmeans'):
        """Visualize clusters in 2D using PCA."""
        
        if model_name not in self.results:
            print(f"Model {model_name} not found")
            return
        
        # Reduce to 2D using PCA for visualization
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
            explained_variance = pca.explained_variance_ratio_.sum()
            print(f"PCA explained variance: {explained_variance:.3f}")
        else:
            X_2d = X
        
        # Get cluster labels
        labels = self.results[model_name]['labels']
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Plot points colored by cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:  # Noise points (for DBSCAN)
                color = 'black'
                marker = 'x'
                label_name = 'Noise'
            else:
                marker = 'o'
                label_name = f'Cluster {label}'
            
            mask = labels == label
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=[color], marker=marker, s=50, label=label_name, alpha=0.7)
        
        plt.title(f'{model_name.title()} Clustering Results')
        plt.xlabel('First Principal Component' if X.shape[1] > 2 else 'Feature 1')
        plt.ylabel('Second Principal Component' if X.shape[1] > 2 else 'Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return X_2d
    
    def cluster_analysis_report(self, X):
        """Generate comprehensive cluster analysis report."""
        
        report = {}
        
        for name, results in self.results.items():
            labels = results['labels']
            model = results['model']
            
            # Basic statistics
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Cluster centers (if available)
            if hasattr(model, 'cluster_centers_'):
                centers = model.cluster_centers_
            elif hasattr(model, 'means_'):  # Gaussian Mixture
                centers = model.means_
            else:
                centers = None
            
            report[name] = {
                'n_clusters': len(unique_labels),
                'cluster_sizes': dict(zip(unique_labels, counts)),
                'silhouette_score': results['silhouette_score'],
                'cluster_centers': centers
            }
        
        print("Clustering Analysis Report:")
        print("=" * 50)
        
        for name, info in report.items():
            print(f"\n{name.upper()}:")
            print(f"  Number of clusters: {info['n_clusters']}")
            print(f"  Cluster sizes: {info['cluster_sizes']}")
            print(f"  Silhouette score: {info['silhouette_score']:.4f}")
        
        return report

# Demonstrate clustering algorithms
# Create sample data for clustering
from sklearn.datasets import make_blobs, make_circles

# Generate sample data
X_blobs, y_true_blobs = make_blobs(n_samples=300, centers=4, n_features=2, 
                                  random_state=42, cluster_std=0.60)

print("Generated sample clustering data")
print(f"Data shape: {X_blobs.shape}")

# Initialize clustering
clustering = ClusteringAlgorithms()

# Find optimal number of clusters
cluster_analysis, optimal_k = clustering.optimal_clusters_analysis(X_blobs, max_clusters=8)

# Setup algorithms with optimal k
models = clustering.setup_algorithms(n_clusters=optimal_k)

# Perform clustering
results = clustering.perform_clustering(X_blobs)

# Visualize results
X_2d = clustering.visualize_clusters_2d(X_blobs, 'kmeans')

# Generate comprehensive report
report = clustering.cluster_analysis_report(X_blobs)
```

### 2. Dimensionality Reduction

```python
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.manifold import TSNE, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class DimensionalityReduction:
    """Comprehensive dimensionality reduction techniques."""
    
    def __init__(self):
        self.reducers = {}
        self.results = {}
    
    def setup_reducers(self, n_components=2):
        """Initialize different dimensionality reduction algorithms."""
        
        self.reducers = {
            'pca': PCA(n_components=n_components, random_state=42),
            'truncated_svd': TruncatedSVD(n_components=n_components, random_state=42),
            'factor_analysis': FactorAnalysis(n_components=n_components, random_state=42),
            'tsne': TSNE(n_components=n_components, random_state=42, perplexity=30),
            'mds': MDS(n_components=n_components, random_state=42)
        }
        
        print(f"Initialized {len(self.reducers)} dimensionality reduction algorithms")
        return self.reducers
    
    def pca_analysis(self, X):
        """Detailed PCA analysis with explained variance."""
        
        # Fit PCA with all components first
        pca_full = PCA()
        pca_full.fit(X)
        
        # Calculate cumulative explained variance
        explained_variance_ratio = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print("PCA Analysis:")
        print(f"Original dimensions: {X.shape[1]}")
        print(f"Components for 95% variance: {n_components_95}")
        print(f"First 10 components explained variance: {explained_variance_ratio[:10]}")
        
        # Create PCA analysis DataFrame
        pca_df = pd.DataFrame({
            'component': range(1, min(21, len(explained_variance_ratio) + 1)),
            'explained_variance_ratio': explained_variance_ratio[:20],
            'cumulative_variance': cumulative_variance[:20]
        })
        
        return pca_df, n_components_95
    
    def apply_all_reductions(self, X):
        """Apply all dimensionality reduction techniques."""
        
        results = {}
        
        for name, reducer in self.reducers.items():
            print(f"\nApplying {name}...")
            
            try:
                # Apply dimensionality reduction
                X_reduced = reducer.fit_transform(X)
                
                # Calculate explained variance if available
                if hasattr(reducer, 'explained_variance_ratio_'):
                    explained_variance = reducer.explained_variance_ratio_.sum()
                else:
                    explained_variance = None
                
                results[name] = {
                    'reducer': reducer,
                    'transformed_data': X_reduced,
                    'explained_variance': explained_variance,
                    'original_shape': X.shape,
                    'reduced_shape': X_reduced.shape
                }
                
                print(f"  Reduced from {X.shape[1]} to {X_reduced.shape[1]} dimensions")
                if explained_variance is not None:
                    print(f"  Explained variance: {explained_variance:.4f}")
                    
            except Exception as e:
                print(f"  Error with {name}: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def visualize_reductions(self, X, y=None):
        """Visualize all dimensionality reduction results."""
        
        n_methods = len(self.results)
        cols = 3
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, result) in enumerate(self.results.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            X_reduced = result['transformed_data']
            
            if y is not None:
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab10', alpha=0.7)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)
            
            ax.set_title(f'{name.upper()}')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_methods, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def compare_methods(self):
        """Compare different dimensionality reduction methods."""
        
        comparison = []
        
        for name, result in self.results.items():
            comparison.append({
                'method': name,
                'original_dimensions': result['original_shape'][1],
                'reduced_dimensions': result['reduced_shape'][1],
                'explained_variance': result['explained_variance'] if result['explained_variance'] else 'N/A',
                'reduction_ratio': result['original_shape'][1] / result['reduced_shape'][1]
            })
        
        comparison_df = pd.DataFrame(comparison)
        
        print("Dimensionality Reduction Comparison:")
        print(comparison_df)
        
        return comparison_df

# Demonstrate dimensionality reduction
# Use a high-dimensional dataset
from sklearn.datasets import load_digits

digits = load_digits()
X_digits, y_digits = digits.data, digits.target

print(f"Original digits dataset shape: {X_digits.shape}")

# Initialize dimensionality reduction
dim_reduction = DimensionalityReduction()

# Detailed PCA analysis
pca_analysis, n_components_95 = dim_reduction.pca_analysis(X_digits)

# Setup reducers
reducers = dim_reduction.setup_reducers(n_components=2)

# Apply all reduction techniques
results = dim_reduction.apply_all_reductions(X_digits)

# Visualize results
dim_reduction.visualize_reductions(X_digits, y_digits)

# Compare methods
comparison = dim_reduction.compare_methods()
```

---

## Model Evaluation and Validation

### 1. Cross-Validation and Model Selection

```python
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, learning_curve
from sklearn.model_selection import validation_curve, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

class ModelValidation:
    """Comprehensive model validation and evaluation techniques."""
    
    def __init__(self):
        self.cv_results = {}
        self.learning_curves = {}
    
    def cross_validation_comparison(self, models, X, y, cv=5):
        """Compare multiple models using cross-validation."""
        
        cv_results = {}
        
        for name, model in models.items():
            print(f"\nCross-validating {name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            # Detailed cross-validation with multiple metrics
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            cv_detailed = cross_validate(model, X, y, cv=cv, scoring=scoring)
            
            cv_results[name] = {
                'accuracy_scores': cv_scores,
                'accuracy_mean': cv_scores.mean(),
                'accuracy_std': cv_scores.std(),
                'detailed_scores': cv_detailed
            }
            
            print(f"  Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.cv_results = cv_results
        return cv_results
    
    def learning_curve_analysis(self, model, X, y, cv=5):
        """Analyze learning curves to detect overfitting/underfitting."""
        
        # Calculate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.2, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.2, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Analyze overfitting
        final_gap = train_mean[-1] - val_mean[-1]
        if final_gap > 0.1:
            diagnosis = "Potential overfitting detected"
        elif val_mean[-1] < 0.7:
            diagnosis = "Potential underfitting detected"
        else:
            diagnosis = "Model appears well-fitted"
        
        print(f"Learning curve diagnosis: {diagnosis}")
        print(f"Final training score: {train_mean[-1]:.4f}")
        print(f"Final validation score: {val_mean[-1]:.4f}")
        print(f"Training-validation gap: {final_gap:.4f}")
        
        return {
            'train_sizes': train_sizes,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'diagnosis': diagnosis
        }
    
    def validation_curve_analysis(self, model, X, y, param_name, param_range, cv=5):
        """Analyze validation curves for hyperparameter tuning."""
        
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot validation curves
        plt.figure(figsize=(10, 6))
        plt.semilogx(param_range, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                        alpha=0.2, color='blue')
        
        plt.semilogx(param_range, val_mean, 'o-', color='red', label='Validation score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                        alpha=0.2, color='red')
        
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(f'Validation Curve for {param_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Find optimal parameter
        optimal_idx = np.argmax(val_mean)
        optimal_param = param_range[optimal_idx]
        
        print(f"Optimal {param_name}: {optimal_param}")
        print(f"Best validation score: {val_mean[optimal_idx]:.4f}")
        
        return optimal_param, val_mean[optimal_idx]
    
    def advanced_hyperparameter_tuning(self, model, param_grid, X, y, cv=5):
        """Advanced hyperparameter tuning with GridSearch and RandomizedSearch."""
        
        print("Performing Grid Search...")
        
        # Grid Search
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy',
            n_jobs=-1, verbose=1, return_train_score=True
        )
        grid_search.fit(X, y)
        
        print("Performing Randomized Search...")
        
        # Randomized Search
        random_search = RandomizedSearchCV(
            model, param_grid, n_iter=50, cv=cv, scoring='accuracy',
            n_jobs=-1, verbose=1, random_state=42, return_train_score=True
        )
        random_search.fit(X, y)
        
        # Compare results
        comparison = {
            'grid_search': {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'n_fits': len(grid_search.cv_results_['params'])
            },
            'random_search': {
                'best_score': random_search.best_score_,
                'best_params': random_search.best_params_,
                'n_fits': len(random_search.cv_results_['params'])
            }
        }
        
        print("\nHyperparameter Tuning Comparison:")
        for method, results in comparison.items():
            print(f"{method}:")
            print(f"  Best score: {results['best_score']:.4f}")
            print(f"  Best params: {results['best_params']}")
            print(f"  Number of fits: {results['n_fits']}")
        
        return grid_search, random_search, comparison
    
    def model_selection_report(self):
        """Generate comprehensive model selection report."""
        
        if not self.cv_results:
            print("No cross-validation results available")
            return
        
        print("Model Selection Report")
        print("=" * 50)
        
        # Sort models by performance
        sorted_models = sorted(self.cv_results.items(), 
                             key=lambda x: x[1]['accuracy_mean'], reverse=True)
        
        for rank, (name, results) in enumerate(sorted_models, 1):
            print(f"\nRank {rank}: {name.upper()}")
            print(f"  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
            
            # Detailed metrics if available
            if 'detailed_scores' in results:
                detailed = results['detailed_scores']
                for metric in ['test_precision_macro', 'test_recall_macro', 'test_f1_macro']:
                    if metric in detailed:
                        scores = detailed[metric]
                        print(f"  {metric.replace('test_', '').replace('_macro', '')}: "
                             f"{scores.mean():.4f} ± {scores.std():.4f}")
        
        return sorted_models

# Demonstrate model validation
from sklearn.datasets import load_breast_cancer

# Load dataset
cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target

# Prepare models for comparison
models_to_compare = {
    'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'svm': SVC(random_state=42),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Initialize validation
validator = ModelValidation()

# Cross-validation comparison
cv_results = validator.cross_validation_comparison(models_to_compare, X_cancer, y_cancer, cv=5)

# Learning curve analysis for best model
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
learning_results = validator.learning_curve_analysis(best_model, X_cancer, y_cancer)

# Validation curve analysis
param_range = [10, 50, 100, 200, 500]
optimal_estimators, best_score = validator.validation_curve_analysis(
    RandomForestClassifier(random_state=42), X_cancer, y_cancer,
    'n_estimators', param_range
)

# Advanced hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search, random_search, tuning_comparison = validator.advanced_hyperparameter_tuning(
    RandomForestClassifier(random_state=42), param_grid, X_cancer, y_cancer
)

# Generate model selection report
model_ranking = validator.model_selection_report()
```

---

## Pipelines and Automation

### 1. Creating ML Pipelines

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

class MLPipelines:
    """Comprehensive ML pipeline examples."""
    
    def __init__(self):
        self.pipelines = {}
    
    def simple_pipeline_example(self):
        """Create a simple preprocessing + model pipeline."""
        
        # Simple pipeline: Scale -> Train
        simple_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        print("Simple Pipeline Steps:")
        for step_name, step_transformer in simple_pipeline.steps:
            print(f"  {step_name}: {type(step_transformer).__name__}")
        
        return simple_pipeline
    
    def comprehensive_pipeline(self):
        """Create a comprehensive pipeline handling mixed data types."""
        
        # Define preprocessing for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Define preprocessing for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, ['numerical_feature_names']),
                ('cat', categorical_transformer, ['categorical_feature_names'])
            ]
        )
        
        # Create full pipeline
        comprehensive_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        return comprehensive_pipeline
    
    def real_world_pipeline_example(self):
        """Create a real-world pipeline with a mixed dataset."""
        
        # Create sample mixed dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Numerical features
        age = np.random.normal(35, 10, n_samples)
        income = np.random.normal(50000, 15000, n_samples)
        
        # Categorical features
        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
        city = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_samples)
        
        # Create target
        target = (age > 30) & (income > 45000)
        
        # Create DataFrame
        df = pd.DataFrame({
            'age': age,
            'income': income,
            'education': education,
            'city': city,
            'target': target
        })
        
        # Add some missing values
        df.loc[np.random.choice(df.index, 50), 'age'] = np.nan
        df.loc[np.random.choice(df.index, 30), 'education'] = np.nan
        
        print("Sample mixed dataset created:")
        print(df.head())
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Define feature columns
        numerical_features = ['age', 'income']
        categorical_features = ['education', 'city']
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Create full pipeline
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Split data
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train pipeline
        print("\nTraining pipeline...")
        full_pipeline.fit(X_train, y_train)
        
        # Evaluate
        train_score = full_pipeline.score(X_train, y_train)
        test_score = full_pipeline.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        return full_pipeline, X_test, y_test
    
    def pipeline_with_feature_selection(self):
        """Create pipeline with feature selection."""
        from sklearn.feature_selection import SelectKBest, f_classif
        
        feature_selection_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        return feature_selection_pipeline
    
    def pipeline_hyperparameter_tuning(self, pipeline, X, y):
        """Hyperparameter tuning for pipelines."""
        
        # Define parameter grid for pipeline
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, None],
            'preprocessor__num__imputer__strategy': ['mean', 'median']
        }
        
        # Grid search on pipeline
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        print("Performing pipeline hyperparameter tuning...")
        grid_search.fit(X, y)
        
        print(f"Best pipeline score: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def save_and_load_pipeline(self, pipeline, filename='ml_pipeline.pkl'):
        """Save and load trained pipeline."""
        import joblib
        
        # Save pipeline
        joblib.dump(pipeline, filename)
        print(f"Pipeline saved to {filename}")
        
        # Load pipeline
        loaded_pipeline = joblib.load(filename)
        print(f"Pipeline loaded from {filename}")
        
        return loaded_pipeline

# Demonstrate ML pipelines
pipeline_demo = MLPipelines()

# Simple pipeline
simple_pipe = pipeline_demo.simple_pipeline_example()

# Real-world pipeline with mixed data
full_pipe, X_test_mixed, y_test_mixed = pipeline_demo.real_world_pipeline_example()

# Pipeline with feature selection
feature_selection_pipe = pipeline_demo.pipeline_with_feature_selection()

# Hyperparameter tuning on pipeline
# best_pipeline = pipeline_demo.pipeline_hyperparameter_tuning(full_pipe, X_test_mixed, y_test_mixed)

# Save and load pipeline
saved_pipeline = pipeline_demo.save_and_load_pipeline(full_pipe)
```

---

## Advanced Topics

### 1. Handling Imbalanced Datasets

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

class ImbalancedDataHandling:
    """Techniques for handling imbalanced datasets."""
    
    def __init__(self):
        pass
    
    def create_imbalanced_dataset(self):
        """Create a sample imbalanced dataset."""
        
        X, y = make_classification(
            n_samples=10000,
            n_features=20,
            n_informative=10,
            n_redundant=10,
            n_clusters_per_class=1,
            weights=[0.95, 0.05],  # 95% class 0, 5% class 1
            random_state=42
        )
        
        unique, counts = np.unique(y, return_counts=True)
        print("Class distribution:")
        for class_label, count in zip(unique, counts):
            print(f"  Class {class_label}: {count} ({count/len(y)*100:.1f}%)")
        
        return X, y
    
    def baseline_performance(self, X, y):
        """Establish baseline performance on imbalanced data."""
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42, stratify=y)
        
        # Train baseline model
        baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        baseline_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = baseline_model.predict(X_test)
        y_pred_proba = baseline_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("Baseline Performance (No Balancing):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return baseline_model, X_train, X_test, y_train, y_test
    
    def class_weight_balancing(self, X_train, X_test, y_train, y_test):
        """Use class weights to handle imbalance."""
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        print(f"Calculated class weights: {class_weight_dict}")
        
        # Train model with class weights
        weighted_model = RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced',
            random_state=42
        )
        weighted_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_weighted = weighted_model.predict(X_test)
        y_pred_proba_weighted = weighted_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
        roc_auc_weighted = roc_auc_score(y_test, y_pred_proba_weighted)
        
        print("\nClass Weight Balancing Performance:")
        print(f"Accuracy: {accuracy_weighted:.4f}")
        print(f"ROC AUC: {roc_auc_weighted:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_weighted))
        
        return weighted_model
    
    def sampling_techniques(self, X, y):
        """Apply different sampling techniques."""
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42, stratify=y)
        
        sampling_results = {}
        
        # Original distribution
        print("Original training distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for class_label, count in zip(unique, counts):
            print(f"  Class {class_label}: {count}")
        
        # 1. Random Oversampling
        ros = RandomOverSampler(random_state=42)
        X_ros, y_ros = ros.fit_resample(X_train, y_train)
        
        print("\nAfter Random Oversampling:")
        unique, counts = np.unique(y_ros, return_counts=True)
        for class_label, count in zip(unique, counts):
            print(f"  Class {class_label}: {count}")
        
        # 2. SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        
        print("\nAfter SMOTE:")
        unique, counts = np.unique(y_smote, return_counts=True)
        for class_label, count in zip(unique, counts):
            print(f"  Class {class_label}: {count}")
        
        # 3. Random Undersampling
        rus = RandomUnderSampler(random_state=42)
        X_rus, y_rus = rus.fit_resample(X_train, y_train)
        
        print("\nAfter Random Undersampling:")
        unique, counts = np.unique(y_rus, return_counts=True)
        for class_label, count in zip(unique, counts):
            print(f"  Class {class_label}: {count}")
        
        # 4. SMOTE + Tomek
        smote_tomek = SMOTETomek(random_state=42)
        X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X_train, y_train)
        
        print("\nAfter SMOTE + Tomek:")
        unique, counts = np.unique(y_smote_tomek, return_counts=True)
        for class_label, count in zip(unique, counts):
            print(f"  Class {class_label}: {count}")
        
        # Train models on each resampled dataset
        sampling_techniques = {
            'random_oversample': (X_ros, y_ros),
            'smote': (X_smote, y_smote),
            'random_undersample': (X_rus, y_rus),
            'smote_tomek': (X_smote_tomek, y_smote_tomek)
        }
        
        for technique, (X_resampled, y_resampled) in sampling_techniques.items():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_resampled, y_resampled)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            sampling_results[technique] = {
                'model': model,
                'accuracy': accuracy,
                'roc_auc': roc_auc
            }
        
        return sampling_results
    
    def evaluate_threshold_tuning(self, model, X_test, y_test):
        """Tune classification threshold for better performance."""
        
        # Get prediction probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.1)
        threshold_results = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred_threshold)
            precision = precision_score(y_test, y_pred_threshold)
            recall = recall_score(y_test, y_pred_threshold)
            f1 = f1_score(y_test, y_pred_threshold)
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        threshold_df = pd.DataFrame(threshold_results)
        
        print("Threshold Tuning Results:")
        print(threshold_df)
        
        # Find best threshold by F1 score
        best_threshold_idx = threshold_df['f1'].idxmax()
        best_threshold = threshold_df.loc[best_threshold_idx, 'threshold']
        
        print(f"\nBest threshold (by F1 score): {best_threshold}")
        
        return threshold_df, best_threshold

# Demonstrate imbalanced data handling
imbalanced_handler = ImbalancedDataHandling()

# Create imbalanced dataset
X_imb, y_imb = imbalanced_handler.create_imbalanced_dataset()

# Baseline performance
baseline_model, X_train_imb, X_test_imb, y_train_imb, y_test_imb = imbalanced_handler.baseline_performance(X_imb, y_imb)

# Class weight balancing
weighted_model = imbalanced_handler.class_weight_balancing(X_train_imb, X_test_imb, y_train_imb, y_test_imb)

# Sampling techniques
sampling_results = imbalanced_handler.sampling_techniques(X_imb, y_imb)

print("\nSampling Techniques Comparison:")
for technique, results in sampling_results.items():
    print(f"{technique}: Accuracy={results['accuracy']:.4f}, ROC AUC={results['roc_auc']:.4f}")

# Threshold tuning
threshold_results, best_threshold = imbalanced_handler.evaluate_threshold_tuning(baseline_model, X_test_imb, y_test_imb)
```

### 2. Ensemble Methods and Advanced Techniques

```python
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

class EnsembleMethods:
    """Advanced ensemble methods and techniques."""
    
    def __init__(self):
        pass
    
    def voting_classifier_example(self, X_train, X_test, y_train, y_test):
        """Demonstrate voting classifiers (hard and soft voting)."""
        
        # Define individual classifiers
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        svm_clf = SVC(probability=True, random_state=42)
        nb_clf = GaussianNB()
        
        # Hard voting classifier
        hard_voting_clf = VotingClassifier(
            estimators=[('rf', rf_clf), ('svm', svm_clf), ('nb', nb_clf)],
            voting='hard'
        )
        
        # Soft voting classifier
        soft_voting_clf = VotingClassifier(
            estimators=[('rf', rf_clf), ('svm', svm_clf), ('nb', nb_clf)],
            voting='soft'
        )
        
        # Train all classifiers
        classifiers = {
            'Random Forest': rf_clf,
            'SVM': svm_clf,
            'Naive Bayes': nb_clf,
            'Hard Voting': hard_voting_clf,
            'Soft Voting': soft_voting_clf
        }
        
        results = {}
        
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            results[name] = accuracy
            print(f"{name}: {accuracy:.4f}")
        
        return results, soft_voting_clf
    
    def bagging_example(self, X_train, X_test, y_train, y_test):
        """Demonstrate bagging ensemble."""
        
        # Bagging with decision trees
        bagging_clf = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        # Extra Trees (Extremely Randomized Trees)
        extra_trees_clf = ExtraTreesClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        # Train and evaluate
        bagging_clf.fit(X_train, y_train)
        extra_trees_clf.fit(X_train, y_train)
        
        bagging_accuracy = bagging_clf.score(X_test, y_test)
        extra_trees_accuracy = extra_trees_clf.score(X_test, y_test)
        
        print(f"Bagging Classifier: {bagging_accuracy:.4f}")
        print(f"Extra Trees: {extra_trees_accuracy:.4f}")
        
        return bagging_clf, extra_trees_clf
    
    def boosting_example(self, X_train, X_test, y_train, y_test):
        """Demonstrate boosting ensemble."""
        
        # AdaBoost
        ada_boost_clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        )
        
        # Gradient Boosting
        gb_clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # Train and evaluate
        ada_boost_clf.fit(X_train, y_train)
        gb_clf.fit(X_train, y_train)
        
        ada_accuracy = ada_boost_clf.score(X_test, y_test)
        gb_accuracy = gb_clf.score(X_test, y_test)
        
        print(f"AdaBoost: {ada_accuracy:.4f}")
        print(f"Gradient Boosting: {gb_accuracy:.4f}")
        
        return ada_boost_clf, gb_clf
    
    def stacking_example(self, X_train, X_test, y_train, y_test):
        """Demonstrate stacking ensemble."""
        
        # Base classifiers
        base_classifiers = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('nb', GaussianNB())
        ]
        
        # Meta classifier
        meta_classifier = LogisticRegression(random_state=42)
        
        # Stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=meta_classifier,
            cv=5,
            n_jobs=-1
        )
        
        # Train and evaluate
        stacking_clf.fit(X_train, y_train)
        stacking_accuracy = stacking_clf.score(X_test, y_test)
        
        print(f"Stacking Classifier: {stacking_accuracy:.4f}")
        
        # Compare with individual base classifiers
        print("\nBase Classifier Performance:")
        for name, clf in base_classifiers:
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            print(f"  {name}: {accuracy:.4f}")
        
        return stacking_clf
    
    def ensemble_feature_importance(self, ensemble_model, feature_names=None):
        """Analyze feature importance in ensemble models."""
        
        if hasattr(ensemble_model, 'feature_importances_'):
            importance = ensemble_model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("Feature Importance:")
            print(feature_importance_df.head(10))
            
            return feature_importance_df
        else:
            print("Model does not support feature importance")
            return None

# Demonstrate ensemble methods
ensemble_demo = EnsembleMethods()

# Use wine dataset for demonstration
wine = load_wine()
X_wine, y_wine = wine.data, wine.target

# Convert to binary classification for simplicity
y_wine_binary = (y_wine == 0).astype(int)

X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine_binary, test_size=0.2, random_state=42, stratify=y_wine_binary
)

# Scale features
scaler = StandardScaler()
X_train_wine_scaled = scaler.fit_transform(X_train_wine)
X_test_wine_scaled = scaler.transform(X_test_wine)

print("Ensemble Methods Demonstration")
print("=" * 40)

# Voting classifiers
print("\n1. Voting Classifiers:")
voting_results, best_voting_clf = ensemble_demo.voting_classifier_example(
    X_train_wine_scaled, X_test_wine_scaled, y_train_wine, y_test_wine
)

# Bagging
print("\n2. Bagging Methods:")
bagging_clf, extra_trees_clf = ensemble_demo.bagging_example(
    X_train_wine_scaled, X_test_wine_scaled, y_train_wine, y_test_wine
)

# Boosting
print("\n3. Boosting Methods:")
ada_clf, gb_clf = ensemble_demo.boosting_example(
    X_train_wine_scaled, X_test_wine_scaled, y_train_wine, y_test_wine
)

# Stacking
print("\n4. Stacking:")
stacking_clf = ensemble_demo.stacking_example(
    X_train_wine_scaled, X_test_wine_scaled, y_train_wine, y_test_wine
)

# Feature importance analysis
print("\n5. Feature Importance Analysis:")
feature_importance = ensemble_demo.ensemble_feature_importance(gb_clf, wine.feature_names)
```

---

## Integration with Your Internship Program

### Week-by-Week Scikit-learn Integration

#### **Week 1-2: Foundation Building**
```python
# Week 1-2 Focus: Data loading, preprocessing, basic visualization
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Practice exercises:
# 1. Load built-in datasets
# 2. Basic data exploration
# 3. Train-test splitting
# 4. Feature scaling
```

#### **Week 3: Supervised Learning Basics**
```python
# Week 3 Focus: Classification and regression fundamentals
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# Practice exercises:
# 1. Implement basic classifiers
# 2. Compare algorithm performance
# 3. Understand bias-variance tradeoff
```

#### **Week 4: Advanced Supervised Learning**
```python
# Week 4 Focus: Ensemble methods, hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Practice exercises:
# 1. Ensemble methods comparison
# 2. Hyperparameter optimization
# 3. Cross-validation strategies
```

#### **Week 5: Unsupervised Learning**
```python
# Week 5 Focus: Clustering, dimensionality reduction
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Practice exercises:
# 1. Customer segmentation
# 2. Data visualization with PCA
# 3. Anomaly detection
```

#### **Week 6-7: Production ML**
```python
# Week 6-7 Focus: Pipelines, model deployment, advanced techniques
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Practice exercises:
# 1. End-to-end ML pipelines
# 2. Model serialization
# 3. Handling real-world data issues
```

#### **Week 8: Capstone Integration**
```python
# Week 8 Focus: Combine sklearn with LangChain and other tools
# 1. Traditional ML + LLM explanations
# 2. Model interpretation and business insights
# 3. Complete ML solution deployment
```

### Project Ideas by Complexity

#### **Beginner Projects (Weeks 1-3)**
1. **Iris Classification**: Complete classification pipeline
2. **Boston Housing Prediction**: Regression with feature engineering
3. **Wine Quality Analysis**: Multi-class classification

#### **Intermediate Projects (Weeks 4-6)**
1. **Customer Segmentation**: Clustering + business insights
2. **Credit Risk Assessment**: Imbalanced classification
3. **Sales Forecasting**: Time series regression

#### **Advanced Projects (Weeks 7-8)**
1. **Fraud Detection System**: Complete ML pipeline with monitoring
2. **Recommendation Engine**: Collaborative filtering + content-based
3. **Predictive Maintenance**: Multi-modal data analysis

---

## Best Practices and Tips

### 1. Code Organization

```python
# Recommended project structure for sklearn projects
"""
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
├── models/
│   ├── trained_models/
│   └── model_configs/
├── requirements.txt
└── README.md
"""

# Example utils.py for sklearn projects
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def save_model(model, filepath):
    """Save trained sklearn model."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load trained sklearn model."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def evaluate_classifier(model, X_test, y_test):
    """Comprehensive classifier evaluation."""
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return y_pred
```

### 2. Common Pitfalls and Solutions

```python
# Common Pitfall 1: Data leakage
# ❌ Wrong way
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on entire dataset
X_train, X_test = train_test_split(X_scaled, y)

# ✅ Correct way
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
X_test_scaled = scaler.transform(X_test)  # Transform test data

# Common Pitfall 2: Inconsistent preprocessing
# ❌ Wrong way
def preprocess_data(X):
    # Different preprocessing each time
    return StandardScaler().fit_transform(X)

# ✅ Correct way
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted yet")
        return self.scaler.transform(X)

# Common Pitfall 3: Ignoring class imbalance
# ❌ Wrong way
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)  # Only looking at accuracy

# ✅ Correct way
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Use multiple metrics
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
```

### 3. Performance Optimization

```python
# Tip 1: Use n_jobs=-1 for parallel processing
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # Use all CPU cores

# Tip 2: Use appropriate data types
df = df.astype({
    'category_col': 'category',  # For categorical data
    'int_col': 'int32',          # Smaller int types when possible
    'float_col': 'float32'       # Smaller float types when appropriate
})

# Tip 3: Use sparse matrices for high-dimensional data
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

# For text data
vectorizer = TfidfVectorizer(max_features=10000)
X_sparse = vectorizer.fit_transform(text_data)  # Returns sparse matrix

# Tip 4: Use pipeline for consistent preprocessing
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_jobs=-1))
])

# Tip 5: Use partial_fit for large datasets
from sklearn.linear_model import SGDClassifier

# For datasets too large for memory
model = SGDClassifier()
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    X_chunk, y_chunk = preprocess_chunk(chunk)
    model.partial_fit(X_chunk, y_chunk, classes=np.unique(y))
```

---

## Additional Resources

### Official Documentation
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [API Reference](https://scikit-learn.org/stable/modules/classes.html)
- [Examples Gallery](https://scikit-learn.org/stable/auto_examples/index.html)

### Learning Resources
- [Scikit-learn Course](https://inria.github.io/scikit-learn-mooc/) - Free online course by Inria
- [Hands-On Machine Learning](https://github.com/ageron/handson-ml3) - Companion notebooks
- [Scikit-learn Tutorials](https://github.com/justmarkham/scikit-learn-videos) - Video tutorials

### Community and Support
- [Stack Overflow - scikit-learn tag](https://stackoverflow.com/questions/tagged/scikit-learn)
- [GitHub Issues](https://github.com/scikit-learn/scikit-learn/issues)
- [Discord Community](https://discord.gg/h9qyrK8Jf9)

### Advanced Topics
- [Scikit-learn Enhancement Proposals (SLEP)](https://scikit-learn-enhancement-proposals.readthedocs.io/)
- [Contributing to Scikit-learn](https://scikit-learn.org/dev/developers/contributing.html)
- [Custom Estimators](https://scikit-learn.org/stable/developers/develop.html)

---

## Quick Reference Cheat Sheet

```python
# Essential imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Basic workflow
# 1. Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 5. Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

Scikit-learn is the foundation of machine learning in Python. Master these concepts and you'll be well-prepared for both traditional ML projects and as a complement to modern generative AI applications in your internship program!
            