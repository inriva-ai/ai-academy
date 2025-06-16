# Week 2 Resources: LangChain & Advanced Preprocessing

Quick reference materials for Week 2 concepts and tools.

---

## 🦜 LangChain Quick Reference

### Core Concepts

#### LCEL (LangChain Expression Language)
```python
# Basic chain composition
chain = prompt | model | output_parser

# Equivalent to:
def chain(input_dict):
    prompt_value = prompt.format(**input_dict)
    model_output = model.invoke(prompt_value)
    return output_parser.parse(model_output)
```

#### Chain Building Patterns
```python
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# 1. Simple chain
prompt = PromptTemplate.from_template("Analyze: {text}")
model = Ollama(model="llama3.2")
parser = StrOutputParser()
chain = prompt | model | parser

# 2. Chain with input transformation
from langchain_core.runnables import RunnableLambda

def preprocess(input_dict):
    return {"text": input_dict["raw_text"].lower()}

chain = RunnableLambda(preprocess) | prompt | model | parser

# 3. Parallel chains
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel({
    "summary": prompt | model | parser,
    "sentiment": sentiment_prompt | model | parser
})
```

### Custom Output Parsers
```python
from langchain_core.output_parsers import BaseOutputParser

class StructuredParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        # Extract structured data from LLM output
        lines = text.strip().split('\n')
        return {
            'summary': lines[0] if lines else '',
            'insights': [line.strip('- ') for line in lines[1:] if line.strip()]
        }

# Usage
parser = StructuredParser()
chain = prompt | model | parser
```

### Error Handling and Fallbacks
```python
from langchain_core.runnables import RunnableLambda

# Fallback chain
primary_chain = prompt | model | parser
fallback_chain = prompt | backup_model | parser

chain_with_fallback = primary_chain.with_fallbacks([fallback_chain])

# Custom error handling
def safe_invoke(input_data):
    try:
        return chain.invoke(input_data)
    except Exception as e:
        return {"error": str(e), "fallback": "Statistical analysis used"}

safe_chain = RunnableLambda(safe_invoke)
```

### Batch Processing
```python
# Process multiple inputs efficiently
inputs = [{"text": f"Analysis {i}"} for i in range(10)]

# Sequential processing
results = [chain.invoke(input_data) for input_data in inputs]

# Batch processing (more efficient)
results = chain.batch(inputs)

# Streaming results
for chunk in chain.stream({"text": "Long analysis..."}):
    print(chunk, end="")
```

---

## 🔧 Advanced Data Preprocessing Patterns

### Missing Value Strategies

#### Group-Based Imputation
```python
# Age imputation by demographics
age_median = df.groupby(['Sex', 'Pclass'])['Age'].transform('median')
df['Age'].fillna(age_median, inplace=True)

# Multiple group levels
income_median = df.groupby(['Education', 'Location'])['Income'].transform('median')
df['Income'].fillna(income_median, inplace=True)
```

#### KNN Imputation
```python
from sklearn.impute import KNNImputer

# For numerical features
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
```

#### Advanced Missing Pattern Analysis
```python
import missingno as msno

# Visualize missing patterns
msno.matrix(df)
msno.heatmap(df)  # Correlation of missingness

# Missing pattern analysis
def analyze_missing_patterns(df):
    missing_patterns = {}
    for col in df.columns:
        if df[col].isnull().any():
            # Find what other columns are missing when this one is missing
            missing_mask = df[col].isnull()
            pattern = df[missing_mask].isnull().sum()
            missing_patterns[col] = pattern.to_dict()
    return missing_patterns
```

### Feature Engineering Patterns

#### Interaction Features
```python
# Numerical interactions
df['Age_Income_Ratio'] = df['Age'] / (df['Income'] / 1000)
df['Income_per_Family'] = df['Income'] / df['Family_Size']

# Categorical interactions
df['Location_Education'] = df['Location'] + '_' + df['Education']

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['Age', 'Income']])
```

#### Text Feature Engineering
```python
# Title extraction
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')

# Text statistics
df['Name_Length'] = df['Name'].str.len()
df['Name_Word_Count'] = df['Name'].str.split().str.len()

# Advanced text features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
text_features = tfidf.fit_transform(df['Text_Column'])
```

#### Binning and Discretization
```python
# Equal-width binning
df['Age_Group'] = pd.cut(df['Age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Mature', 'Senior'])

# Quantile-based binning
df['Income_Quartile'] = pd.qcut(df['Income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

# Custom binning
def custom_age_bins(age):
    if age < 18: return 'Minor'
    elif age < 65: return 'Adult'
    else: return 'Senior'

df['Age_Category'] = df['Age'].apply(custom_age_bins)
```

### Outlier Detection and Handling

#### Multiple Methods
```python
# IQR Method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

# Isolation Forest
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.1)
outliers = clf.fit_predict(df[numeric_columns])

# Z-Score Method
from scipy import stats
z_scores = np.abs(stats.zscore(df[numeric_columns]))
outliers = (z_scores > 3).any(axis=1)
```

#### Outlier Treatment
```python
# Capping (Winsorizing)
def cap_outliers(df, column, lower_percentile=5, upper_percentile=95):
    lower_cap = df[column].quantile(lower_percentile / 100)
    upper_cap = df[column].quantile(upper_percentile / 100)
    df[column] = np.clip(df[column], lower_cap, upper_cap)
    return df

# Log transformation for skewed data
df['Income_Log'] = np.log1p(df['Income'])  # log(1+x) to handle zeros

# Square root transformation
df['Age_Sqrt'] = np.sqrt(df['Age'])
```

---

## 🌊 Metaflow Advanced Patterns

### Error Handling
```python
from metaflow import catch

@catch(var='processing_errors')
@step
def robust_step(self):
    try:
        # Processing logic
        self.result = process_data(self.input_data)
    except Exception as e:
        self.processing_errors = str(e)
        # Fallback logic
        self.result = fallback_processing(self.input_data)
```

### Parallel Processing
```python
@parallel
@step
def parallel_models(self):
    # This step will run in parallel for each model
    model_name = self.input
    self.model_result = train_model(model_name, self.training_data)

@step
def combine_results(self, inputs):
    # Combine results from parallel steps
    self.all_results = {inp.input: inp.model_result for inp in inputs}
```

### Resource Management
```python
from metaflow import resources

@resources(memory=8000, cpu=4)
@step
def memory_intensive_step(self):
    # This step will request more resources
    self.large_dataset = process_large_data()
```

### Data Artifacts
```python
@step
def save_artifacts(self):
    # Save large datasets efficiently
    from metaflow import S3
    self.processed_data = S3(self.dataframe)
    
    # Save models
    import pickle
    self.model_artifact = pickle.dumps(self.trained_model)
```

---

## 🔧 Ollama Quick Reference

### Model Management
```bash
# List available models
ollama list

# Download models
ollama pull llama3.2
ollama pull mistral
ollama pull codellama

# Remove models
ollama rm model_name

# Show model information
ollama show llama3.2
```

### Model Configuration
```python
from langchain_community.llms import Ollama

# Basic configuration
llm = Ollama(model="llama3.2")

# Advanced configuration
llm = Ollama(
    model="llama3.2",
    temperature=0.3,        # Lower = more deterministic
    top_p=0.9,             # Nucleus sampling
    top_k=40,              # Top-k sampling
    num_predict=256,       # Max tokens to generate
    repeat_penalty=1.1,    # Penalty for repetition
    stop=["\n\n", "###"]   # Stop sequences
)

# Timeout configuration
import time
def ollama_with_timeout(model, prompt, timeout=30):
    start_time = time.time()
    try:
        response = model.invoke(prompt)
        if time.time() - start_time > timeout:
            raise TimeoutError("Model response took too long")
        return response
    except Exception as e:
        raise e
```

### Performance Optimization
```bash
# Set environment variables for better performance
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_FLASH_ATTENTION=1

# GPU acceleration (if available)
export OLLAMA_GPU_LAYERS=32
```

---

## 📊 Data Quality Assessment

### Comprehensive Quality Scoring
```python
def calculate_data_quality_score(df):
    score = 100.0
    issues = []
    
    # Missing values penalty
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct > 0:
        penalty = min(missing_pct * 2, 30)
        score -= penalty
        issues.append(f"Missing values: {missing_pct:.1f}%")
    
    # Duplicate rows penalty
    dup_pct = (df.duplicated().sum() / len(df)) * 100
    if dup_pct > 0:
        penalty = min(dup_pct * 3, 20)
        score -= penalty
        issues.append(f"Duplicate rows: {dup_pct:.1f}%")
    
    # Outlier penalty
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total_outliers = 0
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        total_outliers += outliers
    
    outlier_pct = (total_outliers / len(df)) * 100
    if outlier_pct > 5:
        penalty = min((outlier_pct - 5) * 2, 25)
        score -= penalty
        issues.append(f"Outliers: {outlier_pct:.1f}%")
    
    return max(0, score), issues
```

### Data Drift Detection
```python
def detect_data_drift(reference_df, current_df, threshold=0.05):
    """Simple data drift detection using statistical tests"""
    drift_results = {}
    
    for col in reference_df.columns:
        if col in current_df.columns:
            if reference_df[col].dtype in ['int64', 'float64']:
                # KS test for numerical features
                from scipy.stats import ks_2samp
                statistic, p_value = ks_2samp(reference_df[col].dropna(), 
                                            current_df[col].dropna())
                drift_results[col] = {
                    'test': 'ks_test',
                    'p_value': p_value,
                    'drift_detected': p_value < threshold
                }
            else:
                # Chi-square test for categorical features
                from scipy.stats import chi2_contingency
                ref_counts = reference_df[col].value_counts()
                curr_counts = current_df[col].value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                
                if sum(ref_aligned) > 0 and sum(curr_aligned) > 0:
                    statistic, p_value, _, _ = chi2_contingency([ref_aligned, curr_aligned])
                    drift_results[col] = {
                        'test': 'chi2_test',
                        'p_value': p_value,
                        'drift_detected': p_value < threshold
                    }
    
    return drift_results
```

---

## 🎯 Production Patterns

### Configuration Management
```python
import json
from pathlib import Path

class PipelineConfig:
    def __init__(self, config_path="config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return self.get_default_config()
    
    def get_default_config(self):
        return {
            "data_processing": {
                "missing_threshold": 0.3,
                "outlier_method": "iqr",
                "scaling_method": "standard"
            },
            "llm_config": {
                "model": "llama3.2",
                "temperature": 0.3,
                "timeout": 30,
                "max_retries": 3
            },
            "monitoring": {
                "quality_threshold": 80,
                "alert_email": "admin@company.com"
            }
        }
    
    def get(self, key_path, default=None):
        """Get nested config value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key, {})
        return value if value != {} else default

# Usage
config = PipelineConfig()
threshold = config.get('data_processing.missing_threshold', 0.3)
```

### Monitoring and Alerting
```python
import logging
from datetime import datetime

class PipelineMonitor:
    def __init__(self, log_level=logging.INFO):
        self.setup_logging(log_level)
        self.metrics = {}
        self.alerts = []
    
    def setup_logging(self, level):
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_metric(self, name, value, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        self.logger.info(f"Metric {name}: {value}")
    
    def check_threshold(self, metric_name, threshold, comparison='<'):
        if metric_name in self.metrics and self.metrics[metric_name]:
            latest_value = self.metrics[metric_name][-1]['value']
            
            alert_triggered = False
            if comparison == '<' and latest_value < threshold:
                alert_triggered = True
            elif comparison == '>' and latest_value > threshold:
                alert_triggered = True
            
            if alert_triggered:
                alert = {
                    'metric': metric_name,
                    'value': latest_value,
                    'threshold': threshold,
                    'timestamp': datetime.now()
                }
                self.alerts.append(alert)
                self.logger.warning(f"Alert: {metric_name} = {latest_value} {comparison} {threshold}")
                return True
        
        return False
    
    def get_dashboard_data(self):
        return {
            'metrics': self.metrics,
            'alerts': self.alerts,
            'recent_alerts': [a for a in self.alerts if 
                            (datetime.now() - a['timestamp']).seconds < 3600]
        }

# Usage
monitor = PipelineMonitor()
monitor.log_metric('data_quality_score', 85.5)
monitor.check_threshold('data_quality_score', 80, '>')
```

### Caching and Performance
```python
import hashlib
import pickle
from functools import wraps

class ResultCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def cache_key(self, func_name, args, kwargs):
        """Generate cache key from function name and arguments"""
        key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key):
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set(self, key, value):
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)

def cache_result(cache_instance):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache_instance.cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                print(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result)
            print(f"Cache miss for {func.__name__} - result cached")
            return result
        
        return wrapper
    return decorator

# Usage
cache = ResultCache()

@cache_result(cache)
def expensive_computation(data):
    # Simulate expensive operation
    time.sleep(2)
    return data.mean()
```

---

## 🚀 Deployment Checklist

### Pre-Production Checklist
- [ ] **Data Quality**: Score > 80/100
- [ ] **Error Handling**: All critical paths have try/catch
- [ ] **Fallback Systems**: Backup models/methods available
- [ ] **Performance**: Processing time < 5 minutes
- [ ] **Monitoring**: Key metrics tracked
- [ ] **Configuration**: Environment-specific configs
- [ ] **Logging**: Comprehensive logging implemented
- [ ] **Testing**: Unit and integration tests pass
- [ ] **Documentation**: Deployment guide complete

### Monitoring Metrics
- **Data Quality Score**: Overall data health
- **Processing Time**: Per-step and total pipeline time
- **Error Rate**: Percentage of failed operations
- **LLM Response Quality**: Automated quality scoring
- **Resource Utilization**: Memory and CPU usage
- **Cache Hit Rate**: Caching effectiveness

### Alert Thresholds
- **Critical**: Data quality < 60, errors > 10%, processing time > 10 min
- **Warning**: Data quality < 80, errors > 5%, processing time > 5 min
- **Info**: New data patterns detected, cache miss rate > 50%

---

## 🔗 Additional Resources

### Documentation Links
- [Metaflow Documentation](https://docs.metaflow.org/)
- [LangChain Python Docs](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.com/docs)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Community Resources
- [Metaflow Community Slack](https://join.slack.com/t/metaflow-community)
- [LangChain Discord](https://discord.gg/langchain)
- [Ollama GitHub](https://github.com/ollama/ollama)

### Troubleshooting
- **Ollama not responding**: Check `ollama list`, restart service
- **LangChain import errors**: Update with `pip install --upgrade langchain`
- **Memory issues**: Reduce batch size, use streaming
- **Slow processing**: Enable parallel processing, check resource allocation

---

*This reference guide covers the essential patterns and tools for Week 2. Bookmark for quick access during exercises!*