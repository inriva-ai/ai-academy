# Pipeline Optimization Exercises
## Metaflow ML Pipeline Performance & Scalability

Welcome to the Metaflow pipeline optimization exercises! These challenges will help you master advanced Metaflow patterns, performance optimization, and production-ready ML pipeline design.

## 🎯 Learning Objectives

By completing these exercises, you'll master:
- **Resource optimization** with @resources decorator
- **Parallel execution** patterns with @foreach
- **Error handling** and fault tolerance
- **Memory management** for large datasets
- **Pipeline scaling** strategies
- **Production deployment** patterns
- **Advanced Metaflow features** (@batch, @conda, @timeout)

---

## 📋 Prerequisites

- Completed Week 3 workshop
- Understanding of basic Metaflow patterns
- Familiarity with supervised learning concepts
- Python 3.7+ with Metaflow installed

---

## Exercise 1: Resource Optimization Fundamentals
**Difficulty: Beginner | Time: 30 minutes**

### Objective
Learn to optimize memory and CPU allocation for different ML algorithms to improve pipeline efficiency.

### Background
Different ML algorithms have varying computational requirements. Random Forest might need more memory for ensemble trees, while SVM could be more CPU-intensive. Proper resource allocation prevents out-of-memory errors and improves execution time.

### Your Task

1. **Create a resource-aware pipeline** that dynamically allocates resources based on algorithm type:

```python
from metaflow import FlowSpec, step, Parameter, foreach, resources
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

class ResourceOptimizedFlow(FlowSpec):
    
    dataset_size = Parameter('dataset_size', 
                           help='Number of samples (1000, 10000, 100000)', 
                           default=10000)
    
    @step
    def start(self):
        # Create dataset of varying sizes
        self.X, self.y = make_classification(
            n_samples=self.dataset_size,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )
        
        # Define algorithms with their resource requirements
        self.algorithms = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42),
                'memory': 2000,  # MB
                'cpu': 1
            },
            'random_forest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'memory': 8000,  # MB - needs more for tree storage
                'cpu': 4
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'memory': 6000,  # MB
                'cpu': 4
            },
            'svm': {
                'model': SVC(random_state=42),
                'memory': 4000,  # MB - kernel matrix
                'cpu': 2
            },
            'neural_network': {
                'model': MLPClassifier(hidden_layer_sizes=(100, 50), 
                                     max_iter=500, random_state=42),
                'memory': 6000,  # MB
                'cpu': 4
            }
        }
        
        self.algorithm_names = list(self.algorithms.keys())
        self.next(self.train_algorithm, foreach='algorithm_names')
    
    # TODO: Implement dynamic resource allocation step
    @step
    def train_algorithm(self):
        """
        YOUR TASK: 
        1. Get current algorithm and its resource requirements
        2. Apply @resources decorator dynamically (hint: you can't, so design around it)
        3. Train the model and measure performance
        4. Track memory usage and training time
        """
        pass  # Implement this
    
    @step  
    def evaluate_resource_efficiency(self, inputs):
        """
        YOUR TASK:
        1. Collect results from all algorithms
        2. Calculate resource efficiency metrics:
           - Performance per MB used
           - Performance per CPU hour
           - Total resource cost
        3. Identify most efficient algorithm for given dataset size
        """
        pass  # Implement this
    
    @step
    def end(self):
        """Generate resource optimization recommendations"""
        pass  # Implement this

if __name__ == '__main__':
    ResourceOptimizedFlow()
```

2. **Implement the missing steps** following these requirements:

   - **train_algorithm step**: Since @resources can't be applied dynamically, create separate steps for each algorithm type with appropriate resource allocations
   - **Memory tracking**: Monitor actual memory usage during training
   - **Performance metrics**: Track training time, accuracy, and resource consumption
   - **Resource efficiency**: Calculate performance per resource unit

3. **Run experiments** with different dataset sizes:
   ```bash
   python resource_flow.py run --dataset_size 1000
   python resource_flow.py run --dataset_size 10000
   python resource_flow.py run --dataset_size 100000
   ```

### Expected Outcomes
- Understand resource allocation patterns for different algorithms
- Learn to balance performance vs. resource consumption
- Master memory-efficient pipeline design

### Verification Criteria
- [ ] Pipeline runs successfully with all dataset sizes
- [ ] Resource usage is tracked and reported
- [ ] Performance per resource metrics are calculated
- [ ] Recommendations are generated for algorithm selection based on resource constraints

---

## Exercise 2: Advanced Parallel Execution Patterns
**Difficulty: Intermediate | Time: 45 minutes**

### Objective
Master complex parallel execution patterns including nested foreach, dynamic branching, and cross-validation parallelization.

### Background
Real-world ML pipelines often require multiple levels of parallelization: across algorithms, hyperparameters, cross-validation folds, and datasets. Efficient parallel design can reduce training time from hours to minutes.

### Your Task

1. **Implement a nested parallel hyperparameter tuning pipeline**:

```python
from metaflow import FlowSpec, step, Parameter, foreach, resources
from sklearn.model_selection import ParameterGrid

class AdvancedParallelFlow(FlowSpec):
    
    algorithms = Parameter('algorithms', 
                          help='Comma-separated algorithm names',
                          default='rf,gb,svm')
    
    cv_folds = Parameter('cv_folds',
                        help='Number of CV folds', 
                        default=5)
    
    @step
    def start(self):
        # TODO: Define algorithm configurations and parameter grids
        self.algorithm_configs = {
            'rf': {
                'model_class': RandomForestClassifier,
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'gb': {
                'model_class': GradientBoostingClassifier,
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'svm': {
                'model_class': SVC,
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
        
        # TODO: Create combinations for parallel execution
        self.next(self.parallel_algorithm_tuning, foreach='selected_algorithms')
    
    @resources(memory=6000, cpu=4)
    @step
    def parallel_algorithm_tuning(self):
        """
        YOUR TASK:
        1. Get current algorithm configuration
        2. Generate all parameter combinations
        3. Create nested foreach for parameter combinations
        4. Implement parallel hyperparameter evaluation
        """
        pass
    
    @step
    def parallel_cv_evaluation(self):
        """
        YOUR TASK:
        1. Implement parallel cross-validation for each parameter set
        2. Use self.cv_folds for dynamic CV fold creation
        3. Track performance across folds
        """
        pass
    
    @step
    def aggregate_cv_results(self, inputs):
        """Aggregate results from parallel CV folds"""
        pass
    
    @step
    def select_best_parameters(self, inputs):
        """Select best parameters for each algorithm"""
        pass
    
    @step
    def final_model_training(self):
        """Train final models with best parameters"""
        pass
    
    @step
    def compare_algorithms(self, inputs):
        """Compare optimized algorithms"""
        pass
    
    @step
    def end(self):
        """Generate final recommendations"""
        pass
```

2. **Implement parallel cross-validation** within hyperparameter tuning:
   - Each parameter combination should run CV in parallel
   - CV folds should be distributed across available cores
   - Results should be aggregated efficiently

3. **Add dynamic resource scaling**:
   - Adjust resources based on parameter combination complexity
   - Implement timeout handling for long-running combinations
   - Add progress tracking and logging

### Advanced Challenge: Nested Foreach Implementation

```python
@step
def create_hyperparam_combinations(self):
    """
    YOUR ADVANCED TASK:
    Create a truly nested parallel structure:
    
    Algorithms (foreach) 
    ├── Parameter combinations (foreach)
    │   ├── CV fold 1 (foreach)
    │   ├── CV fold 2 (foreach)
    │   └── CV fold N (foreach)
    
    Hint: You'll need multiple foreach steps and careful data passing
    """
    pass
```

### Expected Outcomes
- Master complex parallel execution patterns
- Understand resource management in nested parallelism
- Learn efficient data passing between parallel branches

### Verification Criteria
- [ ] Nested parallelism works correctly
- [ ] All parameter combinations are evaluated
- [ ] Cross-validation results are properly aggregated
- [ ] Resource utilization is optimized
- [ ] Pipeline completes faster than sequential execution

---

## Exercise 3: Production-Ready Error Handling & Fault Tolerance
**Difficulty: Intermediate | Time: 40 minutes**

### Objective
Build robust pipelines that gracefully handle failures, implement retry logic, and maintain data integrity.

### Your Task

1. **Implement comprehensive error handling**:

```python
from metaflow import FlowSpec, step, Parameter, foreach, resources, catch, timeout, retry
import random

class RobustMLFlow(FlowSpec):
    
    failure_rate = Parameter('failure_rate',
                           help='Simulated failure rate (0.0-1.0)',
                           default=0.2)
    
    @step
    def start(self):
        self.algorithms = ['unstable_rf', 'memory_hungry_svm', 'slow_gb', 'reliable_lr']
        self.next(self.train_with_failures, foreach='algorithms')
    
    @catch(var='training_error')
    @retry(times=3)
    @timeout(seconds=300)
    @resources(memory=4000, cpu=2)
    @step
    def train_with_failures(self):
        """
        YOUR TASK:
        1. Simulate different types of failures:
           - Random crashes (unstable_rf)
           - Memory errors (memory_hungry_svm) 
           - Timeout errors (slow_gb)
           - Successful training (reliable_lr)
        2. Implement proper error logging
        3. Create fallback strategies
        """
        
        algorithm = self.input
        
        # TODO: Implement failure simulation and handling
        if algorithm == 'unstable_rf':
            # 30% chance of random failure
            pass
        elif algorithm == 'memory_hungry_svm':
            # Simulate memory issues
            pass
        elif algorithm == 'slow_gb':
            # Simulate timeout scenarios
            pass
        else:
            # Reliable algorithm
            pass
    
    @step
    def handle_failures_and_continue(self, inputs):
        """
        YOUR TASK:
        1. Identify which algorithms failed
        2. Implement fallback strategies:
           - Use simpler algorithm parameters
           - Switch to alternative algorithms
           - Continue with successful results only
        3. Generate failure report
        """
        pass
    
    @step
    def quality_assurance_checks(self):
        """
        YOUR TASK:
        1. Implement data quality checks
        2. Validate model performance thresholds
        3. Check for data leakage
        4. Verify reproducibility
        """
        pass
    
    @step 
    def end(self):
        """Generate robustness report"""
        pass
```

2. **Implement specific error scenarios**:

   - **Memory exhaustion**: Simulate and handle OOM errors
   - **Network timeouts**: Handle external data source failures
   - **Data corruption**: Detect and handle corrupted input data
   - **Algorithm convergence failures**: Handle models that fail to converge

3. **Create intelligent retry logic**:
   ```python
   def intelligent_retry_strategy(self, error_type, attempt_number):
       """
       YOUR TASK: Implement adaptive retry logic
       - Different strategies for different error types
       - Exponential backoff for network errors
       - Resource adjustment for memory errors
       - Alternative algorithms for convergence failures
       """
       pass
   ```

### Advanced Challenge: Circuit Breaker Pattern

```python
class CircuitBreakerFlow(FlowSpec):
    """
    Implement a circuit breaker pattern for unreliable external services
    - Open circuit after N consecutive failures
    - Half-open circuit for testing recovery
    - Closed circuit for normal operation
    """
    
    @step
    def external_data_fetch_with_circuit_breaker(self):
        """
        YOUR ADVANCED TASK:
        Implement circuit breaker for external API calls
        """
        pass
```

### Expected Outcomes
- Build fault-tolerant ML pipelines
- Master Metaflow's error handling decorators
- Implement intelligent failure recovery strategies

### Verification Criteria
- [ ] Pipeline handles simulated failures gracefully
- [ ] Retry logic works appropriately for different error types
- [ ] Fallback strategies are implemented
- [ ] Failure reports are comprehensive
- [ ] Pipeline continues with partial results when possible

---

## Exercise 4: Memory-Efficient Large Dataset Processing
**Difficulty: Advanced | Time: 60 minutes**

### Objective
Design memory-efficient pipelines that can handle datasets larger than available RAM using streaming, chunking, and incremental learning techniques.

### Your Task

1. **Implement streaming data processing**:

```python
from metaflow import FlowSpec, step, Parameter, foreach, resources
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class LargeDatasetFlow(FlowSpec):
    
    total_samples = Parameter('total_samples',
                            help='Total dataset size',
                            default=1000000)
    
    chunk_size = Parameter('chunk_size',
                          help='Processing chunk size',
                          default=10000)
    
    @step
    def start(self):
        """
        YOUR TASK:
        1. Calculate number of chunks needed
        2. Create chunk metadata for parallel processing
        3. Initialize streaming data pipeline
        """
        self.n_chunks = self.total_samples // self.chunk_size
        # TODO: Implement chunk creation strategy
        self.next(self.generate_data_chunks, foreach='chunk_ids')
    
    @resources(memory=2000, cpu=1)  # Low memory per chunk
    @step
    def generate_data_chunks(self):
        """
        YOUR TASK:
        1. Generate data chunk (simulate large dataset)
        2. Apply preprocessing to chunk
        3. Save chunk to temporary storage
        """
        chunk_id = self.input
        # TODO: Implement memory-efficient chunk generation
        pass
    
    @step
    def incremental_model_training(self, inputs):
        """
        YOUR TASK:
        1. Implement incremental learning with SGDClassifier
        2. Process chunks sequentially to maintain memory efficiency
        3. Update model incrementally with partial_fit
        4. Track learning progress and convergence
        """
        pass
    
    @step
    def streaming_feature_engineering(self):
        """
        YOUR TASK:
        1. Implement streaming feature scaling
        2. Handle categorical variables in streaming fashion
        3. Create feature statistics without loading full dataset
        """
        pass
    
    @step
    def memory_efficient_evaluation(self):
        """
        YOUR TASK:
        1. Evaluate model on test data in chunks
        2. Compute metrics incrementally
        3. Generate learning curves without storing all predictions
        """
        pass
    
    @step
    def end(self):
        """Generate memory usage report and final model"""
        pass
```

2. **Implement advanced memory management techniques**:

   - **Memory mapping**: Use memory-mapped files for large datasets
   - **Lazy loading**: Load data only when needed
   - **Garbage collection**: Explicit memory cleanup between chunks
   - **Compression**: Use data compression to reduce memory footprint

3. **Create memory monitoring utilities**:

```python
import psutil
import gc

class MemoryMonitor:
    def __init__(self):
        self.peak_memory = 0
        self.memory_history = []
    
    def track_memory(self, stage_name):
        """
        YOUR TASK: Implement memory tracking
        1. Track current memory usage
        2. Record peak memory consumption
        3. Generate memory usage reports
        """
        pass
    
    def optimize_memory(self):
        """
        YOUR TASK: Implement memory optimization
        1. Force garbage collection
        2. Clear unnecessary variables
        3. Optimize pandas memory usage
        """
        pass
```

### Advanced Challenge: Distributed Processing

```python
@step
def distributed_chunk_processing(self):
    """
    YOUR ADVANCED TASK:
    Implement distributed processing for very large datasets:
    1. Split dataset across multiple machines
    2. Process chunks in parallel across cluster
    3. Aggregate results efficiently
    4. Handle node failures gracefully
    """
    pass
```

### Expected Outcomes
- Master memory-efficient data processing
- Understand incremental learning techniques
- Learn to handle datasets larger than RAM

### Verification Criteria
- [ ] Pipeline processes large datasets without memory errors
- [ ] Memory usage stays within specified limits
- [ ] Incremental learning converges properly
- [ ] Performance metrics are computed accurately
- [ ] Memory optimization techniques are implemented

---

## Exercise 5: Cloud Scaling & Production Deployment
**Difficulty: Advanced | Time: 90 minutes**

### Objective
Deploy production-ready ML pipelines to the cloud with automatic scaling, monitoring, and continuous deployment capabilities.

### Your Task

1. **Implement cloud-ready pipeline with @batch**:

```python
from metaflow import FlowSpec, step, Parameter, foreach, resources, batch

class ProductionMLFlow(FlowSpec):
    
    environment = Parameter('environment',
                          help='Deployment environment (dev/staging/prod)',
                          default='dev')
    
    auto_scale = Parameter('auto_scale',
                          help='Enable automatic scaling',
                          default=True)
    
    @step
    def start(self):
        """
        YOUR TASK:
        1. Setup environment-specific configurations
        2. Initialize monitoring and logging
        3. Validate input parameters
        """
        self.config = self.get_environment_config()
        # TODO: Implement environment setup
        self.next(self.data_validation)
    
    @batch(cpu=2, memory=4000, queue='ml-training')
    @step
    def data_validation(self):
        """
        YOUR TASK:
        1. Implement comprehensive data validation
        2. Check data quality metrics
        3. Detect data drift from training distribution
        4. Generate data health report
        """
        pass
    
    @batch(cpu=8, memory=16000, queue='ml-training-large')
    @step
    def distributed_training(self):
        """
        YOUR TASK:
        1. Implement distributed model training
        2. Use appropriate compute resources for production
        3. Add training progress monitoring
        4. Implement early stopping
        """
        pass
    
    @step
    def model_validation(self):
        """
        YOUR TASK:
        1. Validate model against production criteria
        2. A/B test against current production model
        3. Check for bias and fairness
        4. Generate model report card
        """
        pass
    
    @step
    def deployment_preparation(self):
        """
        YOUR TASK:
        1. Prepare model artifacts for deployment
        2. Create model metadata and documentation
        3. Package model with dependencies
        4. Generate deployment configuration
        """
        pass
    
    @step
    def end(self):
        """Complete deployment pipeline"""
        pass
    
    def get_environment_config(self):
        """
        YOUR TASK: Implement environment-specific configurations
        """
        configs = {
            'dev': {
                'compute_resources': 'small',
                'data_validation_level': 'basic',
                'monitoring_level': 'debug'
            },
            'staging': {
                'compute_resources': 'medium', 
                'data_validation_level': 'comprehensive',
                'monitoring_level': 'detailed'
            },
            'prod': {
                'compute_resources': 'large',
                'data_validation_level': 'strict',
                'monitoring_level': 'production'
            }
        }
        return configs.get(self.environment, configs['dev'])
```

2. **Implement production monitoring**:

```python
class ProductionMonitor:
    def __init__(self, environment):
        self.environment = environment
    
    def track_model_performance(self, predictions, actuals=None):
        """
        YOUR TASK:
        1. Track prediction quality metrics
        2. Monitor for model drift
        3. Alert on performance degradation
        4. Log to production monitoring system
        """
        pass
    
    def track_data_quality(self, input_data):
        """
        YOUR TASK:
        1. Monitor input data distributions
        2. Detect anomalies and outliers
        3. Track feature importance changes
        4. Alert on data quality issues
        """
        pass
    
    def generate_health_dashboard(self):
        """
        YOUR TASK:
        1. Create real-time model health dashboard
        2. Include key performance indicators
        3. Add alerting thresholds
        4. Export metrics for external monitoring
        """
        pass
```

3. **Implement continuous deployment pipeline**:

```python
@step
def continuous_deployment(self):
    """
    YOUR TASK:
    1. Implement blue-green deployment strategy
    2. Add canary deployment for gradual rollout
    3. Implement automatic rollback on failures
    4. Add deployment approval gates
    """
    
    # Blue-green deployment
    if self.environment == 'prod':
        # TODO: Implement production deployment strategy
        pass
    
    # Canary deployment
    # TODO: Implement gradual rollout
    pass
```

### Advanced Challenge: Multi-Region Deployment

```python
@step
def multi_region_deployment(self):
    """
    YOUR ADVANCED TASK:
    Deploy to multiple regions with:
    1. Regional model customization
    2. Data residency compliance
    3. Cross-region monitoring
    4. Regional failover capabilities
    """
    pass
```

### Expected Outcomes
- Master cloud deployment with Metaflow
- Implement production monitoring and alerting
- Learn continuous deployment strategies

### Verification Criteria
- [ ] Pipeline deploys successfully to cloud
- [ ] Monitoring and alerting work correctly
- [ ] Auto-scaling responds to load changes
- [ ] Deployment strategies are implemented
- [ ] Production health checks pass

---

## Exercise 6: Advanced Performance Optimization
**Difficulty: Expert | Time: 2+ hours**

### Objective
Master advanced optimization techniques including caching, incremental processing, pipeline parallelization, and performance profiling.

### Your Task

1. **Implement intelligent caching system**:

```python
from metaflow import FlowSpec, step, Parameter, foreach, resources
import hashlib
import pickle
import os

class OptimizedMLFlow(FlowSpec):
    
    cache_strategy = Parameter('cache_strategy',
                             help='Caching strategy (none/aggressive/smart)',
                             default='smart')
    
    @step
    def start(self):
        """
        YOUR TASK:
        1. Initialize caching system
        2. Create cache invalidation strategy
        3. Implement cache warming
        """
        self.cache_manager = CacheManager(self.cache_strategy)
        # TODO: Implement caching initialization
        self.next(self.cached_data_processing)
    
    @step
    def cached_data_processing(self):
        """
        YOUR TASK:
        1. Implement smart caching for preprocessing steps
        2. Cache expensive feature engineering operations
        3. Implement cache versioning for reproducibility
        """
        
        # Example caching pattern
        cache_key = self.generate_cache_key('preprocessing', self.data_params)
        
        if self.cache_manager.has_cache(cache_key):
            self.processed_data = self.cache_manager.load(cache_key)
        else:
            # Expensive processing
            self.processed_data = self.expensive_preprocessing()
            self.cache_manager.save(cache_key, self.processed_data)
    
    def generate_cache_key(self, operation, parameters):
        """
        YOUR TASK: Generate deterministic cache keys
        """
        pass
    
    @step
    def incremental_model_updates(self):
        """
        YOUR TASK:
        1. Implement incremental model updates
        2. Only retrain when necessary
        3. Detect when full retraining is needed
        4. Optimize hyperparameter search with warm starts
        """
        pass
    
    @step
    def performance_profiling(self):
        """
        YOUR TASK:
        1. Profile pipeline performance bottlenecks
        2. Measure CPU, memory, and I/O usage
        3. Identify optimization opportunities
        4. Generate performance report
        """
        pass
    
    @step
    def end(self):
        """Generate optimization recommendations"""
        pass
```

2. **Implement advanced profiling and optimization**:

```python
import cProfile
import pstats
import time
import functools

class PerformanceProfiler:
    def __init__(self):
        self.profiles = {}
        self.timing_data = {}
    
    def profile_step(self, step_name):
        """
        YOUR TASK: Decorator for profiling pipeline steps
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # TODO: Implement profiling logic
                pass
            return wrapper
        return decorator
    
    def identify_bottlenecks(self):
        """
        YOUR TASK:
        1. Analyze profiling data
        2. Identify performance bottlenecks
        3. Suggest optimization strategies
        """
        pass
    
    def optimize_pipeline_order(self, pipeline_steps):
        """
        YOUR TASK:
        1. Analyze step dependencies
        2. Optimize execution order
        3. Maximize parallel execution opportunities
        """
        pass
```

3. **Implement intelligent resource allocation**:

```python
class ResourceOptimizer:
    def __init__(self):
        self.resource_history = {}
        self.performance_history = {}
    
    def predict_resource_needs(self, algorithm, data_size):
        """
        YOUR TASK:
        1. Predict memory and CPU requirements
        2. Use historical performance data
        3. Account for data size scaling
        """
        pass
    
    def optimize_batch_sizes(self, available_memory, algorithm_type):
        """
        YOUR TASK:
        1. Calculate optimal batch sizes
        2. Balance memory usage and performance
        3. Account for algorithm-specific requirements
        """
        pass
    
    def dynamic_scaling_strategy(self, current_load, queue_depth):
        """
        YOUR TASK:
        1. Implement auto-scaling logic
        2. Scale up/down based on demand
        3. Optimize cost vs. performance trade-offs
        """
        pass
```

### Advanced Challenge: Pipeline Compiler

```python
class PipelineCompiler:
    """
    YOUR ADVANCED TASK:
    Create a pipeline compiler that:
    1. Analyzes pipeline dependencies
    2. Optimizes execution graph
    3. Eliminates redundant computations
    4. Maximizes parallelization
    5. Generates optimized execution plan
    """
    
    def analyze_dependencies(self, pipeline):
        pass
    
    def optimize_execution_graph(self, dependency_graph):
        pass
    
    def generate_execution_plan(self, optimized_graph):
        pass
```

### Expected Outcomes
- Master advanced optimization techniques
- Understand performance profiling and bottleneck identification
- Learn intelligent resource management

### Verification Criteria
- [ ] Caching system improves pipeline performance
- [ ] Performance profiling identifies bottlenecks
- [ ] Resource optimization reduces costs
- [ ] Pipeline execution time is minimized
- [ ] Memory usage is optimized

---

## 🎯 Bonus Challenge: Complete Production MLOps Pipeline
**Difficulty: Expert | Time: 4+ hours**

### Objective
Combine all optimization techniques into a complete production MLOps pipeline with monitoring, CI/CD, and automated optimization.

### Your Task

Design and implement a comprehensive MLOps pipeline that includes:

1. **Automated data ingestion and validation**
2. **Intelligent caching and incremental processing**
3. **Parallel hyperparameter optimization** 
4. **Robust error handling and fault tolerance**
5. **Production deployment with monitoring**
6. **Continuous model improvement**
7. **Cost optimization and resource management**

### Success Criteria

- Pipeline handles multiple datasets and algorithms
- Automatically scales based on demand
- Implements comprehensive monitoring
- Includes CI/CD for model updates
- Optimizes for both performance and cost
- Maintains high availability and reliability

---

## 📚 Additional Resources

### Metaflow Documentation
- [Metaflow Best Practices](https://docs.metaflow.org/metaflow/basics)
- [Resource Management](https://docs.metaflow.org/scaling/remote-tasks/introduction)
- [Error Handling](https://docs.metaflow.org/metaflow/failures)

### Performance Optimization
- [Python Performance Tips](https://docs.python.org/3/howto/perf_profiling.html)
- [Scikit-learn Performance](https://scikit-learn.org/stable/computing/computational_performance.html)
- [Memory Profiling](https://docs.python.org/3/library/tracemalloc.html)

### Production ML
- [MLOps Best Practices](https://ml-ops.org/)
- [Model Monitoring](https://neptune.ai/blog/ml-model-monitoring-best-tools)
- [Continuous Deployment](https://www.databricks.com/glossary/continuous-deployment)

---

## 🏆 Completion Checklist

Mark your progress as you complete each exercise:

- [ ] **Exercise 1**: Resource Optimization Fundamentals
- [ ] **Exercise 2**: Advanced Parallel Execution Patterns  
- [ ] **Exercise 3**: Production-Ready Error Handling
- [ ] **Exercise 4**: Memory-Efficient Large Dataset Processing
- [ ] **Exercise 5**: Cloud Scaling & Production Deployment
- [ ] **Exercise 6**: Advanced Performance Optimization
- [ ] **Bonus Challenge**: Complete Production MLOps Pipeline

**Congratulations! You've mastered advanced Metaflow pipeline optimization! 🚀**

Ready for Week 4: Advanced ML & LangGraph!