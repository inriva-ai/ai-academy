# Metaflow Development Stack Setup Guide

## What is the Metaflow Development Stack?

The Metaflow development stack (metaflow-dev) is an enhanced development environment that provides additional tools and capabilities for building, testing, and deploying Metaflow workflows. It includes:

- **Enhanced UI**: Advanced workflow visualization and monitoring
- **Development Tools**: Better debugging and testing capabilities
- **Local Cloud Simulation**: Simulate cloud execution locally
- **Advanced Integrations**: Extended support for various ML frameworks
- **Performance Monitoring**: Detailed execution metrics and profiling

**Key Benefits:**
- **Faster Development**: Hot reloading and better debugging tools
- **Better Visualization**: Enhanced flow visualization and step-by-step execution tracking
- **Local Testing**: Test cloud-like scenarios without cloud costs
- **Team Collaboration**: Improved sharing and collaboration features
- **Production Readiness**: Tools to prepare workflows for production deployment

---

## Prerequisites

Before installing the metaflow-dev stack, ensure you have:

- **Python 3.8+** installed
- **Docker** installed and running
- **Git** installed
- **Node.js 16+** (for UI components)
- **At least 8GB RAM** available
- **10GB free disk space**

### Verify Prerequisites

```bash
# Check Python version
python --version

# Check Docker
docker --version
docker ps

# Check Node.js
node --version
npm --version

# Check Git
git --version
```

---

## Installation Methods

### Method 1: Quick Setup with Docker Compose (Recommended)

This is the easiest way to get started with the full development stack.

#### Step 1: Clone the Development Repository

```bash
# Clone the metaflow-dev repository
git clone https://github.com/outerbounds/metaflow-dev-stack.git
cd metaflow-dev-stack

# Or if using the official development branch
git clone https://github.com/Netflix/metaflow.git
cd metaflow
git checkout development
```

#### Step 2: Run Setup Script

```bash
# Make setup script executable
chmod +x setup-dev-stack.sh

# Run the setup (this may take 10-15 minutes)
./setup-dev-stack.sh
```

#### Step 3: Start the Development Stack

```bash
# Start all services
docker-compose up -d

# Check that all services are running
docker-compose ps
```

#### Step 4: Verify Installation

```bash
# Check the Metaflow UI (should open in browser)
open http://localhost:3000

# Check the API service
curl http://localhost:8080/api/flows

# Test Metaflow CLI
metaflow version
```

### Method 2: Manual Installation

For more control over the installation process:

#### Step 1: Install Enhanced Metaflow

```bash
# Create a new virtual environment
python -m venv metaflow-dev-env
source metaflow-dev-env/bin/activate  # On Windows: metaflow-dev-env\Scripts\activate

# Install development version of Metaflow
pip install git+https://github.com/Netflix/metaflow.git@development

# Install additional development dependencies
pip install metaflow[dev,ui,aws,azure]
```

#### Step 2: Install UI Dependencies

```bash
# Install Node.js dependencies for the UI
npm install -g @metaflow/ui-dev

# Start the development UI server
metaflow-ui-dev start --port 3000
```

#### Step 3: Setup Local Services

```bash
# Install and start local metadata service
pip install metaflow-service
metaflow-service start --port 8080

# Install local datastore
pip install metaflow-datastore
metaflow-datastore init
```

---

## Configuration

### Environment Variables

Create a `.env` file in your project directory:

```bash
# .env file
METAFLOW_DEFAULT_DATASTORE=local
METAFLOW_DEFAULT_METADATA=local
METAFLOW_DATASTORE_SYSROOT_LOCAL=/tmp/metaflow
METAFLOW_SERVICE_URL=http://localhost:8080
METAFLOW_UI_URL=http://localhost:3000
METAFLOW_DEFAULT_ENVIRONMENT=dev
```

### Development Configuration

Create a `metaflow_config.json` file:

```json
{
  "development": {
    "datastore": {
      "type": "local",
      "path": "/tmp/metaflow-dev"
    },
    "metadata": {
      "type": "local_service",
      "url": "http://localhost:8080"
    },
    "ui": {
      "url": "http://localhost:3000",
      "hot_reload": true
    },
    "compute": {
      "local": {
        "max_workers": 4
      },
      "docker": {
        "image": "python:3.9",
        "enabled": true
      }
    }
  }
}
```

### Apply Configuration

```bash
# Set environment variables
export METAFLOW_PROFILE=development
export METAFLOW_CONFIG_FILE=./metaflow_config.json

# Verify configuration
metaflow configure show
```

---

## Development Stack Components

### 1. Enhanced Metaflow UI

The development UI provides:
- **Real-time Execution Monitoring**: Watch flows execute step-by-step
- **Interactive Debugging**: Set breakpoints and inspect variables
- **Visual Flow Editor**: Drag-and-drop flow creation
- **Performance Profiling**: Detailed execution metrics

#### Accessing the UI

```bash
# Start the UI (if not already running)
metaflow-ui start

# Open in browser
open http://localhost:3000
```

### 2. Local Metadata Service

Enhanced metadata tracking for development:

```bash
# Start metadata service with development features
metaflow-service start --dev-mode --port 8080

# Enable detailed logging
metaflow-service start --log-level debug
```

### 3. Development CLI Tools

Additional CLI commands for development:

```bash
# Development-specific commands
metaflow dev --help

# Hot reload a flow during development
metaflow dev watch MyFlow.py

# Profile a flow execution
metaflow dev profile MyFlow.py run

# Debug a failed step
metaflow dev debug MyFlow.py run_id step_name
```

---

## Example 1: Enhanced Development Workflow

Create a new file called `enhanced_ml_flow.py`:

```python
from metaflow import FlowSpec, step, Parameter, catch, retry, timeout
from metaflow.cards import BlankCard
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class EnhancedMLFlow(FlowSpec):
    """
    Enhanced ML workflow with development stack features.
    """
    
    dataset_size = Parameter('dataset_size', 
                           help='Size of generated dataset',
                           default=1000)
    
    n_estimators = Parameter('n_estimators',
                           help='Number of trees in Random Forest',
                           default=100)
    
    debug_mode = Parameter('debug_mode',
                         help='Enable debug outputs',
                         default=True,
                         type=bool)

    @step
    def start(self):
        """
        Initialize the enhanced workflow with development features.
        """
        if self.debug_mode:
            print(f"🚀 Starting Enhanced ML Workflow")
            print(f"Dataset size: {self.dataset_size}")
            print(f"Random Forest estimators: {self.n_estimators}")
        
        # Generate sample data for binary classification
        np.random.seed(42)
        
        # Features
        X = np.random.randn(self.dataset_size, 5)
        # Add some correlation between features
        X[:, 1] = X[:, 0] * 0.5 + np.random.randn(self.dataset_size) * 0.5
        
        # Target with some logical relationship
        y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 0).astype(int)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(5)]
        self.data = pd.DataFrame(X, columns=feature_names)
        self.data['target'] = y
        
        # Store metadata for development tracking
        self.data_info = {
            'shape': self.data.shape,
            'target_distribution': self.data['target'].value_counts().to_dict(),
            'feature_stats': self.data.describe().to_dict()
        }
        
        if self.debug_mode:
            print(f"✅ Generated data: {self.data.shape}")
            print(f"Target distribution: {self.data_info['target_distribution']}")
        
        self.next(self.explore_data, self.preprocess_data)

    @BlankCard()
    @step
    def explore_data(self):
        """
        Create visualizations for data exploration (development feature).
        """
        if self.debug_mode:
            print("📊 Creating data exploration visualizations...")
        
        # Create correlation matrix
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data.corr()
        
        plt.subplot(2, 2, 1)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        # Feature distributions
        plt.subplot(2, 2, 2)
        self.data.iloc[:, :3].hist(bins=20, alpha=0.7)
        plt.title('Feature Distributions')
        
        # Target distribution
        plt.subplot(2, 2, 3)
        self.data['target'].value_counts().plot(kind='bar')
        plt.title('Target Distribution')
        plt.xticks(rotation=0)
        
        # Feature importance (rough estimate)
        plt.subplot(2, 2, 4)
        feature_importance = np.abs(correlation_matrix['target'][:-1])
        feature_importance.plot(kind='bar')
        plt.title('Feature-Target Correlation')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save visualization for the development UI
        self.exploration_plot = plt.gcf()
        
        if self.debug_mode:
            print("✅ Data exploration completed")
        
        self.next(self.train_model)

    @retry(times=3)
    @timeout(seconds=300)
    @step
    def preprocess_data(self):
        """
        Preprocess data with error handling (development features).
        """
        if self.debug_mode:
            print("🔧 Preprocessing data...")
        
        try:
            # Check for missing values
            missing_values = self.data.isnull().sum()
            if missing_values.any():
                if self.debug_mode:
                    print(f"⚠️ Found missing values: {missing_values[missing_values > 0]}")
                # Handle missing values
                self.data = self.data.fillna(self.data.mean())
            
            # Feature scaling (simple standardization)
            feature_cols = [col for col in self.data.columns if col != 'target']
            self.data[feature_cols] = (self.data[feature_cols] - self.data[feature_cols].mean()) / self.data[feature_cols].std()
            
            # Store preprocessing info
            self.preprocessing_info = {
                'missing_values_found': missing_values.sum(),
                'features_scaled': feature_cols,
                'final_shape': self.data.shape
            }
            
            if self.debug_mode:
                print(f"✅ Preprocessing completed: {self.preprocessing_info}")
        
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Preprocessing failed: {e}")
            raise
        
        self.next(self.train_model)

    @catch(var='training_error')
    @step
    def train_model(self, inputs):
        """
        Train model with enhanced error handling and monitoring.
        """
        # Merge inputs from parallel steps
        self.data = inputs.preprocess_data.data
        if hasattr(inputs, 'explore_data'):
            self.exploration_plot = inputs.explore_data.exploration_plot
        
        if self.debug_mode:
            print("🎯 Training Random Forest model...")
        
        # Prepare features and target
        feature_cols = [col for col in self.data.columns if col != 'target']
        X = self.data[feature_cols]
        y = self.data['target']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model with progress tracking
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42,
            verbose=1 if self.debug_mode else 0
        )
        
        # Fit model
        self.model.fit(self.X_train, self.y_train)
        
        # Store training info
        self.training_info = {
            'n_estimators': self.n_estimators,
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_))
        }
        
        if self.debug_mode:
            print(f"✅ Model training completed")
            print(f"Feature importance: {self.training_info['feature_importance']}")
        
        self.next(self.evaluate_model)

    @BlankCard()
    @step
    def evaluate_model(self):
        """
        Evaluate model with enhanced metrics and visualizations.
        """
        if self.debug_mode:
            print("📈 Evaluating model performance...")
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred),
            'recall': recall_score(self.y_test, self.y_pred),
            'f1_score': f1_score(self.y_test, self.y_pred),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba[:, 1])
        }
        
        # Detailed classification report
        self.classification_report = classification_report(
            self.y_test, self.y_pred, output_dict=True
        )
        
        # Create evaluation visualizations
        plt.figure(figsize=(15, 10))
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba[:, 1])
        plt.plot(fpr, tpr, label=f'ROC AUC = {self.metrics["roc_auc"]:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # Feature importance
        plt.subplot(2, 3, 3)
        importance_df = pd.DataFrame({
            'feature': list(self.training_info['feature_importance'].keys()),
            'importance': list(self.training_info['feature_importance'].values())
        }).sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance')
        
        # Prediction distribution
        plt.subplot(2, 3, 4)
        plt.hist(self.y_pred_proba[:, 1], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Predicted Probability (Class 1)')
        plt.ylabel('Count')
        plt.title('Prediction Probability Distribution')
        
        # Metrics comparison
        plt.subplot(2, 3, 5)
        metrics_names = list(self.metrics.keys())
        metrics_values = list(self.metrics.values())
        plt.bar(metrics_names, metrics_values)
        plt.title('Model Metrics')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Learning curve simulation
        plt.subplot(2, 3, 6)
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        for size in train_sizes:
            n_samples = int(len(self.X_train) * size)
            temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
            temp_model.fit(self.X_train[:n_samples], self.y_train[:n_samples])
            score = temp_model.score(self.X_test, self.y_test)
            train_scores.append(score)
        
        plt.plot(train_sizes, train_scores, 'o-')
        plt.xlabel('Training Set Size Ratio')
        plt.ylabel('Test Accuracy')
        plt.title('Learning Curve')
        
        plt.tight_layout()
        self.evaluation_plot = plt.gcf()
        
        if self.debug_mode:
            print(f"✅ Model evaluation completed")
            for metric, value in self.metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        self.next(self.end)

    @step
    def end(self):
        """
        Finalize workflow with comprehensive summary.
        """
        if self.debug_mode:
            print("🎉 Enhanced ML Workflow completed!")
        
        # Create comprehensive summary
        self.final_summary = {
            'workflow_parameters': {
                'dataset_size': self.dataset_size,
                'n_estimators': self.n_estimators,
                'debug_mode': self.debug_mode
            },
            'data_info': self.data_info,
            'training_info': self.training_info,
            'performance_metrics': self.metrics,
            'best_metric': max(self.metrics.items(), key=lambda x: x[1])
        }
        
        print(f"\n📋 Workflow Summary:")
        print(f"Dataset: {self.final_summary['data_info']['shape']}")
        print(f"Best metric: {self.final_summary['best_metric'][0]} = {self.final_summary['best_metric'][1]:.4f}")
        print(f"Training samples: {self.final_summary['training_info']['training_samples']}")
        print(f"Test samples: {self.final_summary['training_info']['test_samples']}")

if __name__ == '__main__':
    EnhancedMLFlow()
```

### Running with Development Features

```bash
# Run with development UI monitoring
metaflow dev run enhanced_ml_flow.py run --dataset_size 2000

# Run with profiling
metaflow dev profile enhanced_ml_flow.py run --n_estimators 200

# Watch for changes and auto-reload
metaflow dev watch enhanced_ml_flow.py --dataset_size 1500
```

---

## Example 2: Advanced Generative AI Pipeline

Create `advanced_genai_flow.py`:

```python
from metaflow import FlowSpec, step, Parameter, card, catch, current
from metaflow.cards import Markdown, Table, Image
import pandas as pd
import requests
import json
import time
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedGenAIFlow(FlowSpec):
    """
    Advanced Generative AI pipeline with development stack integration.
    """
    
    api_provider = Parameter('api_provider',
                           help='API provider (openai, anthropic, or mock)',
                           default='mock')
    
    batch_size = Parameter('batch_size',
                         help='Batch size for processing',
                         default=5)
    
    max_retries = Parameter('max_retries',
                          help='Maximum retries for API calls',
                          default=3)

    @step
    def start(self):
        """
        Initialize the advanced GenAI workflow.
        """
        print("🚀 Starting Advanced Generative AI Pipeline")
        
        # Sample tasks for different GenAI applications
        self.tasks = [
            {
                'id': 1,
                'type': 'text_generation',
                'prompt': 'Write a short product description for a smart water bottle',
                'parameters': {'max_tokens': 100, 'temperature': 0.7}
            },
            {
                'id': 2,
                'type': 'code_generation',
                'prompt': 'Write a Python function to calculate fibonacci numbers',
                'parameters': {'max_tokens': 200, 'temperature': 0.3}
            },
            {
                'id': 3,
                'type': 'data_analysis',
                'prompt': 'Analyze this sales trend: Q1: $100k, Q2: $120k, Q3: $150k, Q4: $180k',
                'parameters': {'max_tokens': 150, 'temperature': 0.5}
            },
            {
                'id': 4,
                'type': 'creative_writing',
                'prompt': 'Write a haiku about machine learning',
                'parameters': {'max_tokens': 50, 'temperature': 0.9}
            },
            {
                'id': 5,
                'type': 'summarization',
                'prompt': 'Summarize: Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.',
                'parameters': {'max_tokens': 80, 'temperature': 0.4}
            }
        ]
        
        # Development tracking
        self.workflow_metadata = {
            'start_time': time.time(),
            'api_provider': self.api_provider,
            'total_tasks': len(self.tasks),
            'batch_size': self.batch_size
        }
        
        print(f"📝 Created {len(self.tasks)} tasks for processing")
        print(f"🔧 Using API provider: {self.api_provider}")
        
        self.next(self.setup_api_client)

    @catch(var='api_setup_error')
    @step
    def setup_api_client(self):
        """
        Setup API client with development error handling.
        """
        print(f"🔌 Setting up {self.api_provider} API client...")
        
        if self.api_provider == 'openai':
            # OpenAI setup (mock for development)
            self.api_config = {
                'base_url': 'https://api.openai.com/v1',
                'model': 'gpt-3.5-turbo',
                'headers': {'Authorization': 'Bearer YOUR_API_KEY'}
            }
        elif self.api_provider == 'anthropic':
            # Anthropic setup (mock for development)
            self.api_config = {
                'base_url': 'https://api.anthropic.com/v1',
                'model': 'claude-3-sonnet-20240229',
                'headers': {'x-api-key': 'YOUR_API_KEY'}
            }
        else:
            # Mock API for development
            self.api_config = {
                'base_url': 'mock',
                'model': 'mock-model',
                'headers': {}
            }
        
        # Test API connection
        self.api_status = self.test_api_connection()
        
        print(f"✅ API client setup completed: {self.api_status}")
        
        self.next(self.process_batches)

    def test_api_connection(self) -> Dict:
        """Test API connection with mock responses for development."""
        if self.api_config['base_url'] == 'mock':
            return {
                'status': 'connected',
                'latency_ms': 50,
                'rate_limit': 1000,
                'model_available': True
            }
        else:
            # In real implementation, test actual API
            return {
                'status': 'mock_connected',
                'latency_ms': 100,
                'rate_limit': 60,
                'model_available': True
            }

    @step
    def process_batches(self):
        """
        Process tasks in batches for efficient API usage.
        """
        print(f"🔄 Processing {len(self.tasks)} tasks in batches of {self.batch_size}")
        
        # Split tasks into batches
        batches = []
        for i in range(0, len(self.tasks), self.batch_size):
            batch = self.tasks[i:i + self.batch_size]
            batches.append(batch)
        
        self.batch_info = {
            'total_batches': len(batches),
            'batch_sizes': [len(batch) for batch in batches]
        }
        
        print(f"📦 Created {len(batches)} batches: {self.batch_info['batch_sizes']}")
        
        # Process each batch
        self.batch_results = []
        for batch_idx, batch in enumerate(batches):
            print(f"Processing batch {batch_idx + 1}/{len(batches)}...")
            batch_result = self.process_single_batch(batch_idx, batch)
            self.batch_results.append(batch_result)
        
        self.next(self.analyze_results)

    def process_single_batch(self, batch_idx: int, batch: List[Dict]) -> Dict:
        """Process a single batch of tasks."""
        batch_start_time = time.time()
        results = []
        
        for task in batch:
            try:
                # Simulate API call with retry logic
                result = self.call_api_with_retry(task)
                results.append(result)
                
                # Add small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Failed to process task {task['id']}: {e}")
                results.append({
                    'task_id': task['id'],
                    'success': False,
                    'error': str(e),
                    'response': None
                })
        
        batch_end_time = time.time()
        
        return {
            'batch_id': batch_idx,
            'processing_time': batch_end_time - batch_start_time,
            'success_count': sum(1 for r in results if r['success']),
            'failure_count': sum(1 for r in results if not r['success']),
            'results': results
        }

    def call_api_with_retry(self, task: Dict) -> Dict:
        """Call API with retry logic for development."""
        for attempt in range(self.max_retries):
            try:
                # Mock API call for development
                if self.api_config['base_url'] == 'mock':
                    response = self.mock_api_call(task)
                else:
                    response = self.real_api_call(task)
                
                return {
                    'task_id': task['id'],
                    'success': True,
                    'response': response,
                    'attempts': attempt + 1,
                    'processing_time': 0.5 + attempt * 0.1  # Mock processing time
                }
                
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed for task {task['id']}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def mock_api_call(self, task: Dict) -> Dict:
        """Mock API responses for development."""
        mock_responses = {
            'text_generation': "Introducing the AquaSmart Pro - a revolutionary smart water bottle that tracks your hydration, reminds you to drink, and keeps your water at the perfect temperature all day long.",
            'code_generation': "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            'data_analysis': "The sales data shows a strong upward trend with 20% quarter-over-quarter growth. Q1 to Q4 represents an 80% increase, indicating robust business performance and market expansion.",
            'creative_writing': "Data flows like streams\nAlgorithms learn and adapt\nWisdom from patterns",
            'summarization': "ML is an AI subset enabling computers to learn and improve from experience without explicit programming."
        }
        
        return {
            'text': mock_responses.get(task['type'], "Mock response for " + task['type']),
            'tokens_used': task['parameters']['max_tokens'] // 2,
            'model': self.api_config['model'],
            'finish_reason': 'stop'
        }

    def real_api_call(self, task: Dict) -> Dict:
        """Real API call implementation (placeholder)."""
        # In a real implementation, this would make actual API calls
        # For development, we'll use mock responses
        return self.mock_api_call(task)

    @card
    @step
    def analyze_results(self):
        """
        Analyze results with enhanced visualizations for development.
        """
        print("📊 Analyzing batch processing results...")
        
        # Collect all results
        all_results = []
        for batch in self.batch_results:
            all_results.extend(batch['results'])
        
        # Calculate summary statistics
        total_tasks = len(all_results)
        successful_tasks = sum(1 for r in all_results if r['success'])
        failed_tasks = total_tasks - successful_tasks
        
        # Processing time analysis
        processing_times = [r.get('processing_time', 0) for r in all_results if r['success']]
        total_processing_time = sum(batch['processing_time'] for batch in self.batch_results)
        
        # Token usage analysis (for successful tasks)
        token_usage = []
        for result in all_results:
            if result['success'] and result['response']:
                tokens = result['response'].get('tokens_used', 0)
                token_usage.append(tokens)
        
        self.analysis_summary = {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': successful_tasks / total_tasks if total_tasks > 0 else 0,
            'total_processing_time': total_processing_time,
            'average_task_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'total_tokens_used': sum(token_usage),
            'average_tokens_per_task': sum(token_usage) / len(token_usage) if token_usage else 0
        }
        
        # Create visualizations
        self.create_analysis_visualizations(all_results)
        
        # Generate Metaflow card content
        self.create_results_card()
        
        print(f"✅ Analysis completed:")
        print(f"  Success rate: {self.analysis_summary['success_rate']:.2%}")
        print(f"  Total processing time: {self.analysis_summary['total_processing_time']:.2f}s")
        print(f"  Average tokens per task: {self.analysis_summary['average_tokens_per_task']:.0f}")
        
        self.next(self.end)

    def create_analysis_visualizations(self, all_results: List[Dict]):
        """Create comprehensive visualizations for development analysis."""
        plt.figure(figsize=(16, 12))
        
        # Success/Failure distribution
        plt.subplot(2, 4, 1)
        success_counts = [self.analysis_summary['successful_tasks'], self.analysis_summary['failed_tasks']]
        plt.pie(success_counts, labels=['Success', 'Failure'], autopct='%1.1f%%', startangle=90)
        plt.title('Task Success Rate')
        
        # Processing time by batch
        plt.subplot(2, 4, 2)
        batch_times = [batch['processing_time'] for batch in self.batch_results]
        batch_ids = [f"Batch {i+1}" for i in range(len(batch_times))]
        plt.bar(batch_ids, batch_times)
        plt.title('Processing Time by Batch')
        plt.xticks(rotation=45)
        
        # Token usage distribution
        plt.subplot(2, 4, 3)
        token_usage = []
        for result in all_results:
            if result['success'] and result['response']:
                tokens = result['response'].get('tokens_used', 0)
                token_usage.append(tokens)
        
        if token_usage:
            plt.hist(token_usage, bins=10, edgecolor='black', alpha=0.7)
            plt.title('Token Usage Distribution')
            plt.xlabel('Tokens Used')
            plt.ylabel('Frequency')
        
        # Task type performance
        plt.subplot(2, 4, 4)
        task_types = {}
        for i, task in enumerate(self.tasks):
            task_type = task['type']
            success = all_results[i]['success'] if i < len(all_results) else False
            if task_type not in task_types:
                task_types[task_type] = {'success': 0, 'total': 0}
            task_types[task_type]['total'] += 1
            if success:
                task_types[task_type]['success'] += 1
        
        types = list(task_types.keys())
        success_rates = [task_types[t]['success'] / task_types[t]['total'] for t in types]
        plt.bar(types, success_rates)
        plt.title('Success Rate by Task Type')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Batch size vs processing time
        plt.subplot(2, 4, 5)
        batch_sizes = [len(batch['results']) for batch in self.batch_results]
        batch_times = [batch['processing_time'] for batch in self.batch_results]
        plt.scatter(batch_sizes, batch_times)
        plt.title('Batch Size vs Processing Time')
        plt.xlabel('Batch Size')
        plt.ylabel('Processing Time (s)')
        
        # Retry attempts distribution
        plt.subplot(2, 4, 6)
        retry_counts = []
        for result in all_results:
            if result['success']:
                attempts = result.get('attempts', 1)
                retry_counts.append(attempts)
        
        if retry_counts:
            unique_attempts, counts = np.unique(retry_counts, return_counts=True)
            plt.bar(unique_attempts, counts)
            plt.title('Retry Attempts Distribution')
            plt.xlabel('Number of Attempts')
            plt.ylabel('Count')
        
        # API latency over time
        plt.subplot(2, 4, 7)
        successful_results = [r for r in all_results if r['success']]
        latencies = [r.get('processing_time', 0) for r in successful_results]
        plt.plot(range(len(latencies)), latencies, 'o-')
        plt.title('API Latency Over Time')
        plt.xlabel('Request Number')
        plt.ylabel('Latency (s)')
        
        # Cost estimation (mock)
        plt.subplot(2, 4, 8)
        cost_per_1k_tokens = 0.002  # Mock pricing
        total_cost = (self.analysis_summary['total_tokens_used'] / 1000) * cost_per_1k_tokens
        plt.bar(['Estimated Cost'], [total_cost])
        plt.title(f'Estimated API Cost: ${total_cost:.4f}')
        plt.ylabel('Cost ($)')
        
        plt.tight_layout()
        self.analysis_plot = plt.gcf()

    def create_results_card(self):
        """Create a Metaflow card with results summary."""
        # This will be displayed in the Metaflow UI
        current.card.append(Markdown("# Advanced GenAI Pipeline Results"))
        
        # Summary table
        summary_data = [
            ["Metric", "Value"],
            ["Total Tasks", str(self.analysis_summary['total_tasks'])],
            ["Success Rate", f"{self.analysis_summary['success_rate']:.2%}"],
            ["Total Processing Time", f"{self.analysis_summary['total_processing_time']:.2f}s"],
            ["Average Tokens/Task", f"{self.analysis_summary['average_tokens_per_task']:.0f}"],
            ["API Provider", self.api_provider],
            ["Batch Size", str(self.batch_size)]
        ]
        
        current.card.append(Table(summary_data))
        
        # Add sample results
        current.card.append(Markdown("## Sample Generated Content"))
        
        for result in [r for r in self.batch_results[0]['results'] if r['success']][:3]:
            task = next(t for t in self.tasks if t['id'] == result['task_id'])
            current.card.append(Markdown(f"**{task['type'].replace('_', ' ').title()}:**"))
            current.card.append(Markdown(f"*Prompt:* {task['prompt'][:100]}..."))
            current.card.append(Markdown(f"*Response:* {result['response']['text'][:200]}..."))
            current.card.append(Markdown("---"))

    @step
    def end(self):
        """
        Finalize the advanced workflow with comprehensive reporting.
        """
        print("🎉 Advanced Generative AI Pipeline completed!")
        
        # Calculate final metrics
        end_time = time.time()
        total_duration = end_time - self.workflow_metadata['start_time']
        
        self.final_report = {
            'workflow_metadata': self.workflow_metadata,
            'analysis_summary': self.analysis_summary,
            'batch_info': self.batch_info,
            'api_status': self.api_status,
            'total_duration': total_duration,
            'throughput': self.analysis_summary['total_tasks'] / total_duration
        }
        
        print(f"\n📋 Final Report:")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Throughput: {self.final_report['throughput']:.2f} tasks/second")
        print(f"API Provider: {self.api_provider}")
        print(f"Success Rate: {self.analysis_summary['success_rate']:.2%}")

if __name__ == '__main__':
    AdvancedGenAIFlow()
```

### Running Advanced Pipeline

```bash
# Run with development monitoring
metaflow dev run advanced_genai_flow.py run --api_provider mock --batch_size 3

# Profile the execution
metaflow dev profile advanced_genai_flow.py run --batch_size 10

# Watch with auto-reload
metaflow dev watch advanced_genai_flow.py --api_provider openai
```

---

## Development UI Features

### Real-time Monitoring

The development UI provides several enhanced features:

#### 1. Live Execution View
```bash
# Start UI with live monitoring
metaflow-ui start --live-mode

# Watch specific flow execution
metaflow dev monitor MyFlow.py run_id
```

#### 2. Interactive Debugging
```bash
# Set breakpoints in your flow
metaflow dev debug MyFlow.py run_id step_name

# Inspect variables at runtime
metaflow dev inspect MyFlow.py run_id step_name variable_name
```

#### 3. Performance Profiling
```bash
# Profile memory usage
metaflow dev profile --memory MyFlow.py run

# Profile CPU usage
metaflow dev profile --cpu MyFlow.py run

# Profile I/O operations
metaflow dev profile --io MyFlow.py run
```

---

## Development Best Practices

### 1. Use Development Decorators

```python
from metaflow import step, card, catch, retry, timeout
from metaflow.cards import BlankCard, Markdown, Table

class DevelopmentFlow(FlowSpec):
    
    @catch(var='data_error')  # Catch and store errors
    @retry(times=3)           # Retry failed steps
    @timeout(seconds=300)     # Timeout long-running steps
    @card                     # Generate UI card
    @step
    def robust_step(self):
        # Your step implementation
        pass
```

### 2. Implement Comprehensive Logging

```python
import logging
from metaflow import current

class LoggingFlow(FlowSpec):
    
    @step
    def start(self):
        # Setup logging for development
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Log to Metaflow artifacts
        self.logger.info(f"Starting flow run: {current.run_id}")
        
        self.next(self.process_data)
```

### 3. Use Development Parameters

```python
from metaflow import Parameter

class ConfigurableFlow(FlowSpec):
    
    # Development switches
    debug_mode = Parameter('debug', default=True, type=bool)
    sample_size = Parameter('sample_size', default=1000)
    use_cache = Parameter('cache', default=True, type=bool)
    
    @step
    def start(self):
        if self.debug_mode:
            print("Running in debug mode")
        
        # Use smaller dataset in development
        self.data_size = self.sample_size if self.debug_mode else 100000
        
        self.next(self.process_data)
```

### 4. Mock External Dependencies

```python
class MockableFlow(FlowSpec):
    
    environment = Parameter('env', default='development')
    
    @step
    def start(self):
        if self.environment == 'development':
            self.api_client = MockAPIClient()
        else:
            self.api_client = RealAPIClient()
        
        self.next(self.call_api)
    
    def call_api(self):
        response = self.api_client.get_data()
        # Process response
        self.next(self.end)
```

---

## Troubleshooting Development Stack

### Common Issues and Solutions

#### 1. Docker Issues
```bash
# Clean up Docker containers
docker-compose down -v
docker system prune -f

# Rebuild containers
docker-compose build --no-cache
docker-compose up -d
```

#### 2. UI Not Loading
```bash
# Check UI service status
docker-compose logs metaflow-ui

# Restart UI service
docker-compose restart metaflow-ui

# Check port conflicts
netstat -tulpn | grep :3000
```

#### 3. Metadata Service Issues
```bash
# Reset metadata database
metaflow-service reset --force

# Check service logs
docker-compose logs metaflow-service

# Verify database connection
metaflow configure show
```

#### 4. Performance Issues
```bash
# Increase Docker memory allocation
# Edit docker-compose.yml:
# services:
#   metaflow-service:
#     deploy:
#       resources:
#         limits:
#           memory: 4G

# Clean up old artifacts
metaflow-datastore clean --older-than 7d
```

#### 5. Development Tools Not Working
```bash
# Reinstall development tools
pip uninstall metaflow
pip install git+https://github.com/Netflix/metaflow.git@development

# Clear Python cache
find . -type d -name __pycache__ -delete
find . -name "*.pyc" -delete
```

---

## Integration with Your Internship Program

### Week 2-3: Basic Development Workflow
- Set up development stack
- Practice with enhanced UI features
- Implement basic flows with development decorators

### Week 4-5: Advanced Development Features
- Use profiling and debugging tools
- Implement parallel processing workflows
- Practice with mock API integrations

### Week 6-7: Production Preparation
- Test deployment workflows
- Implement comprehensive monitoring
- Practice with real API integrations

### Week 8: Capstone with Full Stack
- Use all development features for capstone project
- Implement production-ready workflows
- Create comprehensive documentation and monitoring

---

## Additional Resources

- [Metaflow Development Documentation](https://docs.metaflow.org/internals-of-metaflow/contributing-to-metaflow)
- [Outerbounds Development Guide](https://outerbounds.com/docs/development-guide/)
- [Metaflow UI Documentation](https://docs.metaflow.org/metaflow/visualizing-results/easy-custom-reports-with-card-decorator)
- [Docker Compose Best Practices](https://docs.docker.com/compose/production/)
- [Development Stack Examples](https://github.com/outerbounds/metaflow-dev-examples)

The development stack provides powerful tools for building, testing, and monitoring ML workflows. Start with basic features and gradually explore advanced capabilities as you become more comfortable with the platform!