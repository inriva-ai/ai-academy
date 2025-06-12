# Metaflow Introduction for AI/ML Interns

## What is Metaflow?

Metaflow is an open-source Python framework originally developed at Netflix for managing data science and machine learning workflows. It helps you build, deploy, and manage ML pipelines with ease, making it perfect for both traditional ML projects and modern generative AI applications.

**Key Benefits:**
- **Reproducibility:** Version your code, data, and experiments automatically
- **Scalability:** Run workflows locally or scale to cloud infrastructure
- **Collaboration:** Share workflows and results with your team
- **Monitoring:** Track experiment progress and debug failures easily
- **Integration:** Works seamlessly with popular ML libraries and cloud platforms

---

## Installation and Setup

### Step 1: Install Metaflow

First, ensure you have Python 3.7+ installed. Then install Metaflow using pip:

```bash
# Install Metaflow
pip install metaflow

# Verify installation
python -c "import metaflow; print(metaflow.__version__)"
```

### Step 2: Set Up Your Environment

Create a dedicated directory for your Metaflow projects:

```bash
mkdir metaflow-projects
cd metaflow-projects
```

### Step 3: Configure Metaflow (Optional)

For basic local usage, no configuration is needed. For cloud features:

```bash
# Configure for AWS (optional)
metaflow configure aws

# Configure for Azure (optional) 
metaflow configure azure
```

---

## Setting Up an Open Source Notebook Environment

### Option 1: Jupyter Lab Setup

```bash
# Install JupyterLab if not already installed
pip install jupyterlab

# Install Metaflow kernel for Jupyter
pip install metaflow[jupyter]

# Start JupyterLab
jupyter lab
```

### Option 2: VS Code Setup

1. Install the Python extension for VS Code
2. Install the Jupyter extension for VS Code
3. Create a new `.ipynb` file
4. Select your Python interpreter with Metaflow installed

### Option 3: Google Colab Setup

In a new Colab notebook:

```python
# Install Metaflow in Colab
!pip install metaflow

# Import and verify
import metaflow
print(f"Metaflow version: {metaflow.__version__}")
```

---

## Core Concepts

### 1. Flows
A Flow is a collection of steps that define your ML workflow.

### 2. Steps
Individual units of work in your workflow (data loading, preprocessing, training, etc.).

### 3. Artifacts
Data produced by steps that can be accessed by other steps or external systems.

### 4. Parameters
Input values that can be changed when running the flow.

---

## Example 1: Basic Linear Regression Workflow

Create a new file called `linear_regression_flow.py`:

```python
from metaflow import FlowSpec, step, Parameter
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class LinearRegressionFlow(FlowSpec):
    """
    A simple linear regression workflow for house price prediction.
    """
    
    # Parameters that can be set when running the flow
    test_size = Parameter('test_size', 
                         help='Test set size ratio', 
                         default=0.2)
    
    random_state = Parameter('random_state',
                           help='Random state for reproducibility',
                           default=42)

    @step
    def start(self):
        """
        Initialize the flow and generate sample data.
        """
        print("Starting Linear Regression Workflow")
        
        # Generate sample house data
        np.random.seed(self.random_state)
        n_samples = 1000
        
        # Features: house size (sq ft), number of bedrooms, age
        house_size = np.random.normal(2000, 500, n_samples)
        bedrooms = np.random.poisson(3, n_samples)
        age = np.random.uniform(0, 50, n_samples)
        
        # Target: house price (with some realistic relationships)
        price = (house_size * 100 + 
                bedrooms * 5000 + 
                (50 - age) * 1000 + 
                np.random.normal(0, 20000, n_samples))
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'house_size': house_size,
            'bedrooms': bedrooms,
            'age': age,
            'price': price
        })
        
        print(f"Generated {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")
        
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        """
        Clean and prepare the data for modeling.
        """
        print("Preprocessing data...")
        
        # Basic data cleaning
        self.data = self.data.dropna()
        
        # Remove outliers (simple method)
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Keep data within 1.5 * IQR
        self.data = self.data[~((self.data < (Q1 - 1.5 * IQR)) | 
                               (self.data > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        print(f"Data shape after cleaning: {self.data.shape}")
        
        # Store basic statistics
        self.data_stats = self.data.describe()
        
        self.next(self.split_data)

    @step  
    def split_data(self):
        """
        Split data into training and testing sets.
        """
        print(f"Splitting data with test_size={self.test_size}")
        
        # Separate features and target
        X = self.data[['house_size', 'bedrooms', 'age']]
        y = self.data['price']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train the linear regression model.
        """
        print("Training linear regression model...")
        
        # Initialize and train model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        # Store model coefficients
        self.coefficients = {
            'house_size': self.model.coef_[0],
            'bedrooms': self.model.coef_[1], 
            'age': self.model.coef_[2],
            'intercept': self.model.intercept_
        }
        
        print("Model coefficients:")
        for feature, coef in self.coefficients.items():
            print(f"  {feature}: {coef:.2f}")
            
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        """
        Evaluate the trained model on test data.
        """
        print("Evaluating model performance...")
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.rmse = np.sqrt(self.mse)
        self.r2 = r2_score(self.y_test, self.y_pred)
        
        # Store evaluation results
        self.evaluation_results = {
            'mse': self.mse,
            'rmse': self.rmse,
            'r2_score': self.r2
        }
        
        print("Model Performance:")
        print(f"  MSE: {self.mse:,.2f}")
        print(f"  RMSE: {self.rmse:,.2f}")
        print(f"  R² Score: {self.r2:.4f}")
        
        self.next(self.end)

    @step
    def end(self):
        """
        Finalize the workflow and save results.
        """
        print("Workflow completed successfully!")
        
        # Summary of the entire workflow
        self.workflow_summary = {
            'data_shape': self.data.shape,
            'model_type': 'Linear Regression',
            'test_size': self.test_size,
            'performance': self.evaluation_results,
            'coefficients': self.coefficients
        }
        
        print("\nWorkflow Summary:")
        print(f"  Data samples: {self.workflow_summary['data_shape'][0]}")
        print(f"  Features: {self.workflow_summary['data_shape'][1] - 1}")
        print(f"  Model R² Score: {self.workflow_summary['performance']['r2']:.4f}")

if __name__ == '__main__':
    LinearRegressionFlow()
```

### Running the Flow

```bash
# Run the flow with default parameters
python linear_regression_flow.py run

# Run with custom parameters
python linear_regression_flow.py run --test_size 0.3 --random_state 123
```

---

## Example 2: Text Classification with Generative AI

Create `text_classification_flow.py`:

```python
from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import openai
import os

class TextClassificationFlow(FlowSpec):
    """
    A workflow comparing traditional ML text classification with LLM-based classification.
    """
    
    openai_api_key = Parameter('openai_api_key',
                              help='OpenAI API key for LLM comparison',
                              default='')
    
    sample_size = Parameter('sample_size',
                           help='Number of samples to use',
                           default=100)

    @step
    def start(self):
        """
        Initialize and create sample text data for sentiment analysis.
        """
        print("Starting Text Classification Workflow")
        
        # Create sample movie review data
        positive_reviews = [
            "This movie was absolutely fantastic! Great acting and plot.",
            "Amazing cinematography and wonderful storyline. Highly recommend!",
            "Brilliant performance by all actors. A masterpiece!",
            "Loved every minute of it. Best movie I've seen this year.",
            "Outstanding direction and incredible special effects.",
        ]
        
        negative_reviews = [
            "Terrible movie. Poor acting and boring plot.",
            "Waste of time. Nothing happens and the ending is awful.",
            "Poorly written script and bad direction.",
            "Disappointing. Expected much better from this director.",
            "Boring and confusing. Would not recommend.",
        ]
        
        # Expand the dataset
        reviews = []
        labels = []
        
        # Generate more samples by slight variations
        np.random.seed(42)
        for _ in range(self.sample_size // 2):
            # Add positive reviews
            review = np.random.choice(positive_reviews)
            reviews.append(review)
            labels.append('positive')
            
            # Add negative reviews  
            review = np.random.choice(negative_reviews)
            reviews.append(review)
            labels.append('negative')
        
        self.data = pd.DataFrame({
            'review': reviews,
            'sentiment': labels
        })
        
        print(f"Created dataset with {len(self.data)} reviews")
        print(f"Positive reviews: {sum(self.data['sentiment'] == 'positive')}")
        print(f"Negative reviews: {sum(self.data['sentiment'] == 'negative')}")
        
        self.next(self.split_data)

    @step
    def split_data(self):
        """
        Split data for training traditional ML model.
        """
        print("Splitting data for traditional ML approach...")
        
        X = self.data['review']
        y = self.data['sentiment']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        
        self.next(self.train_traditional_model, self.llm_classification)

    @step
    def train_traditional_model(self):
        """
        Train traditional ML model using TF-IDF and Logistic Regression.
        """
        print("Training traditional ML model...")
        
        # Vectorize text using TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        
        # Train logistic regression
        self.traditional_model = LogisticRegression(random_state=42)
        self.traditional_model.fit(X_train_tfidf, self.y_train)
        
        # Make predictions on test set
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        self.traditional_predictions = self.traditional_model.predict(X_test_tfidf)
        
        # Calculate accuracy
        self.traditional_accuracy = accuracy_score(self.y_test, self.traditional_predictions)
        
        print(f"Traditional ML Accuracy: {self.traditional_accuracy:.4f}")
        
        self.next(self.compare_results)

    @step  
    def llm_classification(self):
        """
        Perform classification using OpenAI's GPT model (if API key provided).
        """
        print("Running LLM-based classification...")
        
        if not self.openai_api_key:
            print("No OpenAI API key provided. Skipping LLM classification.")
            self.llm_predictions = ['positive'] * len(self.X_test)  # Dummy predictions
            self.llm_accuracy = 0.5
        else:
            # Set up OpenAI client
            openai.api_key = self.openai_api_key
            
            llm_predictions = []
            
            for review in self.X_test:
                prompt = f"""
                Analyze the sentiment of this movie review. 
                Respond with only 'positive' or 'negative'.
                
                Review: {review}
                
                Sentiment:"""
                
                try:
                    response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        max_tokens=1,
                        temperature=0
                    )
                    
                    prediction = response.choices[0].text.strip().lower()
                    if prediction not in ['positive', 'negative']:
                        prediction = 'positive'  # Default fallback
                        
                    llm_predictions.append(prediction)
                    
                except Exception as e:
                    print(f"Error with LLM prediction: {e}")
                    llm_predictions.append('positive')  # Fallback
            
            self.llm_predictions = llm_predictions
            self.llm_accuracy = accuracy_score(self.y_test, self.llm_predictions)
            
            print(f"LLM Accuracy: {self.llm_accuracy:.4f}")
        
        self.next(self.compare_results)

    @step
    def compare_results(self, inputs):
        """
        Compare results from traditional ML and LLM approaches.
        """
        print("Comparing Traditional ML vs LLM results...")
        
        # Merge results from parallel steps
        self.traditional_accuracy = inputs[0].traditional_accuracy
        self.traditional_predictions = inputs[0].traditional_predictions
        self.llm_accuracy = inputs[1].llm_accuracy
        self.llm_predictions = inputs[1].llm_predictions
        
        # Create comparison report
        self.comparison_results = {
            'traditional_ml_accuracy': self.traditional_accuracy,
            'llm_accuracy': self.llm_accuracy,
            'difference': abs(self.traditional_accuracy - self.llm_accuracy),
            'traditional_better': self.traditional_accuracy > self.llm_accuracy
        }
        
        print("\nComparison Results:")
        print(f"Traditional ML Accuracy: {self.traditional_accuracy:.4f}")
        print(f"LLM Accuracy: {self.llm_accuracy:.4f}")
        print(f"Difference: {self.comparison_results['difference']:.4f}")
        print(f"Traditional ML performs better: {self.comparison_results['traditional_better']}")
        
        self.next(self.end)

    @step
    def end(self):
        """
        Finalize the workflow.
        """
        print("Text Classification Workflow completed!")
        
        # Create final summary
        self.final_summary = {
            'dataset_size': len(self.data),
            'test_samples': len(self.X_test),
            'comparison_results': self.comparison_results
        }
        
        print(f"\nFinal Summary:")
        print(f"Dataset size: {self.final_summary['dataset_size']}")
        print(f"Test samples: {self.final_summary['test_samples']}")
        print(f"Best performing approach: {'Traditional ML' if self.comparison_results['traditional_better'] else 'LLM'}")

if __name__ == '__main__':
    TextClassificationFlow()
```

---

## Working with Metaflow in Notebooks

### Jupyter Notebook Example

Create a new notebook called `metaflow_exploration.ipynb`:

```python
# Cell 1: Import libraries and explore a completed flow
from metaflow import Flow, get_metadata
import pandas as pd

# Cell 2: List all flows
print("Available flows:")
flows = get_metadata().get_all_flows()
for flow in flows:
    print(f"  - {flow}")

# Cell 3: Explore a specific flow run
# Get the latest run of LinearRegressionFlow
flow = Flow('LinearRegressionFlow')
latest_run = flow.latest_run

print(f"Latest run ID: {latest_run.id}")
print(f"Run status: {latest_run.successful}")

# Cell 4: Access artifacts from the flow
if latest_run.successful:
    # Access the evaluation results
    evaluation_step = latest_run['evaluate_model']
    results = evaluation_step.task.data.evaluation_results
    
    print("Model Performance:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")

# Cell 5: Access and visualize data
start_step = latest_run['start']
data = start_step.task.data.data

print(f"Dataset shape: {data.shape}")
print("\nFirst few rows:")
print(data.head())

# Cell 6: Create visualizations
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(data['house_size'], data['price'], alpha=0.6)
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Price vs House Size')

plt.subplot(1, 3, 2)
plt.scatter(data['bedrooms'], data['price'], alpha=0.6)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price ($)')
plt.title('Price vs Bedrooms')

plt.subplot(1, 3, 3)
plt.scatter(data['age'], data['price'], alpha=0.6)
plt.xlabel('Age (years)')
plt.ylabel('Price ($)')
plt.title('Price vs Age')

plt.tight_layout()
plt.show()
```

---

## Best Practices for Interns

### 1. Code Organization
- Keep steps focused on single responsibilities
- Use descriptive step names and docstrings
- Store related artifacts together

### 2. Parameter Management
- Use Parameters for values that might change between runs
- Provide helpful descriptions and reasonable defaults
- Consider parameter validation in your steps

### 3. Error Handling
```python
@step
def robust_data_loading(self):
    """
    Load data with proper error handling.
    """
    try:
        self.data = pd.read_csv('data.csv')
        print(f"Successfully loaded {len(self.data)} rows")
    except FileNotFoundError:
        print("Data file not found. Creating sample data...")
        self.data = self.create_sample_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    self.next(self.preprocess_data)
```

### 4. Artifact Management
- Store intermediate results as artifacts for debugging
- Use meaningful variable names for artifacts
- Consider artifact size and storage implications

### 5. Documentation
- Document each step's purpose and expected inputs/outputs
- Include parameter descriptions
- Add comments for complex logic

---

## Integration with Your Internship Program

### Week 2-3: Basic Workflows
- Use Metaflow for data preprocessing pipelines
- Implement the linear regression example
- Practice version control with different parameter sets

### Week 4-5: Advanced Workflows  
- Build ensemble model comparison workflows
- Implement the text classification example
- Compare traditional ML vs generative AI approaches

### Week 6-7: Production Workflows
- Deploy workflows to cloud infrastructure
- Implement monitoring and error handling
- Build end-to-end ML pipelines

### Week 8: Capstone Integration
- Use Metaflow for your capstone project workflow
- Implement both ML and generative AI components
- Create reproducible, documented pipelines

---

## Troubleshooting Common Issues

### Issue 1: Import Errors
```bash
# Solution: Ensure all required packages are installed
pip install metaflow scikit-learn pandas numpy matplotlib seaborn
```

### Issue 2: Flow Not Found
```python
# Check if flow was run successfully
from metaflow import Flow
try:
    flow = Flow('YourFlowName')
    print("Flow found!")
except:
    print("Flow not found. Make sure you've run it first.")
```

### Issue 3: Memory Issues with Large Datasets
```python
@step
def process_large_data(self):
    """
    Process data in chunks to avoid memory issues.
    """
    chunk_size = 10000
    processed_chunks = []
    
    for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
        processed_chunk = self.process_chunk(chunk)
        processed_chunks.append(processed_chunk)
    
    self.data = pd.concat(processed_chunks, ignore_index=True)
    self.next(self.next_step)
```

---

## Next Steps

After completing this introduction:

1. **Practice**: Run both example workflows multiple times with different parameters
2. **Experiment**: Modify the examples to work with different datasets
3. **Integrate**: Use Metaflow in your weekly internship projects
4. **Explore**: Check out advanced Metaflow features like:
   - Remote execution on AWS/Azure
   - Custom decorators for specific requirements
   - Integration with MLflow for experiment tracking
   - Kubernetes deployment options

5. **Build**: Create your own workflows for your capstone project

Remember: Metaflow is designed to make ML workflows more manageable and reproducible. Start simple and gradually add complexity as you become more comfortable with the framework!

---

## Additional Resources

- [Official Metaflow Documentation](https://docs.metaflow.org/)
- [Metaflow Tutorials](https://outerbounds.com/docs/)
- [Netflix Tech Blog - Metaflow](https://netflixtechblog.com/open-sourcing-metaflow-a-human-centric-framework-for-data-science-fa72e04a5d9)
- [Metaflow GitHub Repository](https://github.com/Netflix/metaflow)
- [Community Examples](https://github.com/outerbounds/metaflow-examples)