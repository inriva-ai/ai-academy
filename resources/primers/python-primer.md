# Python Programming Primer for AI/ML Interns

## What is Python and Why for AI/ML?

Python is a high-level, interpreted programming language that has become the de facto standard for data science, machine learning, and AI development. Its popularity in these fields stems from:

**Key Advantages for AI/ML:**
- **Simplicity**: Clean, readable syntax that lets you focus on concepts rather than complex code
- **Rich Ecosystem**: Extensive libraries for data manipulation, ML, and AI (NumPy, pandas, scikit-learn, TensorFlow, PyTorch)
- **Community Support**: Massive community with tutorials, documentation, and help resources
- **Versatility**: Can handle everything from data preprocessing to model deployment
- **Integration**: Easy integration with other languages and tools

**What You'll Learn:**
- Python fundamentals and syntax
- Data structures and control flow
- Functions and object-oriented programming
- Essential libraries for data science and ML
- File handling and data manipulation
- Basic debugging and testing

---

## Installation and Setup

### Step 1: Choose Your Python Distribution

#### Option A: Anaconda (Recommended for Data Science)

Anaconda includes Python plus essential data science packages and tools.

**Download and Install:**
1. Visit [anaconda.com](https://www.anaconda.com/products/distribution)
2. Download the installer for your operating system
3. Run the installer with default settings

**Verify Installation:**
```bash
# Check Python version
python --version

# Check conda
conda --version

# List installed packages
conda list
```

#### Option B: Official Python + pip

**Download and Install:**
1. Visit [python.org](https://www.python.org/downloads/)
2. Download Python 3.9+ for your operating system
3. During installation, check "Add Python to PATH"

**Verify Installation:**
```bash
# Check Python version
python --version

# Check pip (package manager)
pip --version
```

### Step 2: Set Up Development Environment

#### Create a Virtual Environment

Virtual environments keep your projects isolated and dependencies organized.

**Using conda (Anaconda):**
```bash
# Create environment for AI/ML work
conda create -n aiml-env python=3.9

# Activate environment
conda activate aiml-env

# Install essential packages
conda install numpy pandas matplotlib seaborn scikit-learn jupyter
```

**Using venv (Standard Python):**
```bash
# Create virtual environment
python -m venv aiml-env

# Activate environment
# On Windows:
aiml-env\Scripts\activate
# On macOS/Linux:
source aiml-env/bin/activate

# Install essential packages
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Step 3: Choose Your Code Editor

#### Option A: Jupyter Notebook (Interactive Development)
```bash
# Start Jupyter Notebook
jupyter notebook

# Or Jupyter Lab (more advanced)
jupyter lab
```

#### Option B: VS Code (Full Development Environment)
1. Download [VS Code](https://code.visualstudio.com/)
2. Install Python extension
3. Install Jupyter extension
4. Select your Python interpreter (Ctrl+Shift+P → "Python: Select Interpreter")

#### Option C: PyCharm (Professional IDE)
1. Download [PyCharm Community Edition](https://www.jetbrains.com/pycharm/)
2. Configure Python interpreter during setup

### Step 4: Verify Complete Setup

Create a test file `setup_test.py`:

```python
# Test script to verify your Python setup
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Python Setup Verification")
print("=" * 30)
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Test basic functionality
data = np.array([1, 2, 3, 4, 5])
df = pd.DataFrame({'numbers': data, 'squares': data**2})
print("\nSample data:")
print(df)

# Test plotting
plt.figure(figsize=(6, 4))
plt.plot(data, data**2, 'o-')
plt.title('Setup Test: Numbers vs Squares')
plt.xlabel('Numbers')
plt.ylabel('Squares')
plt.show()

print("\n✅ Setup verification complete!")
```

Run the test:
```bash
python setup_test.py
```

---

## Python Fundamentals

### 1. Basic Syntax and Variables

```python
# Variables and data types
name = "Alice"              # String
age = 25                    # Integer
height = 5.6               # Float
is_student = True          # Boolean

# Print and string formatting
print(f"Hello, I'm {name}, {age} years old")
print("Height: {:.1f} feet".format(height))

# Multiple assignment
x, y, z = 1, 2, 3

# Constants (by convention, use ALL_CAPS)
PI = 3.14159
MAX_ATTEMPTS = 3
```

### 2. Data Structures

#### Lists (Ordered, Mutable)
```python
# Creating and manipulating lists
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]

# Accessing elements
first_fruit = fruits[0]        # "apple"
last_number = numbers[-1]      # 5 (negative indexing)

# Slicing
some_numbers = numbers[1:4]    # [2, 3, 4]
every_other = numbers[::2]     # [1, 3, 5]

# Adding elements
fruits.append("grape")         # Add to end
fruits.insert(1, "kiwi")      # Insert at position

# List comprehensions (very important for data science!)
squares = [x**2 for x in range(1, 6)]  # [1, 4, 9, 16, 25]
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]  # [4, 16, 36, 64, 100]
```

#### Dictionaries (Key-Value Pairs)
```python
# Creating dictionaries
student = {
    "name": "John",
    "age": 22,
    "grades": [85, 92, 78, 96],
    "major": "Computer Science"
}

# Accessing values
student_name = student["name"]
student_age = student.get("age", 0)  # Safe access with default

# Adding/updating
student["gpa"] = 3.7
student["email"] = "john@email.com"

# Dictionary comprehension
squared_dict = {x: x**2 for x in range(1, 6)}  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
```

#### Sets (Unique Elements)
```python
# Creating sets
unique_numbers = {1, 2, 3, 4, 5}
colors = set(["red", "blue", "green", "red"])  # Duplicates removed

# Set operations
set1 = {1, 2, 3}
set2 = {3, 4, 5}
intersection = set1 & set2    # {3}
union = set1 | set2          # {1, 2, 3, 4, 5}
```

#### Tuples (Ordered, Immutable)
```python
# Creating tuples
coordinates = (10, 20)
rgb_color = (255, 128, 0)

# Unpacking
x, y = coordinates
red, green, blue = rgb_color

# Named tuples (more advanced)
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)  # 10 20
```

### 3. Control Flow

#### Conditional Statements
```python
# Basic if-else
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"

print(f"Score: {score}, Grade: {grade}")

# Ternary operator
status = "Pass" if score >= 70 else "Fail"
```

#### Loops
```python
# For loops
numbers = [1, 2, 3, 4, 5]

# Basic for loop
for num in numbers:
    print(f"Number: {num}")

# For loop with index
for i, num in enumerate(numbers):
    print(f"Index {i}: {num}")

# Range function
for i in range(5):          # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 10, 2):   # 2, 4, 6, 8
    print(i)

# While loops
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1

# Loop control
for num in range(10):
    if num == 3:
        continue  # Skip this iteration
    if num == 7:
        break     # Exit loop
    print(num)
```

### 4. Functions

```python
# Basic function definition
def greet(name):
    """Function to greet a person."""
    return f"Hello, {name}!"

# Function with default parameters
def calculate_area(length, width=1):
    """Calculate area of rectangle."""
    return length * width

# Function with multiple return values
def get_stats(numbers):
    """Calculate basic statistics."""
    return min(numbers), max(numbers), sum(numbers) / len(numbers)

# Using functions
message = greet("Alice")
area = calculate_area(5, 3)      # 15
square_area = calculate_area(4)  # 4 (width defaults to 1)

min_val, max_val, avg_val = get_stats([1, 2, 3, 4, 5])

# Lambda functions (anonymous functions)
square = lambda x: x**2
numbers = [1, 2, 3, 4, 5]
squared = list(map(square, numbers))  # [1, 4, 9, 16, 25]

# Filter and map examples
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
doubled = list(map(lambda x: x * 2, numbers))
```

### 5. Error Handling

```python
# Basic try-except
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None
    except TypeError:
        print("Error: Please provide numbers only!")
        return None

# Multiple exceptions and finally
def read_file_safely(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"File {filename} not found!")
    except PermissionError:
        print(f"Permission denied to read {filename}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("File operation completed")

# Custom exceptions
class InvalidDataError(Exception):
    """Custom exception for invalid data."""
    pass

def validate_data(data):
    if not isinstance(data, (int, float)):
        raise InvalidDataError("Data must be a number")
    if data < 0:
        raise InvalidDataError("Data must be positive")
    return data
```

---

## Essential Libraries for AI/ML

### 1. NumPy (Numerical Computing)

```python
import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
random_nums = np.random.random((3, 3))

# Array operations
print(f"Shape: {arr2.shape}")      # (2, 3)
print(f"Size: {arr2.size}")        # 6
print(f"Data type: {arr2.dtype}")  # int64

# Mathematical operations (vectorized!)
arr = np.array([1, 2, 3, 4, 5])
squared = arr ** 2                  # [1, 4, 9, 16, 25]
sqrt_vals = np.sqrt(arr)           # [1.0, 1.414, 1.732, 2.0, 2.236]

# Statistical functions
mean_val = np.mean(arr)
std_val = np.std(arr)
max_val = np.max(arr)

# Boolean indexing (very useful!)
large_values = arr[arr > 3]        # [4, 5]

# Example: Basic linear algebra
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])
matrix_product = np.dot(matrix_a, matrix_b)  # Matrix multiplication
print("Matrix multiplication result:")
print(matrix_product)
```

### 2. Pandas (Data Manipulation)

```python
import pandas as pd

# Creating DataFrames
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'city': ['New York', 'London', 'Tokyo', 'Paris'],
    'salary': [70000, 80000, 90000, 75000]
}
df = pd.DataFrame(data)

# Basic DataFrame operations
print("DataFrame shape:", df.shape)
print("Column names:", df.columns.tolist())
print("Data types:")
print(df.dtypes)

# Data inspection
print("First 3 rows:")
print(df.head(3))

print("Basic statistics:")
print(df.describe())

print("Info about DataFrame:")
print(df.info())

# Selecting data
names = df['name']                    # Select column
young_people = df[df['age'] < 30]     # Filter rows
subset = df[['name', 'salary']]       # Select multiple columns

# Data manipulation
df['salary_k'] = df['salary'] / 1000  # Create new column
df_sorted = df.sort_values('age')     # Sort by age
grouped = df.groupby('city').mean()   # Group by city and calculate mean

# Example: Working with real-world data patterns
# Simulating some data analysis tasks
print("\nData Analysis Examples:")

# Calculate salary statistics by age group
df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40], labels=['20s', '30s'])
salary_by_age = df.groupby('age_group')['salary'].agg(['mean', 'std', 'count'])
print("Salary statistics by age group:")
print(salary_by_age)

# Find highest earners
top_earners = df.nlargest(2, 'salary')[['name', 'salary']]
print("\nTop earners:")
print(top_earners)
```

### 3. Matplotlib (Basic Plotting)

```python
import matplotlib.pyplot as plt
import numpy as np

# Basic line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.plot(x, np.cos(x), label='cos(x)', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trigonometric Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Scatter plot
axes[0, 0].scatter(df['age'], df['salary'], alpha=0.7)
axes[0, 0].set_title('Age vs Salary')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Salary')

# Bar plot
ages = df['age']
names = df['name']
axes[0, 1].bar(names, ages, color=['red', 'green', 'blue', 'orange'])
axes[0, 1].set_title('Ages by Person')
axes[0, 1].tick_params(axis='x', rotation=45)

# Histogram
random_data = np.random.normal(100, 15, 1000)
axes[1, 0].hist(random_data, bins=30, alpha=0.7, color='purple')
axes[1, 0].set_title('Normal Distribution')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')

# Pie chart
city_counts = df['city'].value_counts()
axes[1, 1].pie(city_counts.values, labels=city_counts.index, autopct='%1.1f%%')
axes[1, 1].set_title('Distribution by City')

plt.tight_layout()
plt.show()
```

---

## File Handling and Data Processing

### 1. Working with Files

```python
# Reading and writing text files
def write_sample_data():
    """Create a sample data file."""
    data = """Name,Age,City,Salary
Alice,25,New York,70000
Bob,30,London,80000
Charlie,35,Tokyo,90000
Diana,28,Paris,75000"""
    
    with open('sample_data.csv', 'w') as file:
        file.write(data)

def read_text_file(filename):
    """Read and process a text file."""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            # Process each line
            processed_lines = [line.strip().split(',') for line in lines]
            return processed_lines
    except FileNotFoundError:
        print(f"File {filename} not found!")
        return []

# Create sample file and read it
write_sample_data()
data = read_text_file('sample_data.csv')
print("Raw data from file:")
for row in data:
    print(row)
```

### 2. Working with CSV Files

```python
import csv

# Reading CSV files with csv module
def read_csv_properly(filename):
    """Read CSV file using csv module."""
    with open(filename, 'r') as file:
        csv_reader = csv.DictReader(file)
        data = []
        for row in csv_reader:
            # Convert numeric fields
            row['Age'] = int(row['Age'])
            row['Salary'] = int(row['Salary'])
            data.append(row)
        return data

# Reading with pandas (preferred for data science)
def analyze_csv_data(filename):
    """Analyze CSV data using pandas."""
    df = pd.read_csv(filename)
    
    analysis = {
        'total_records': len(df),
        'average_age': df['Age'].mean(),
        'average_salary': df['Salary'].mean(),
        'cities': df['City'].unique().tolist(),
        'age_range': (df['Age'].min(), df['Age'].max())
    }
    
    return analysis

# Use the functions
csv_data = read_csv_properly('sample_data.csv')
analysis = analyze_csv_data('sample_data.csv')

print("Analysis results:")
for key, value in analysis.items():
    print(f"{key}: {value}")
```

### 3. Working with JSON Data

```python
import json

# Sample JSON data (common in APIs)
sample_json_data = {
    "employees": [
        {"id": 1, "name": "Alice", "department": "Engineering", "skills": ["Python", "ML"]},
        {"id": 2, "name": "Bob", "department": "Data Science", "skills": ["R", "Statistics"]},
        {"id": 3, "name": "Charlie", "department": "Engineering", "skills": ["Java", "SQL"]}
    ],
    "company": "TechCorp",
    "location": "San Francisco"
}

# Write JSON to file
with open('company_data.json', 'w') as file:
    json.dump(sample_json_data, file, indent=2)

# Read JSON from file
with open('company_data.json', 'r') as file:
    loaded_data = json.load(file)

# Process JSON data
def analyze_company_data(data):
    """Analyze company data from JSON."""
    employees = data['employees']
    
    # Extract information
    all_skills = []
    departments = []
    
    for emp in employees:
        all_skills.extend(emp['skills'])
        departments.append(emp['department'])
    
    # Count occurrences
    from collections import Counter
    skill_counts = Counter(all_skills)
    dept_counts = Counter(departments)
    
    return {
        'total_employees': len(employees),
        'departments': dict(dept_counts),
        'top_skills': dict(skill_counts.most_common(3)),
        'company': data['company']
    }

json_analysis = analyze_company_data(loaded_data)
print("JSON analysis:")
print(json.dumps(json_analysis, indent=2))
```

---

## Object-Oriented Programming Basics

```python
# Basic class definition
class DataProcessor:
    """A class for processing data."""
    
    # Class variable (shared by all instances)
    supported_formats = ['csv', 'json', 'txt']
    
    def __init__(self, name, data_format='csv'):
        """Initialize the processor."""
        self.name = name
        self.data_format = data_format
        self.processed_count = 0
        
        if data_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {data_format}")
    
    def load_data(self, filename):
        """Load data from file."""
        if self.data_format == 'csv':
            self.data = pd.read_csv(filename)
        elif self.data_format == 'json':
            with open(filename, 'r') as f:
                self.data = json.load(f)
        else:
            with open(filename, 'r') as f:
                self.data = f.read()
        
        print(f"Loaded data from {filename}")
        return self.data
    
    def process_data(self):
        """Process the loaded data."""
        if not hasattr(self, 'data'):
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.processed_count += 1
        print(f"Processing data (count: {self.processed_count})")
        
        # Return summary based on data type
        if isinstance(self.data, pd.DataFrame):
            return {
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'memory_usage': self.data.memory_usage().sum()
            }
        else:
            return {'type': type(self.data).__name__}
    
    def __str__(self):
        """String representation of the processor."""
        return f"DataProcessor(name='{self.name}', format='{self.data_format}')"
    
    def __repr__(self):
        """Developer representation."""
        return f"DataProcessor('{self.name}', '{self.data_format}')"

# Using the class
processor = DataProcessor("MyProcessor", "csv")
print(processor)

# Load and process data
try:
    processor.load_data('sample_data.csv')
    summary = processor.process_data()
    print("Processing summary:", summary)
except Exception as e:
    print(f"Error: {e}")

# Inheritance example
class AdvancedProcessor(DataProcessor):
    """Advanced data processor with additional features."""
    
    def __init__(self, name, data_format='csv', auto_clean=True):
        super().__init__(name, data_format)
        self.auto_clean = auto_clean
    
    def clean_data(self):
        """Clean the data."""
        if isinstance(self.data, pd.DataFrame):
            # Remove duplicates and handle missing values
            original_rows = len(self.data)
            self.data = self.data.drop_duplicates()
            self.data = self.data.fillna(self.data.mean(numeric_only=True))
            
            cleaned_rows = len(self.data)
            print(f"Cleaned data: {original_rows - cleaned_rows} rows removed")
        
        return self.data
    
    def process_data(self):
        """Enhanced processing with auto-cleaning."""
        if self.auto_clean and isinstance(self.data, pd.DataFrame):
            self.clean_data()
        
        return super().process_data()

# Using inheritance
advanced_processor = AdvancedProcessor("AdvancedProcessor", auto_clean=True)
print(f"Advanced processor: {advanced_processor}")
```

---

## Practical Examples for AI/ML

### Example 1: Simple Data Analysis Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_sample_sales_data(n_days=365):
    """Generate sample sales data for analysis."""
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate sales data with trends and seasonality
    base_sales = 1000
    trend = np.linspace(0, 200, n_days)  # Upward trend
    seasonality = 100 * np.sin(2 * np.pi * np.arange(n_days) / 365)  # Yearly cycle
    noise = np.random.normal(0, 50, n_days)  # Random noise
    
    sales = base_sales + trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'day_of_week': [d.strftime('%A') for d in dates],
        'month': [d.strftime('%B') for d in dates],
        'quarter': [f"Q{(d.month-1)//3 + 1}" for d in dates]
    })
    
    return df

def analyze_sales_data(df):
    """Perform comprehensive sales analysis."""
    analysis_results = {}
    
    # Basic statistics
    analysis_results['basic_stats'] = {
        'total_sales': df['sales'].sum(),
        'average_daily_sales': df['sales'].mean(),
        'best_day': df.loc[df['sales'].idxmax()],
        'worst_day': df.loc[df['sales'].idxmin()]
    }
    
    # Trends by time period
    analysis_results['monthly_avg'] = df.groupby('month')['sales'].mean().sort_values(ascending=False)
    analysis_results['quarterly_sum'] = df.groupby('quarter')['sales'].sum()
    analysis_results['day_of_week_avg'] = df.groupby('day_of_week')['sales'].mean()
    
    # Growth analysis
    df_monthly = df.groupby(df['date'].dt.to_period('M'))['sales'].sum()
    analysis_results['monthly_growth'] = df_monthly.pct_change().dropna()
    
    return analysis_results

def visualize_sales_analysis(df, analysis):
    """Create visualizations for sales analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Time series plot
    axes[0, 0].plot(df['date'], df['sales'])
    axes[0, 0].set_title('Daily Sales Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Sales')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Monthly averages
    monthly_avg = analysis['monthly_avg']
    axes[0, 1].bar(monthly_avg.index, monthly_avg.values)
    axes[0, 1].set_title('Average Sales by Month')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Day of week analysis
    dow_avg = analysis['day_of_week_avg']
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_ordered = dow_avg.reindex(days_order)
    axes[0, 2].bar(dow_ordered.index, dow_ordered.values)
    axes[0, 2].set_title('Average Sales by Day of Week')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Quarterly comparison
    quarterly = analysis['quarterly_sum']
    axes[1, 0].pie(quarterly.values, labels=quarterly.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Sales Distribution by Quarter')
    
    # Sales distribution histogram
    axes[1, 1].hist(df['sales'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution of Daily Sales')
    axes[1, 1].set_xlabel('Sales Amount')
    axes[1, 1].set_ylabel('Frequency')
    
    # Monthly growth rate
    growth = analysis['monthly_growth']
    axes[1, 2].plot(growth.index.astype(str), growth.values, marker='o')
    axes[1, 2].set_title('Month-over-Month Growth Rate')
    axes[1, 2].set_ylabel('Growth Rate (%)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# Run the complete analysis
print("🔄 Generating sample sales data...")
sales_df = generate_sample_sales_data(365)

print("📊 Analyzing sales data...")
analysis_results = analyze_sales_data(sales_df)

print("📈 Key insights:")
print(f"Total annual sales: ${analysis_results['basic_stats']['total_sales']:,.2f}")
print(f"Average daily sales: ${analysis_results['basic_stats']['average_daily_sales']:,.2f}")
print(f"Best performing month: {analysis_results['monthly_avg'].index[0]}")
print(f"Best performing day: {analysis_results['day_of_week_avg'].index[0]}")

print("🎨 Creating visualizations...")
visualize_sales_analysis(sales_df, analysis_results)
```

### Example 2: Basic Machine Learning Preparation

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def prepare_ml_dataset():
    """Create a dataset suitable for machine learning."""
    np.random.seed(42)
    
    # Generate features
    n_samples = 1000
    house_size = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.poisson(3, n_samples)
    bathrooms = np.random.poisson(2, n_samples)
    age = np.random.uniform(0, 50, n_samples)
    
    # Generate categorical features
    neighborhoods = np.random.choice(['Downtown', 'Suburbs', 'Uptown'], n_samples)
    house_types = np.random.choice(['Single Family', 'Condo', 'Townhouse'], n_samples)
    
    # Generate target with realistic relationships
    price = (house_size * 100 + 
             bedrooms * 15000 + 
             bathrooms * 10000 + 
             (50 - age) * 1000 +
             np.random.normal(0, 20000, n_samples))
    
    # Create DataFrame
    df = pd.DataFrame({
        'house_size': house_size,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'neighborhood': neighborhoods,
        'house_type': house_types,
        'price': price
    })
    
    return df

def preprocess_for_ml(df):
    """Preprocess data for machine learning."""
    df_processed = df.copy()
    
    # Handle missing values (if any)
    df_processed = df_processed.dropna()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['neighborhood', 'house_type']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Remove original categorical columns
    df_processed = df_processed.drop(categorical_columns, axis=1)
    
    # Feature scaling
    feature_columns = [col for col in df_processed.columns if col != 'price']
    scaler = StandardScaler()
    df_processed[feature_columns] = scaler.fit_transform(df_processed[feature_columns])
    
    return df_processed, label_encoders, scaler

def simple_ml_pipeline(df):
    """Demonstrate a complete ML pipeline."""
    print("🔧 Preprocessing data...")
    df_processed, encoders, scaler = preprocess_for_ml(df)
    
    # Prepare features and target
    feature_columns = [col for col in df_processed.columns if col != 'price']
    X = df_processed[feature_columns]
    y = df_processed['price']
    
    print(f"Features: {feature_columns}")
    print(f"Dataset shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train model
    print("🎯 Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("📊 Model Performance:")
    print(f"MSE: {mse:,.2f}")
    print(f"RMSE: {np.sqrt(mse):,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\n🔍 Feature Importance (by coefficient magnitude):")
    print(feature_importance)
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    
    plt.subplot(1, 3, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.subplot(1, 3, 3)
    plt.bar(feature_importance['feature'], np.abs(feature_importance['coefficient']))
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    plt.ylabel('|Coefficient|')
    
    plt.tight_layout()
    plt.show()
    
    return model, scaler, encoders

# Run the ML pipeline
print("🏠 Creating house price dataset...")
house_df = prepare_ml_dataset()

print("Dataset overview:")
print(house_df.head())
print(f"\nDataset shape: {house_df.shape}")
print(f"Data types:\n{house_df.dtypes}")

print("\n🤖 Running ML pipeline...")
trained_model, trained_scaler, trained_encoders = simple_ml_pipeline(house_df)
```

---

## Debugging and Testing

### 1. Debugging Techniques

```python
# Using print statements for debugging
def debug_function(data):
    print(f"Input data type: {type(data)}")
    print(f"Input data length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    
    if isinstance(data, list):
        print(f"First few elements: {data[:3]}")
        print(f"Last few elements: {data[-3:]}")
    
    # Process data
    result = [x * 2 for x in data if isinstance(x, (int, float))]
    
    print(f"Result: {result}")
    return result

# Using assertions for validation
def validate_input(value, min_val=0, max_val=100):
    assert isinstance(value, (int, float)), f"Expected number, got {type(value)}"
    assert min_val <= value <= max_val, f"Value {value} not in range [{min_val}, {max_val}]"
    return True

# Python debugger (pdb)
import pdb

def complex_calculation(numbers):
    result = 0
    for i, num in enumerate(numbers):
        # Uncomment the next line to set a breakpoint
        # pdb.set_trace()
        
        if num < 0:
            result -= num * 2
        else:
            result += num
    
    return result

# Test the functions
test_data = [1, 2, 3, 4, 5]
debug_result = debug_function(test_data)

try:
    validate_input(50)  # Should pass
    validate_input(150) # Should fail
except AssertionError as e:
    print(f"Validation error: {e}")
```

### 2. Basic Testing

```python
import unittest

class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing functions."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = [1, 2, 3, 4, 5]
        self.sample_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
    
    def test_list_processing(self):
        """Test list processing function."""
        result = debug_function(self.sample_data)
        expected = [2, 4, 6, 8, 10]
        self.assertEqual(result, expected)
    
    def test_validation_success(self):
        """Test successful validation."""
        result = validate_input(50, 0, 100)
        self.assertTrue(result)
    
    def test_validation_failure(self):
        """Test validation failure."""
        with self.assertRaises(AssertionError):
            validate_input(150, 0, 100)
    
    def test_dataframe_operations(self):
        """Test DataFrame operations."""
        result = self.sample_df.sum()
        expected_sum_A = 6  # 1 + 2 + 3
        expected_sum_B = 15 # 4 + 5 + 6
        
        self.assertEqual(result['A'], expected_sum_A)
        self.assertEqual(result['B'], expected_sum_B)

# Simple test runner
def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)

# Uncomment to run tests
# run_tests()

# Alternative: pytest (more modern, install with pip install pytest)
"""
To use pytest, create a file test_example.py:

def test_simple_math():
    assert 2 + 2 == 4

def test_list_operations():
    data = [1, 2, 3]
    result = [x * 2 for x in data]
    assert result == [2, 4, 6]

Then run: pytest test_example.py
"""
```

---

## Best Practices for AI/ML Development

### 1. Code Organization

```python
# Good project structure:
"""
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── exploratory/
│   └── experiments/
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── models.py
│   └── utils.py
├── tests/
├── requirements.txt
└── README.md
"""

# Example utils.py
def load_config(config_path):
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def setup_logging(log_level='INFO'):
    """Set up logging configuration."""
    import logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def save_model_results(model, results, filepath):
    """Save model and results for later use."""
    import pickle
    
    model_data = {
        'model': model,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {filepath}")
```

### 2. Documentation and Comments

```python
def process_dataset(data, target_column, test_size=0.2, random_state=42):
    """
    Process dataset for machine learning.
    
    This function handles the complete preprocessing pipeline including
    data cleaning, feature engineering, and train-test splitting.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset containing features and target
    target_column : str
        Name of the target column
    test_size : float, default=0.2
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducible results
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) - preprocessed data splits
    
    Example:
    --------
    >>> df = pd.read_csv('data.csv')
    >>> X_train, X_test, y_train, y_test = process_dataset(df, 'price')
    >>> print(f"Training samples: {len(X_train)}")
    """
    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
```

### 3. Error Handling and Validation

```python
class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

def validate_ml_data(X, y, min_samples=100):
    """
    Validate data for machine learning.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    min_samples : int
        Minimum number of samples required
    
    Raises:
    -------
    DataValidationError
        If data validation fails
    """
    # Check data types
    if not hasattr(X, 'shape'):
        raise DataValidationError("X must be array-like with shape attribute")
    
    if not hasattr(y, '__len__'):
        raise DataValidationError("y must have length")
    
    # Check dimensions
    if len(X) != len(y):
        raise DataValidationError(f"X and y must have same length: {len(X)} vs {len(y)}")
    
    if len(X) < min_samples:
        raise DataValidationError(f"Not enough samples: {len(X)} < {min_samples}")
    
    # Check for missing values
    if hasattr(X, 'isnull') and X.isnull().any().any():
        raise DataValidationError("X contains missing values")
    
    if hasattr(y, 'isnull') and y.isnull().any():
        raise DataValidationError("y contains missing values")
    
    print("✅ Data validation passed")
    return True

# Example usage with error handling
def safe_model_training(X, y):
    """Train model with comprehensive error handling."""
    try:
        # Validate data
        validate_ml_data(X, y)
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        print("✅ Model training completed successfully")
        return model
        
    except DataValidationError as e:
        print(f"❌ Data validation failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error during training: {e}")
        return None
```

---

## Next Steps and Resources

### Integration with Your Internship Program

**Week 1 (Foundations):**
- Complete this Python primer
- Practice with all examples
- Set up development environment
- Master basic data structures and control flow

**Week 2 (Data Manipulation):**
- Focus on pandas and NumPy
- Practice file handling and data cleaning
- Work through the sales analysis example

**Week 3-4 (ML Preparation):**
- Master the ML pipeline example
- Practice preprocessing and validation
- Learn debugging and testing techniques

**Ongoing:**
- Apply these concepts in weekly projects
- Use Python best practices in all assignments
- Build on this foundation for advanced topics

### Essential Practice Exercises

1. **Data Analysis Challenge**:
   - Download a real dataset from Kaggle
   - Perform complete exploratory data analysis
   - Create visualizations and insights

2. **Mini ML Project**:
   - Implement the house price prediction from scratch
   - Add feature engineering
   - Compare different preprocessing approaches

3. **File Processing Challenge**:
   - Work with different file formats (CSV, JSON, Excel)
   - Build a data pipeline that handles multiple sources
   - Implement error handling and validation

### Recommended Resources

**Free Online Resources:**
- [Python.org Official Tutorial](https://docs.python.org/3/tutorial/)
- [Kaggle Learn Python Course](https://www.kaggle.com/learn/python)
- [Real Python Tutorials](https://realpython.com/)
- [Python for Data Analysis (free chapters)](https://wesmckinney.com/book/)

**Interactive Practice:**
- [HackerRank Python Domain](https://www.hackerrank.com/domains/python)
- [LeetCode Python Problems](https://leetcode.com/)
- [Codewars Python Kata](https://www.codewars.com/)

**Books:**
- "Python Crash Course" by Eric Matthes
- "Automate the Boring Stuff with Python" by Al Sweigart (free online)
- "Python for Data Analysis" by Wes McKinney

**YouTube Channels:**
- Corey Schafer (Python tutorials)
- sentdex (Python for ML/Data Science)
- Tech With Tim (Python projects)

### Quick Reference Cheat Sheet

```python
# Essential imports for data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Common data operations
df = pd.read_csv('file.csv')
df.info()                    # Dataset overview
df.describe()                # Statistical summary
df.head()                    # First 5 rows
df['column'].value_counts()  # Count unique values
df.groupby('col').mean()     # Group by and aggregate

# Common ML workflow
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Common plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y)              # Line plot
plt.scatter(x, y)           # Scatter plot
plt.hist(data, bins=30)     # Histogram
plt.show()
```

Remember: Python is a tool to solve problems. Focus on understanding concepts and applying them to real-world scenarios. The syntax will become natural with practice!

---

## Appendix: Advanced Topics

### A. Web Scraping for Data Collection

Web scraping is essential for gathering real-world data for ML projects.

#### A.1 Basic Web Scraping with Requests and BeautifulSoup

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from urllib.parse import urljoin, urlparse

def scrape_webpage(url, headers=None):
    """
    Scrape content from a webpage.
    
    Parameters:
    -----------
    url : str
        URL to scrape
    headers : dict, optional
        HTTP headers to include in request
    
    Returns:
    --------
    BeautifulSoup object or None if failed
    """
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
        
    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

def scrape_news_headlines(base_url="https://news.ycombinator.com"):
    """
    Example: Scrape news headlines from Hacker News.
    """
    soup = scrape_webpage(base_url)
    if not soup:
        return []
    
    headlines = []
    
    # Find all story links
    story_links = soup.find_all('span', class_='titleline')
    
    for link in story_links:
        try:
            # Extract headline and URL
            a_tag = link.find('a')
            if a_tag:
                title = a_tag.get_text().strip()
                url = a_tag.get('href', '')
                
                # Handle relative URLs
                if url.startswith('http'):
                    full_url = url
                else:
                    full_url = urljoin(base_url, url)
                
                headlines.append({
                    'title': title,
                    'url': full_url,
                    'domain': urlparse(full_url).netloc
                })
                
        except Exception as e:
            print(f"Error processing headline: {e}")
            continue
    
    return headlines

def scrape_product_data(search_term="laptop", max_pages=3):
    """
    Example: Scrape product data (mock structure).
    Note: Always check robots.txt and terms of service!
    """
    products = []
    
    for page in range(1, max_pages + 1):
        print(f"Scraping page {page}...")
        
        # Mock URL structure (replace with actual site)
        url = f"https://example-store.com/search?q={search_term}&page={page}"
        
        soup = scrape_webpage(url)
        if not soup:
            continue
        
        # Mock product extraction (adapt to actual site structure)
        product_containers = soup.find_all('div', class_='product-item')
        
        for container in product_containers:
            try:
                name = container.find('h3', class_='product-name')
                price = container.find('span', class_='price')
                rating = container.find('div', class_='rating')
                
                if name and price:
                    product_data = {
                        'name': name.get_text().strip(),
                        'price': price.get_text().strip(),
                        'rating': rating.get_text().strip() if rating else 'N/A',
                        'page': page
                    }
                    products.append(product_data)
                    
            except Exception as e:
                print(f"Error extracting product data: {e}")
                continue
        
        # Be respectful - add delay between requests
        time.sleep(1)
    
    return products

# Example usage
print("🕷️ Web Scraping Examples")

# Scrape news headlines
headlines = scrape_news_headlines()
if headlines:
    headlines_df = pd.DataFrame(headlines)
    print(f"Scraped {len(headlines)} headlines")
    print(headlines_df.head())
    
    # Analyze domains
    domain_counts = headlines_df['domain'].value_counts()
    print("\nTop domains:")
    print(domain_counts.head())

# Advanced scraping with session management
class WebScraper:
    """Advanced web scraper with session management."""
    
    def __init__(self, base_url, delay=1):
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_with_pagination(self, path_template, max_pages=5):
        """Scrape multiple pages with pagination."""
        all_data = []
        
        for page in range(1, max_pages + 1):
            url = urljoin(self.base_url, path_template.format(page=page))
            
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                page_data = self.extract_data(soup, page)
                all_data.extend(page_data)
                
                print(f"Scraped page {page}: {len(page_data)} items")
                
                # Respectful delay
                time.sleep(self.delay)
                
            except Exception as e:
                print(f"Error on page {page}: {e}")
                break
        
        return all_data
    
    def extract_data(self, soup, page_num):
        """Override this method for specific extraction logic."""
        # This is a template method - implement for specific sites
        return []
    
    def save_data(self, data, filename):
        """Save scraped data to file."""
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Saved {len(data)} records to {filename}")

# Example of respectful scraping practices
def ethical_scraping_example():
    """Demonstrate ethical scraping practices."""
    
    # 1. Check robots.txt
    def check_robots_txt(base_url):
        robots_url = urljoin(base_url, '/robots.txt')
        try:
            response = requests.get(robots_url)
            if response.status_code == 200:
                print("Robots.txt content:")
                print(response.text[:500])  # Show first 500 chars
        except:
            print("Could not retrieve robots.txt")
    
    # 2. Implement rate limiting
    class RateLimitedScraper:
        def __init__(self, requests_per_second=1):
            self.min_delay = 1.0 / requests_per_second
            self.last_request = 0
        
        def get(self, url, **kwargs):
            # Ensure minimum delay between requests
            elapsed = time.time() - self.last_request
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
            
            self.last_request = time.time()
            return requests.get(url, **kwargs)
    
    # 3. Handle errors gracefully
    def robust_scraper(urls):
        results = []
        failed_urls = []
        
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    results.append({'url': url, 'status': 'success'})
                else:
                    failed_urls.append({'url': url, 'status': response.status_code})
            except Exception as e:
                failed_urls.append({'url': url, 'error': str(e)})
        
        return results, failed_urls
    
    print("Ethical scraping guidelines:")
    print("1. Always check robots.txt")
    print("2. Implement rate limiting")
    print("3. Handle errors gracefully")
    print("4. Respect terms of service")
    print("5. Consider using official APIs when available")
```

#### A.2 Advanced Scraping with Selenium

For JavaScript-heavy sites:

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

def setup_driver(headless=True):
    """Set up Selenium Chrome driver."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # You need to download ChromeDriver and add to PATH
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def scrape_dynamic_content(url):
    """Scrape content that loads via JavaScript."""
    driver = setup_driver()
    
    try:
        driver.get(url)
        
        # Wait for specific element to load
        wait = WebDriverWait(driver, 10)
        element = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "content"))
        )
        
        # Extract data after JavaScript execution
        data = []
        elements = driver.find_elements(By.CLASS_NAME, "item")
        
        for elem in elements:
            item_data = {
                'text': elem.text,
                'href': elem.get_attribute('href')
            }
            data.append(item_data)
        
        return data
        
    finally:
        driver.quit()

# Example: Scraping social media posts (with proper authorization)
def scrape_social_media_posts():
    """
    Example structure for social media scraping.
    Note: Always use official APIs when available!
    """
    driver = setup_driver()
    
    try:
        # Login process (if authorized)
        driver.get("https://example-social-site.com/login")
        
        # Find login elements and authenticate
        # (Implementation depends on specific site)
        
        # Navigate to data source
        driver.get("https://example-social-site.com/feed")
        
        # Scroll to load more content
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        # Extract posts
        posts = driver.find_elements(By.CLASS_NAME, "post")
        post_data = []
        
        for post in posts:
            try:
                text = post.find_element(By.CLASS_NAME, "post-text").text
                timestamp = post.find_element(By.CLASS_NAME, "timestamp").text
                likes = post.find_element(By.CLASS_NAME, "likes").text
                
                post_data.append({
                    'text': text,
                    'timestamp': timestamp,
                    'likes': likes
                })
            except:
                continue
        
        return post_data
        
    finally:
        driver.quit()
```

### B. API Usage for LLM Integration

Modern AI applications heavily rely on API integrations.

#### B.1 Basic API Interactions

```python
import requests
import json
import time
from typing import Dict, List, Optional
import asyncio
import aiohttp

class APIClient:
    """Generic API client with common functionality."""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set default headers
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            })
    
    def get(self, endpoint: str, params: Dict = None) -> Dict:
        """Make GET request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"GET request failed: {e}")
            return {}
    
    def post(self, endpoint: str, data: Dict = None) -> Dict:
        """Make POST request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.post(url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"POST request failed: {e}")
            return {}

class OpenAIClient(APIClient):
    """OpenAI API client for LLM interactions."""
    
    def __init__(self, api_key: str):
        super().__init__("https://api.openai.com/v1", api_key)
    
    def chat_completion(self, messages: List[Dict], model: str = "gpt-3.5-turbo", **kwargs) -> str:
        """Get chat completion from OpenAI."""
        data = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        response = self.post("chat/completions", data)
        
        if response and 'choices' in response:
            return response['choices'][0]['message']['content']
        return ""
    
    def generate_embeddings(self, texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """Generate embeddings for texts."""
        data = {
            "model": model,
            "input": texts
        }
        
        response = self.post("embeddings", data)
        
        if response and 'data' in response:
            return [item['embedding'] for item in response['data']]
        return []

class AnthropicClient(APIClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str):
        super().__init__("https://api.anthropic.com/v1", api_key)
        self.session.headers.update({
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01'
        })
    
    def chat_completion(self, messages: List[Dict], model: str = "claude-3-sonnet-20240229", **kwargs) -> str:
        """Get chat completion from Claude."""
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 1000,
            **kwargs
        }
        
        response = self.post("messages", data)
        
        if response and 'content' in response:
            return response['content'][0]['text']
        return ""

# Example usage
def llm_api_examples():
    """Demonstrate LLM API usage."""
    
    # Note: Replace with actual API keys
    # openai_client = OpenAIClient("your-openai-api-key")
    # anthropic_client = AnthropicClient("your-anthropic-api-key")
    
    # Mock clients for demonstration
    class MockLLMClient:
        def chat_completion(self, messages, **kwargs):
            return f"Mock response to: {messages[-1]['content'][:50]}..."
        
        def generate_embeddings(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]
    
    mock_client = MockLLMClient()
    
    # Basic chat completion
    messages = [
        {"role": "system", "content": "You are a helpful data scientist."},
        {"role": "user", "content": "Explain the difference between supervised and unsupervised learning."}
    ]
    
    response = mock_client.chat_completion(messages)
    print(f"LLM Response: {response}")
    
    # Generate embeddings for text similarity
    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Python is a programming language"
    ]
    
    embeddings = mock_client.generate_embeddings(texts)
    print(f"Generated {len(embeddings)} embeddings")

# Advanced: Batch processing with rate limiting
class BatchAPIProcessor:
    """Process API requests in batches with rate limiting."""
    
    def __init__(self, client, requests_per_minute=60):
        self.client = client
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
    
    def process_batch(self, prompts: List[str], batch_size: int = 5) -> List[str]:
        """Process prompts in batches."""
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch:
                # Rate limiting
                elapsed = time.time() - self.last_request_time
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                
                # Make request
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat_completion(messages)
                batch_results.append(response)
                
                self.last_request_time = time.time()
            
            results.extend(batch_results)
            print(f"Processed batch {i//batch_size + 1}, total: {len(results)}")
        
        return results

# Async API client for high-throughput applications
class AsyncAPIClient:
    """Asynchronous API client for concurrent requests."""
    
    def __init__(self, base_url: str, api_key: str, max_concurrent: int = 10):
        self.base_url = base_url
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def async_post(self, session: aiohttp.ClientSession, endpoint: str, data: Dict) -> Dict:
        """Make async POST request."""
        async with self.semaphore:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            try:
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}"}
            except Exception as e:
                return {"error": str(e)}
    
    async def process_concurrent_requests(self, requests_data: List[Dict]) -> List[Dict]:
        """Process multiple requests concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.async_post(session, "chat/completions", data)
                for data in requests_data
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

# Example: Building a document analysis pipeline
def document_analysis_pipeline():
    """Example pipeline combining web scraping and LLM analysis."""
    
    # Step 1: Collect documents (web scraping)
    documents = [
        "This is a sample document about machine learning...",
        "Another document discussing data science trends...",
        "A technical paper on neural networks..."
    ]
    
    # Step 2: Analyze documents with LLM
    mock_client = MockLLMClient()
    
    analysis_prompts = [
        f"Summarize this document in 2 sentences: {doc}"
        for doc in documents
    ]
    
    # Step 3: Process with batch processor
    processor = BatchAPIProcessor(mock_client, requests_per_minute=30)
    summaries = processor.process_batch(analysis_prompts)
    
    # Step 4: Create analysis report
    analysis_df = pd.DataFrame({
        'document': documents,
        'summary': summaries,
        'length': [len(doc) for doc in documents]
    })
    
    print("Document Analysis Results:")
    print(analysis_df)
    
    return analysis_df

# API error handling and retry logic
class RobustAPIClient:
    """API client with comprehensive error handling."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def request_with_retry(self, method: str, endpoint: str, data: Dict = None, max_retries: int = 3) -> Dict:
        """Make request with exponential backoff retry."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(max_retries):
            try:
                if method.upper() == 'GET':
                    response = self.session.get(url, params=data, timeout=30)
                else:
                    response = self.session.post(url, json=data, timeout=30)
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    return {"error": "Request timeout"}
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif e.response.status_code >= 500:  # Server error
                    wait_time = 2 ** attempt
                    print(f"Server error. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return {"error": f"HTTP {e.response.status_code}"}
                
            except Exception as e:
                print(f"Unexpected error: {e}")
                if attempt == max_retries - 1:
                    return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}

# Example usage
print("🔌 API Integration Examples")
llm_api_examples()
analysis_results = document_analysis_pipeline()
```

### C. Advanced Pandas Operations

Master advanced data manipulation techniques.

#### C.1 Advanced Data Manipulation

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Create comprehensive sample dataset
def create_advanced_dataset():
    """Create a complex dataset for advanced pandas operations."""
    np.random.seed(42)
    
    # Generate time series data
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    
    # Generate hierarchical data
    regions = ['North', 'South', 'East', 'West']
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D']
    
    data = []
    for date in dates:
        for region in regions:
            for product in products:
                # Create realistic sales patterns
                base_sales = np.random.poisson(100)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                weekend_factor = 0.8 if date.weekday() >= 5 else 1.0
                
                sales = int(base_sales * seasonal_factor * weekend_factor)
                revenue = sales * np.random.uniform(10, 50)
                
                data.append({
                    'date': date,
                    'region': region,
                    'product': product,
                    'sales': sales,
                    'revenue': revenue,
                    'cost': revenue * np.random.uniform(0.6, 0.8),
                    'customer_count': np.random.poisson(20),
                    'promotion': np.random.choice([True, False], p=[0.2, 0.8])
                })
    
    df = pd.DataFrame(data)
    return df

# Advanced groupby operations
def advanced_groupby_operations(df):
    """Demonstrate advanced groupby techniques."""
    print("🔧 Advanced GroupBy Operations")
    
    # Multiple aggregation functions
    agg_dict = {
        'sales': ['sum', 'mean', 'std', 'count'],
        'revenue': ['sum', 'mean'],
        'cost': 'sum',
        'customer_count': 'mean'
    }
    
    monthly_summary = df.groupby([df['date'].dt.to_period('M'), 'region']).agg(agg_dict)
    print("Monthly summary with multiple aggregations:")
    print(monthly_summary.head())
    
    # Custom aggregation functions
    def revenue_per_customer(series):
        return series.sum() / len(series)
    
    def sales_volatility(series):
        return series.std() / series.mean() if series.mean() > 0 else 0
    
    custom_agg = df.groupby('region').agg({
        'revenue': revenue_per_customer,
        'sales': sales_volatility,
        'promotion': lambda x: (x == True).sum() / len(x)  # Promotion rate
    })
    
    print("\nCustom aggregations:")
    print(custom_agg)
    
    # Transform operations (group-wise operations)
    df['sales_rank_in_region'] = df.groupby(['region', 'product'])['sales'].rank(ascending=False)
    df['revenue_pct_of_region'] = df.groupby(['region', df['date'].dt.date])['revenue'].transform(
        lambda x: x / x.sum() * 100
    )
    
    # Rolling windows within groups
    df_sorted = df.sort_values(['region', 'product', 'date'])
    df_sorted['sales_7day_avg'] = df_sorted.groupby(['region', 'product'])['sales'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    print("\nTransform operations added:")
    print(df_sorted[['region', 'product', 'date', 'sales', 'sales_7day_avg']].head(10))
    
    return df_sorted

# Advanced time series operations
def advanced_time_series_operations(df):
    """Demonstrate advanced time series techniques."""
    print("📅 Advanced Time Series Operations")
    
    # Set date as index for time series operations
    df_ts = df.set_index('date').sort_index()
    
    # Resampling with custom functions
    daily_sales = df_ts.groupby(['region', 'product'])['sales'].resample('D').sum()
    weekly_stats = df_ts.groupby(['region', 'product'])['sales'].resample('W').agg({
        'total_sales': 'sum',
        'avg_sales': 'mean',
        'peak_sales': 'max'
    })
    
    print("Weekly resampling:")
    print(weekly_stats.head())
    
    # Time-based indexing and slicing
    q1_data = df_ts['2023-01-01':'2023-03-31']
    print(f"\nQ1 data shape: {q1_data.shape}")
    
    # Lead and lag operations
    df_lead_lag = df_ts.groupby(['region', 'product']).apply(
        lambda group: group.assign(
            sales_lag1=group['sales'].shift(1),
            sales_lead1=group['sales'].shift(-1),
            sales_lag7=group['sales'].shift(7)
        )
    ).reset_index(level=[0, 1], drop=True)
    
    # Calculate growth rates
    df_lead_lag['daily_growth'] = df_lead_lag.groupby(['region', 'product'])['sales'].pct_change()
    df_lead_lag['week_over_week'] = df_lead_lag.groupby(['region', 'product'])['sales'].pct_change(periods=7)
    
    print("\nLead/lag operations:")
    print(df_lead_lag[['region', 'product', 'sales', 'sales_lag1', 'daily_growth']].head())
    
    return df_lead_lag

# Advanced merging and joining
def advanced_merge_operations():
    """Demonstrate complex merge scenarios."""
    print("🔗 Advanced Merge Operations")
    
    # Create additional datasets
    product_info = pd.DataFrame({
        'product': ['Product_A', 'Product_B', 'Product_C', 'Product_D'],
        'category': ['Electronics', 'Clothing', 'Electronics', 'Home'],
        'launch_date': pd.to_datetime(['2022-01-01', '2022-06-01', '2023-01-01', '2022-03-01']),
        'price_tier': ['Premium', 'Budget', 'Premium', 'Mid']
    })
    
    region_info = pd.DataFrame({
        'region': ['North', 'South', 'East', 'West'],
        'manager': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'size': ['Large', 'Medium', 'Small', 'Large'],
        'timezone': ['EST', 'CST', 'EST', 'PST']
    })
    
    # Complex merges with multiple keys
    df_sample = create_advanced_dataset().head(100)
    
    # Merge with product information
    df_with_product = df_sample.merge(product_info, on='product', how='left')
    
    # Merge with region information
    df_complete = df_with_product.merge(region_info, on='region', how='left')
    
    print("Merged dataset:")
    print(df_complete[['date', 'region', 'product', 'sales', 'category', 'manager']].head())
    
    # Conditional merges using merge_asof (time-based)
    price_changes = pd.DataFrame({
        'product': ['Product_A', 'Product_A', 'Product_B', 'Product_B'],
        'date': pd.to_datetime(['2023-01-01', '2023-06-01', '2023-02-01', '2023-08-01']),
        'new_price': [45, 50, 25, 30]
    })
    
    # Find the most recent price for each sale
    df_sample_sorted = df_sample.sort_values('date')
    price_changes_sorted = price_changes.sort_values('date')
    
    df_with_prices = pd.merge_asof(
        df_sample_sorted,
        price_changes_sorted,
        on='date',
        by='product',
        direction='backward'
    )
    
    print("\nTime-based merge with prices:")
    print(df_with_prices[['date', 'product', 'sales', 'new_price']].head())
    
    return df_complete

# Advanced data cleaning and transformation
def advanced_data_cleaning():
    """Demonstrate advanced cleaning techniques."""
    print("🧹 Advanced Data Cleaning")
    
    # Create messy dataset
    messy_data = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C001', 'C003', None, 'C002'],
        'name': ['John Doe', 'jane smith', 'JOHN DOE', 'Bob Wilson', 'Alice Johnson', 'Jane Smith'],
        'email': ['john@email.com', 'jane@email.com', 'john@email.com', None, 'alice@email', 'jane@email.com'],
        'age': [25, '30', 25, 'unknown', 28, 30],
        'salary': ['$50,000', '60000', '$50,000', '$75,000', '55000', '60,000'],
        'join_date': ['2023-01-15', '2023/02/10', '2023-01-15', '2023-03-20', '2023-04-01', '2023-02-10']
    })
    
    print("Original messy data:")
    print(messy_data)
    
    # Clean and standardize
    df_clean = messy_data.copy()
    
    # Standardize text fields
    df_clean['name'] = df_clean['name'].str.title()
    
    # Clean salary field
    df_clean['salary'] = (df_clean['salary']
                         .str.replace(', '', regex=False)
                         .str.replace(',', '', regex=False)
                         .astype(float))
    
    # Clean age field
    df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
    
    # Standardize date format
    df_clean['join_date'] = pd.to_datetime(df_clean['join_date'], errors='coerce')
    
    # Validate email format
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
    df_clean['valid_email'] = df_clean['email'].apply(
        lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
    )
    
    # Identify duplicates (sophisticated approach)
    df_clean['name_normalized'] = df_clean['name'].str.lower().str.replace(' ', '')
    potential_duplicates = df_clean.groupby('name_normalized').size() > 1
    duplicate_groups = df_clean[df_clean['name_normalized'].isin(
        potential_duplicates[potential_duplicates].index
    )]
    
    print("\nCleaned data:")
    print(df_clean)
    
    print("\nPotential duplicates:")
    print(duplicate_groups)
    
    return df_clean

# Advanced pivot operations and cross-tabulation
def advanced_pivot_operations(df):
    """Demonstrate advanced pivot and crosstab operations."""
    print("📊 Advanced Pivot Operations")
    
    # Multi-level pivot tables
    pivot_complex = df.pivot_table(
        values=['sales', 'revenue'],
        index=['region', df['date'].dt.quarter],
        columns=['product'],
        aggfunc={
            'sales': ['sum', 'mean'],
            'revenue': 'sum'
        },
        fill_value=0,
        margins=True,
        margins_name='Total'
    )
    
    print("Complex pivot table:")
    print(pivot_complex.head())
    
    # Cross-tabulation with custom functions
    crosstab_promo = pd.crosstab(
        [df['region'], df['product']],
        df['promotion'],
        values=df['sales'],
        aggfunc='sum',
        normalize='index'  # Normalize by row
    )
    
    print("\nCross-tabulation of promotions:")
    print(crosstab_promo)
    
    # Melt operations for reshaping
    df_sample = df.head(100)
    df_melted = df_sample.melt(
        id_vars=['date', 'region', 'product'],
        value_vars=['sales', 'revenue', 'cost'],
        var_name='metric',
        value_name='amount'
    )
    
    print("\nMelted data:")
    print(df_melted.head())
    
    # Advanced unstacking
    df_unstacked = df.set_index(['date', 'region', 'product'])['sales'].unstack(['region', 'product'])
    print("\nUnstacked sales data:")
    print(df_unstacked.head())
    
    return pivot_complex, df_melted

# Performance optimization techniques
def pandas_performance_optimization():
    """Demonstrate pandas performance optimization."""
    print("⚡ Performance Optimization")
    
    # Create large dataset
    large_df = create_advanced_dataset()
    print(f"Dataset size: {large_df.shape}")
    
    # Memory usage optimization
    print("\nMemory usage before optimization:")
    print(large_df.info(memory_usage='deep'))
    
    # Optimize data types
    large_df_optimized = large_df.copy()
    
    # Convert to categorical for repeated strings
    large_df_optimized['region'] = large_df_optimized['region'].astype('category')
    large_df_optimized['product'] = large_df_optimized['product'].astype('category')
    
    # Downcast numeric types
    large_df_optimized['sales'] = pd.to_numeric(large_df_optimized['sales'], downcast='integer')
    large_df_optimized['customer_count'] = pd.to_numeric(large_df_optimized['customer_count'], downcast='integer')
    
    print("\nMemory usage after optimization:")
    print(large_df_optimized.info(memory_usage='deep'))
    
    # Vectorized operations vs loops
    import time
    
    # Slow approach (loops)
    start_time = time.time()
    profit_slow = []
    for _, row in large_df.head(1000).iterrows():
        profit = row['revenue'] - row['cost']
        profit_slow.append(profit)
    loop_time = time.time() - start_time
    
    # Fast approach (vectorized)
    start_time = time.time()
    profit_fast = large_df.head(1000)['revenue'] - large_df.head(1000)['cost']
    vectorized_time = time.time() - start_time
    
    print(f"\nLoop approach time: {loop_time:.4f}s")
    print(f"Vectorized approach time: {vectorized_time:.4f}s")
    print(f"Speedup: {loop_time/vectorized_time:.1f}x")
    
    # Efficient groupby with query
    start_time = time.time()
    result1 = large_df[(large_df['region'] == 'North') & (large_df['sales'] > 100)].groupby('product')['revenue'].sum()
    query_time = time.time() - start_time
    
    start_time = time.time()
    result2 = large_df.query("region == 'North' and sales > 100").groupby('product')['revenue'].sum()
    query_method_time = time.time() - start_time
    
    print(f"\nTraditional filtering time: {query_time:.4f}s")
    print(f"Query method time: {query_method_time:.4f}s")
    
    return large_df_optimized

# Example usage
def run_advanced_pandas_examples():
    """Run all advanced pandas examples."""
    print("🐼 Advanced Pandas Operations Examples")
    
    # Create main dataset
    df = create_advanced_dataset()
    print(f"Created dataset with {len(df)} rows")
    
    # Run examples
    df_transformed = advanced_groupby_operations(df)
    df_timeseries = advanced_time_series_operations(df)
    df_merged = advanced_merge_operations()
    df_cleaned = advanced_data_cleaning()
    pivot_results, melted_df = advanced_pivot_operations(df)
    optimized_df = pandas_performance_optimization()
    
    print("\n✅ All advanced pandas examples completed!")
    
    return {
        'original': df,
        'transformed': df_transformed,
        'timeseries': df_timeseries,
        'merged': df_merged,
        'cleaned': df_cleaned,
        'optimized': optimized_df
    }

# Run all examples
if __name__ == "__main__":
    results = run_advanced_pandas_examples()
```

These advanced topics provide powerful tools for:

- **Web Scraping**: Collecting real-world data for ML projects
- **API Integration**: Building modern AI applications with LLM services
- **Advanced Pandas**: Mastering complex data manipulation for sophisticated analysis

Each section includes practical examples and best practices that directly apply to the AI/ML internship program. Students can reference these topics as they progress through more advanced projects in weeks 4-8.