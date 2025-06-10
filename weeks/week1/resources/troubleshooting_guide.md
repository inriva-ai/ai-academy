# Troubleshooting Guide

## Environment Setup Issues

### Conda Environment Problems

**Issue**: `conda: command not found`
```bash
# Solution: Install Anaconda or Miniconda
# Download from: https://www.anaconda.com/products/distribution
# Or install Miniconda: https://docs.conda.io/en/latest/miniconda.html
```

**Issue**: Environment creation fails
```bash
# Try updating conda first
conda update conda

# Create environment with explicit python version
conda create -n aiml-workshop python=3.9

# If still failing, use pip instead
python -m venv aiml-workshop
source aiml-workshop/bin/activate  # Linux/Mac
# or
aiml-workshop\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Package Installation Issues

**Issue**: `ModuleNotFoundError: No module named 'metaflow'`
```bash
# Make sure environment is activated
conda activate aiml-workshop

# Reinstall metaflow
pip install --upgrade metaflow

# If on Apple Silicon Mac
conda install -c conda-forge metaflow
```

**Issue**: Scikit-learn version conflicts
```bash
# Uninstall and reinstall
pip uninstall scikit-learn
pip install scikit-learn>=1.1.0

# Or use conda
conda install scikit-learn=1.1.0
```

## Jupyter Notebook Issues

### Jupyter Won't Start

**Issue**: `jupyter: command not found`
```bash
# Install jupyter
pip install jupyter

# Or try jupyterlab
pip install jupyterlab
jupyter lab
```

**Issue**: Jupyter starts but kernel won't connect
```bash
# Install ipykernel
pip install ipykernel

# Add environment to jupyter
python -m ipykernel install --user --name=aiml-workshop

# Start jupyter and select the correct kernel
```

**Issue**: Port already in use
```bash
# Use different port
jupyter notebook --port=8889

# Or find and kill process using port 8888
lsof -i :8888  # Find process ID
kill -9 <PID>  # Kill the process
```

## Metaflow Issues

### Metaflow Flow Execution Problems

**Issue**: `ImportError` when running flows
```bash
# Make sure all dependencies are installed
pip install pandas numpy scikit-learn matplotlib seaborn

# Check if running in correct environment
which python
# Should point to your conda environment
```

**Issue**: Flow runs but artifacts not accessible
```bash
# Check Metaflow configuration
python -c "from metaflow import get_metadata; print(get_metadata())"

# List recent runs
python flow_name.py list

# Show specific run details
python flow_name.py show <run_id>
```

**Issue**: Permission denied errors
```bash
# Check if .metaflow directory is writable
ls -la ~/.metaflow

# If needed, fix permissions
chmod -R 755 ~/.metaflow
```

## Data Science Library Issues

### Matplotlib Display Problems

**Issue**: Plots not showing
```python
# Add this at the beginning of notebook
%matplotlib inline

# Or for interactive plots
%matplotlib widget

# For better quality
%config InlineBackend.figure_format = 'retina'
```

**Issue**: Font warnings
```python
# Clear matplotlib cache
import matplotlib
matplotlib.font_manager._rebuild()

# Or ignore warnings
import warnings
warnings.filterwarnings('ignore')
```

### Pandas Performance Issues

**Issue**: Slow data loading
```python
# Specify dtypes when loading
dtypes = {'column1': 'int32', 'column2': 'category'}
df = pd.read_csv('file.csv', dtype=dtypes)

# Load only needed columns
df = pd.read_csv('file.csv', usecols=['col1', 'col2'])
```

**Issue**: Memory errors with large datasets
```python
# Read in chunks
chunk_iter = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in chunk_iter:
    # Process chunk
    pass

# Use more efficient dtypes
df = df.astype({'int_col': 'int32', 'float_col': 'float32'})
```

## Common Python Errors

### Import Errors

**Issue**: `ModuleNotFoundError`
```python
# Check if package is installed
pip list | grep package_name

# Install missing package
pip install package_name

# Check if correct environment is active
import sys
print(sys.executable)
```

### Memory Errors

**Issue**: `MemoryError` during model training
```python
# Use fewer samples for testing
X_sample = X[:1000]
y_sample = y[:1000]

# Or use models with lower memory footprint
from sklearn.linear_model import SGDClassifier
model = SGDClassifier()  # Uses less memory than SVM
```

### File Path Issues

**Issue**: `FileNotFoundError`
```python
# Check current directory
import os
print(os.getcwd())

# List files in directory
os.listdir('.')

# Use absolute paths
import os
file_path = os.path.abspath('data/file.csv')
```

## Workshop-Specific Issues

### Wine Dataset Problems

**Issue**: Dataset not loading
```python
# Try alternative loading
from sklearn.datasets import load_wine
try:
    wine = load_wine()
    print("✅ Dataset loaded successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    # Download manually if needed
```

### Visualization Issues

**Issue**: Plots look different than expected
```python
# Reset matplotlib style
import matplotlib.pyplot as plt
plt.style.use('default')

# Set figure size
plt.rcParams['figure.figsize'] = (10, 6)

# For consistent colors
import seaborn as sns
sns.set_palette("husl")
```

## Getting Help

### Check Your Setup
```bash
# Run the setup test script
python setup/setup_test.py

# Check all package versions
pip list
```

### Debugging Steps
1. **Read the error message carefully**
2. **Check if environment is activated**
3. **Verify package versions**
4. **Try minimal reproduction**
5. **Search for specific error messages**

### Where to Get Help
- **Workshop**: Google Chat #urgent-help
- **Stack Overflow**: Search for specific errors
- **Package Documentation**: Official docs for libraries
- **GitHub Issues**: Check package repositories

### Emergency Backup Plan
If nothing works, you can:
1. **Use Google Colab**: Upload notebooks to Colab
2. **Use Binder**: Run notebooks in browser
3. **Pair with classmate**: Share screen and work together
4. **Follow along**: Watch demonstration and review materials later