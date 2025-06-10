# Week 1 Practice Exercises

## Exercise 1: Modify the Wine Pipeline

**Objective**: Customize the wine classification pipeline with different configurations.

### Tasks:
1. **Parameter Experimentation**
   - Run the pipeline with `test_size=0.3`
   - Run with different `random_state` values (42, 123, 999)
   - Compare how results change

2. **Add a New Model**
   - Add a Decision Tree classifier to the comparison
   - Hint: Use `from sklearn.tree import DecisionTreeClassifier`

3. **Feature Analysis**
   - Modify the flow to print the top 10 most important features
   - Create a visualization of feature importance

### Expected Output:
- Multiple runs with different parameters
- Comparison of at least 4 models
- Feature importance analysis

---

## Exercise 2: Create Your Own Dataset Flow

**Objective**: Build a complete pipeline for a different dataset.

### Tasks:
1. **Choose a Dataset**
   - Use `load_iris()`, `load_breast_cancer()`, or `load_digits()`
   - Or create synthetic data with `make_classification()`

2. **Build the Pipeline**
   - Follow the wine classification structure
   - Include data exploration, preprocessing, and modeling
   - Compare at least 3 different algorithms

3. **Add Custom Analysis**
   - Include correlation analysis
   - Add visualization steps
   - Create a detailed performance report

### Bonus Challenges:
- Add cross-validation
- Include hyperparameter tuning
- Create custom visualizations

---

## Exercise 3: Advanced Metaflow Features

**Objective**: Explore more advanced Metaflow capabilities.

### Tasks:
1. **Error Handling**
   - Add `@catch` decorator to handle potential errors
   - Create a step that might fail and handle it gracefully

2. **Parallel Processing**
   - Use `@foreach` to train multiple models in parallel
   - Compare execution time with sequential processing

3. **Custom Artifacts**
   - Save trained models as artifacts
   - Store visualizations and reports
   - Access artifacts from previous runs

### Code Template:
```python
@catch(var='training_error')
@step
def train_models(self):
    try:
        # Your training code
        pass
    except Exception as e:
        print(f"Training failed: {e}")
        self.training_error = str(e)
        self.models = {}
    
    self.next(self.evaluate)
```

---

## Exercise 4: Data Exploration Challenge

**Objective**: Master pandas data exploration techniques.

### Tasks:
1. **Load and Explore**
   ```python
   from sklearn.datasets import load_boston
   boston = load_boston()
   df = pd.DataFrame(boston.data, columns=boston.feature_names)
   df['target'] = boston.target
   ```

2. **Comprehensive Analysis**
   - Calculate missing value percentages
   - Find highly correlated features (|correlation| > 0.7)
   - Identify outliers using IQR method
   - Create distribution plots for all numeric features

3. **Advanced Insights**
   - Calculate feature importance using correlation
   - Group analysis by categorical features
   - Time series analysis if date columns exist

### Deliverable:
Create a Jupyter notebook with professional data exploration that includes:
- Executive summary
- Data quality assessment  
- Key insights and recommendations
- Visualization dashboard

---

## Exercise 5: Visualization Mastery

**Objective**: Create publication-ready visualizations.

### Tasks:
1. **Recreate Workshop Plots**
   - Reproduce the 6-panel visualization from the workshop
   - Use your own dataset instead of wine data
   - Ensure professional styling

2. **Interactive Elements**
   - Add hover information to plots
   - Create multi-layer visualizations
   - Implement color mapping for additional dimensions

3. **Custom Dashboard**
   - Combine multiple plot types in one figure
   - Add statistical annotations
   - Include summary tables

### Requirements:
- Use consistent color scheme
- Professional typography and layout
- Clear axis labels and legends
- Statistical overlays where appropriate

---

## Exercise 6: Production Pipeline Extension

**Objective**: Extend the complete ML pipeline with production features.

### Tasks:
1. **Model Persistence**
   - Save trained models using pickle
   - Implement model loading functionality
   - Create model versioning system

2. **Automated Reporting**
   - Generate PDF reports from pipeline results
   - Include executive summaries
   - Add visualizations to reports

3. **Configuration Management**
   - Create configuration files for different environments
   - Implement parameter validation
   - Add logging throughout the pipeline

### Code Structure:
```python
class ProductionMLPipeline(FlowSpec):
    config_file = Parameter('config', default='config.json')
    
    @step
    def load_config(self):
        # Load configuration from file
        pass
    
    @step
    def validate_inputs(self):
        # Validate all parameters and data
        pass
    
    @step
    def save_models(self):
        # Save models with versioning
        pass
    
    @step
    def generate_report(self):
        # Create comprehensive PDF report
        pass
```

---

## Solutions Available

Check the `/solutions/` directory for complete solutions to all exercises.

## Submission Guidelines

### For Each Exercise:
1. **Create a new branch**:
   ```bash
   git checkout -b exercise-X-[your-name]
   ```

2. **Document your work**:
   - Add comments to your code
   - Include README with approach
   - Note any challenges faced

3. **Submit for review**:
   ```bash
   git add .
   git commit -m "Exercise X: [description]"
   git push origin exercise-X-[your-name]
   ```

### Evaluation Criteria:
- **Functionality** (40%): Does the code work correctly?
- **Code Quality** (25%): Is the code clean and well-documented?
- **Completeness** (20%): Are all requirements met?
- **Innovation** (15%): Any creative additions or improvements?

## Need Help?

### Resources:
- **Workshop Materials**: Review notebooks and flows
- **Documentation**: Check package documentation
- **Community**: Ask questions in Google Chat

### Office Hours:
- **When**: Friday 2:00-3:00 PM
- **Where**: Google Meet (link in calendar)
- **What**: One-on-one help with exercises

### Common Issues:
1. **Environment Problems**: Run `setup_test.py` first
2. **Import Errors**: Check package versions
3. **Data Issues**: Verify dataset loading
4. **Metaflow Issues**: Check flow syntax and parameters

## Bonus Challenges

For advanced participants:

### Challenge 1: Real-World Dataset
- Find a dataset from Kaggle or UCI ML Repository
- Build complete pipeline from scratch
- Address real data quality issues

### Challenge 2: Ensemble Methods
- Implement voting classifier
- Create stacked ensemble
- Compare with individual models

### Challenge 3: Hyperparameter Optimization
- Add GridSearchCV to pipeline
- Implement Bayesian optimization
- Compare optimization strategies

### Challenge 4: Model Interpretation
- Add SHAP values for model explanation
- Create feature interaction plots
- Generate interpretation reports

Remember: The goal is learning, not perfection. Focus on understanding the concepts and building practical skills!