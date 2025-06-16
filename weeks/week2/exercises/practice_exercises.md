# Week 2 Exercises: Data Preprocessing and LangChain Integration

Welcome to the Week 2 practice exercises! These challenges will help you master advanced data preprocessing with Metaflow and LangChain fundamentals.

## 📋 Exercise Overview

### Difficulty Levels
- 🟢 **Beginner**: Basic concepts and guided practice
- 🟡 **Intermediate**: Applied skills with some complexity
- 🔴 **Advanced**: Creative problem-solving and integration

### Time Estimates
- Each exercise designed for 15-30 minutes
- Solutions available in `/solutions/` directory
- Feel free to experiment and extend beyond requirements!

---

## 🟢 Exercise 1: Enhanced Data Preprocessing Pipeline

**Objective**: Extend the basic preprocessing pipeline with additional features and validation.

### Background
The workshop pipeline handles basic preprocessing well, but real-world data often requires more sophisticated techniques. Your task is to build upon the existing pipeline with advanced features.

### Tasks

#### 1.1 Advanced Missing Value Strategy (10 minutes)
```python
# Extend the AdvancedPreprocessingFlow class
class EnhancedPreprocessingFlow(AdvancedPreprocessingFlow):
    
    missing_threshold = Parameter('missing_threshold',
                                 help='Drop columns with >X% missing',
                                 default=0.5)
    
    @step
    def advanced_missing_handling(self):
        """
        Implement sophisticated missing value strategies:
        - Drop columns with >missing_threshold missing values
        - Use KNN imputation for numerical variables
        - Use decision tree imputation for mixed data types
        """
        # Your implementation here
        pass
```

#### 1.2 Feature Interaction Creation (10 minutes)
Create meaningful feature interactions:
- Age × Class interactions
- Family size × Fare per person
- Gender × Class survival patterns
- Port of embarkation × Class patterns

#### 1.3 Advanced Validation (5 minutes)
Add data validation checks for:
- Feature correlation analysis (>0.95 correlation warning)
- Class imbalance detection
- Outlier percentage limits
- Feature importance pre-screening

### Deliverables
- [ ] Extended preprocessing flow with new features
- [ ] Validation report highlighting potential issues
- [ ] Feature importance analysis output
- [ ] Documentation of design decisions

---

## 🟡 Exercise 2: Multi-Model LangChain Comparison

**Objective**: Create LCEL chains that compare different LLM models and route tasks appropriately.

### Background
Different LLM models have different strengths. This exercise builds a system that can intelligently route different types of analysis to appropriate models.

### Tasks

#### 2.1 Model Setup (5 minutes)
Set up multiple Ollama models:
```bash
# Download different models for comparison
ollama pull llama3.2
ollama pull mistral
ollama pull codellama  # If you want to try code analysis
```

#### 2.2 Comparative Analysis Chain (15 minutes)
```python
class MultiModelAnalysisFlow(FlowSpec):
    """
    Compare analysis quality across different LLM models
    """
    
    models = Parameter('models',
                      help='Comma-separated list of models to compare',
                      default='llama3.2,mistral')
    
    @step
    def compare_models(self):
        """
        Run the same analysis task across multiple models and compare:
        - Response quality
        - Analysis depth
        - Processing time
        - Consistency of insights
        """
        # Your implementation here
        pass
```

#### 2.3 Intelligent Routing (10 minutes)
Create a routing chain that:
- Analyzes the input type (statistical, narrative, technical)
- Routes to the most appropriate model
- Combines results when beneficial
- Handles model failures gracefully

### Advanced Challenge
Build a meta-analysis chain that:
- Takes outputs from multiple models
- Identifies common themes and discrepancies
- Generates a consensus analysis
- Highlights areas of model disagreement

### Deliverables
- [ ] Multi-model comparison system
- [ ] Routing logic for different analysis types
- [ ] Performance benchmarking results
- [ ] Meta-analysis pipeline (bonus)

---

## 🔴 Exercise 3: Production-Ready Hybrid Pipeline

**Objective**: Build a production-ready pipeline that seamlessly integrates Metaflow and LangChain with monitoring, error handling, and scalability.

### Background
Moving from prototype to production requires robust error handling, monitoring, and scalability considerations. This exercise focuses on building enterprise-ready pipelines.

### Tasks

#### 3.1 Advanced Error Handling (10 minutes)
```python
class ProductionHybridFlow(FlowSpec):
    """
    Production-ready hybrid pipeline with comprehensive error handling
    """
    
    @catch(var='preprocessing_errors')
    @step
    def robust_preprocessing(self):
        """
        Implement robust preprocessing with:
        - Data validation at each step
        - Automatic fallback strategies
        - Error logging and recovery
        - Data lineage tracking
        """
        pass
    
    @catch(var='llm_errors')
    @step  
    def fault_tolerant_llm_analysis(self):
        """
        Implement fault-tolerant LLM analysis:
        - Multiple model fallbacks
        - Timeout handling
        - Quality validation of outputs
        - Graceful degradation to statistical methods
        """
        pass
```

#### 3.2 Pipeline Monitoring (15 minutes)
Add comprehensive monitoring:
- Data quality scores at each step
- Processing time tracking
- LLM response quality assessment
- Resource utilization monitoring
- Alert system for data drift or quality issues

#### 3.3 Scalability Features (10 minutes)
Implement scalability improvements:
- Batch processing for large datasets
- Parallel LLM inference where possible
- Caching mechanisms for repeated analyses
- Memory-efficient data handling

#### 3.4 Configuration Management (5 minutes)
Create flexible configuration system:
- Environment-specific settings
- A/B testing capabilities for different models
- Feature flags for experimental features
- Dynamic parameter adjustment

### Advanced Challenges

#### Challenge A: Real-time Pipeline
Convert the batch pipeline to handle streaming data:
- Incremental preprocessing
- Real-time LLM analysis
- Online model performance tracking

#### Challenge B: Multi-dataset Integration
Extend to handle multiple data sources:
- Schema alignment across datasets
- Cross-dataset analysis and insights
- Federated learning considerations

#### Challenge C: Explainable AI Integration
Add explainability features:
- Feature importance explanations
- Decision boundary visualization
- LLM reasoning chain extraction
- Bias detection and mitigation

### Deliverables
- [ ] Production-ready hybrid pipeline
- [ ] Comprehensive monitoring dashboard
- [ ] Error handling and recovery documentation
- [ ] Performance benchmarking report
- [ ] At least one advanced challenge completed

---

## 🎯 Bonus Exercises

### Bonus 1: Custom Output Parser Development
Create sophisticated output parsers for:
- Extracting structured data from LLM responses
- Handling multi-format outputs (JSON, tables, narratives)
- Quality assessment and validation
- Error correction and standardization

### Bonus 2: LangChain Memory Integration
Implement conversation memory in your analysis chains:
- Context retention across analysis steps
- User preference learning
- Historical analysis comparison
- Personalized insight generation

### Bonus 3: Integration Testing Suite
Build comprehensive testing for your hybrid pipeline:
- Unit tests for individual components
- Integration tests for workflow orchestration
- Load testing for scalability validation
- Data quality regression testing

---

## 📚 Resources and Hints

### Metaflow Tips
```python
# Use @resources for scaling heavy operations
@resources(memory=8000, cpu=4)
@step
def heavy_preprocessing(self):
    pass

# Use @parallel for independent operations
@parallel
@step  
def parallel_analysis(self):
    pass

# Store large artifacts efficiently
self.large_dataset = S3(self.df)  # For cloud storage
```

### LangChain Tips
```python
# Batch processing for efficiency
chain.batch([input1, input2, input3])

# Streaming for real-time applications
for chunk in chain.stream(input_data):
    print(chunk)

# Error handling in chains
chain.with_fallbacks([backup_chain])

# Custom retry logic
chain.with_retry(stop_after_attempt=3)
```

### Debugging Tips
1. **Start Simple**: Get basic functionality working first
2. **Log Everything**: Add comprehensive logging at each step
3. **Test Incrementally**: Test each component in isolation
4. **Use Mock Data**: Create simple test cases for validation
5. **Monitor Resources**: Watch memory and CPU usage

---

## 🏆 Success Criteria

### For Each Exercise:
- [ ] **Functionality**: Core requirements implemented correctly
- [ ] **Code Quality**: Clean, readable, well-documented code
- [ ] **Error Handling**: Robust error handling and recovery
- [ ] **Testing**: Basic validation and edge case handling
- [ ] **Documentation**: Clear explanation of approach and decisions

### Bonus Points For:
- Creative extensions beyond requirements
- Performance optimizations
- Production-ready features
- Comprehensive testing
- Clear documentation and examples

---

## 🎯 Getting Started

1. **Choose Your Exercise**: Pick based on your comfort level and interests
2. **Set Up Environment**: Ensure all required packages are installed
3. **Review Workshop Code**: Use the workshop examples as starting points
4. **Start Coding**: Begin with the basic requirements
5. **Test Frequently**: Validate your work at each step
6. **Experiment**: Try variations and improvements
7. **Document**: Explain your approach and learnings

### Need Help?
- Check the workshop notebook for patterns and examples
- Review the `/solutions/` directory for guidance
- Ask questions in Google Chat or Discord
- Use Friday's office hours for detailed help

---

## 🎓 Learning Objectives Review

By completing these exercises, you should be able to:
- ✅ Build sophisticated data preprocessing pipelines
- ✅ Create complex LCEL chains with error handling
- ✅ Integrate multiple LLM models effectively
- ✅ Design production-ready hybrid workflows
- ✅ Implement monitoring and quality assessment
- ✅ Handle edge cases and failure scenarios gracefully

**Ready to put your Week 2 skills to the test? Choose an exercise and start building! 🚀**