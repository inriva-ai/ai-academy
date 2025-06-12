# INRIVA AI 2025 Academy: 8-Week AI/ML & Generative AI Internship Program

## Program Overview

This intensive 8-week program introduces college students to artificial intelligence, machine learning, and generative AI through a structured blend of self-study, hands-on practice, mentorship, and real-world applications. Students will progress from fundamental concepts to developing production-ready AI/ML solutions using modern MLOps and LLMOps frameworks, with a manageable 20-hour weekly commitment.

### Program Goals

- Build foundational knowledge in traditional ML and generative AI theory and applications
- Gain practical experience with production-grade tools: Metaflow, LangChain, LangGraph, and Ollama
- Develop problem-solving skills through hands-on projects spanning both domains
- Learn from experienced professionals through targeted shadowing and mentorship
- Create a portfolio-worthy capstone project using industry-standard MLOps/LLMOps practices

### Core Technology Stack Throughout Program

- **Data & ML Pipelines:** [Metaflow](https://metaflow.org/) for scalable data science workflows
- **LLM Orchestration:** [LangChain](https://python.langchain.com/) with [LCEL](https://python.langchain.com/docs/expression_language/) for chain composition
- **Agent Systems:** [LangGraph](https://langchain-ai.github.io/langgraph/) for complex AI agent workflows
- **Local Models:** [Ollama](https://ollama.com/) for privacy-focused local LLM deployment
- **Commercial Models:** [OpenAI GPT](https://platform.openai.com/), [Google Gemini](https://ai.google.dev/), [Anthropic Claude](https://docs.anthropic.com/)

## Week 1: Foundations of AI/ML and MLOps with Metaflow

### Self-Study Materials (8-10 hours)

- **Core Reading:** ["Introduction to Statistical Learning"](https://www.statlearning.com/) - Chapter 1 (free PDF available)
- **Videos:** [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning) - Week 1 (selected videos)
- **MLOps Foundation:** [What is Metaflow](https://docs.metaflow.org/introduction/what-is-metaflow) and [Metaflow Tutorials](https://docs.metaflow.org/getting-started/tutorials)
- **Interactive:** [Kaggle Learn - Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)

### Hands-On Exercises (8-10 hours)

**Tutorial:** [Metaflow Quickstart](https://docs.metaflow.org/getting-started/tutorials/season-1-the-local-experience) and [Installation Guide](https://docs.metaflow.org/getting-started/install)

**Exercise 1: Environment Setup with Metaflow (3 hours)**
- Install [Metaflow](https://pypi.org/project/metaflow/) and dependencies
- Complete [Episode 0: Hello World](https://docs.metaflow.org/getting-started/tutorials/season-1-the-local-experience/episode00) tutorial
- Set up development environment (Python, Jupyter, Git)
- Run metaflow tutorials list to explore available examples

**Exercise 2: Your First ML Pipeline with Metaflow (3 hours)**
- Tutorial: [Episode 1: Playing with Data](https://docs.metaflow.org/getting-started/tutorials/season-1-the-local-experience/episode01)
- Implement basic linear regression using Metaflow steps
- Practice data flow between steps using artifacts
- Code along: [Linear Regression with Metaflow](https://github.com/Netflix/metaflow/tree/master/metaflow/tutorials)

**Exercise 3: Data Versioning and Reproducibility (2-3 hours)**
- Tutorial: [Episode 2: Statistics Redux](https://docs.metaflow.org/getting-started/tutorials/season-1-the-local-experience/episode02)
- Learn Metaflow's automatic versioning and artifact management
- Practice accessing previous run results
- Compare traditional ML approach vs. Metaflow-managed workflow

### Weekly Planning, Workshops, Office Hours (4.5 hours)

- Monday, 12:00 - 2:00 PM: Program overview, team introductions, tool setup, Q&A
- Wednesday, 1:00 - 2:30 PM: Hands-on tutorial for week's core concepts + peer collaboration
- Friday, 2:00 - 3:00 PM: Review week's work + open office hours for questions

### Deliverables

- Working Metaflow installation and first successful flow execution
- Linear regression pipeline implemented in Metaflow
- Comparison report: traditional ML development vs. MLOps with Metaflow

### Week 1 Knowledge Check Quiz

1. **Multiple Choice:** What is Metaflow's primary purpose?
   - A. Only for data visualization
   - B. Managing scalable data science workflows with versioning
   - C. Web development framework
   - D. Database management

2. **Short Answer:** Name three benefits of using Metaflow over traditional Python scripts for ML projects.

3. **Multiple Choice:** In Metaflow, what are artifacts?
   - A. Configuration files
   - B. Data objects that flow between steps and are automatically versioned
   - C. Error logs
   - D. User interfaces

4. **Code Completion:** Complete this Metaflow step signature: @step def _____(self):

5. **Multiple Choice:** What happens when you run a Metaflow flow?
   - A. It only runs locally
   - B. It automatically versions code, data, and execution environment
   - C. It deletes previous runs
   - D. It requires manual backup

6. **True/False:** Metaflow can automatically track and reproduce any previous run.

7. **Short Answer:** What is the difference between a step and a flow in Metaflow?

8. **Multiple Choice:** Which command lists all available Metaflow tutorials?
   - A. metaflow help
   - B. metaflow tutorials list
   - C. metaflow examples
   - D. metaflow --tutorials

9. **Short Answer:** Why is reproducibility important in machine learning projects?

10. **Multiple Choice:** Metaflow was originally developed by which company?
    - A. Google
    - B. Netflix
    - C. Facebook
    - D. Amazon

**Answer Key:** 1-B, 2-Automatic versioning, scalability, reproducibility, 3-B, 4-start/process_data/train_model/end, 5-B, 6-True, 7-Step is a single operation, flow is the complete workflow, 8-B, 9-Ensures consistent results and debuggability, 10-B

## Week 2: Data Preprocessing with Metaflow and Introduction to LangChain

### Self-Study Materials (8-10 hours)

- **Reading:** ["Python for Data Analysis"](https://wesmckinney.com/book/) - Chapters 5-6 (selected sections)
- **Metaflow:** [Working with Data](https://docs.metaflow.org/metaflow/data) and [Scaling Out](https://docs.metaflow.org/scaling/introduction)
- **LangChain Basics:** [Introduction to LangChain](https://python.langchain.com/docs/introduction/) and [LCEL Basics](https://python.langchain.com/docs/expression_language/)

### Hands-On Exercises (8-10 hours)

**Tutorial:** [Metaflow Episode 3: Playlist Paradise](https://docs.metaflow.org/getting-started/tutorials/season-1-the-local-experience/episode03) and [LangChain Quickstart](https://python.langchain.com/docs/tutorials/llm_chain/)

**Exercise 1: Data Pipeline with Metaflow (4 hours)**
- Dataset: [Titanic Dataset](https://www.kaggle.com/c/titanic)
- Tutorial: [Creating Flows](https://docs.metaflow.org/metaflow/basics)
- Build complete data preprocessing pipeline with Metaflow steps
- Handle missing values, feature engineering, and data validation
- Practice: [Data Processing Best Practices](https://docs.metaflow.org/metaflow/data)

**Exercise 2: Introduction to LangChain and LCEL (3 hours)**
- Install [LangChain](https://pypi.org/project/langchain/) and [Ollama](https://ollama.com/download)
- Tutorial: [LCEL Quickstart](https://python.langchain.com/docs/expression_language/get_started)
- Set up Ollama with a local model (e.g., llama3.2)
- Create your first LCEL chain: prompt | model | output_parser
- Practice: [Basic Chain Composition](https://python.langchain.com/docs/tutorials/llm_chain/)

**Exercise 3: Combining Metaflow with LangChain (3 hours)**
- Build a Metaflow pipeline that includes LLM-powered data analysis
- Use LangChain to generate data insights within Metaflow steps
- Compare traditional statistical analysis with LLM-generated insights
- Document integration patterns for MLOps + LLMOps

### Weekly Planning, Workshops, Office Hours (3.5 hours)

- Monday, 12:00 - 1:00 PM: Team standup, weekly planning
- Wednesday, 1:00 - 2:00 PM: Hands-on tutorial for week's core concepts + peer collaboration
- Wednesday, 2:00 - 2:30 PM: Guest speaker on Real-world ML deployment challenges and solutions
- Friday, 2:00 - 3:00 PM: Review week's work + open office hours for questions

### Deliverables

- Complete data preprocessing pipeline in Metaflow
- Working LangChain + Ollama setup with basic LCEL chains
- Hybrid pipeline combining Metaflow data processing with LangChain analysis

### Week 2 Knowledge Check Quiz

1. **Multiple Choice:** In LCEL, what does the | operator do?
   - A. Performs mathematical operations
   - B. Chains components together, passing output from left to right
   - C. Creates parallel execution
   - D. Handles errors

2. **Code Completion:** Complete this LCEL chain: prompt | _____ | output_parser

3. **Multiple Choice:** What is Ollama primarily used for?
   - A. Data visualization
   - B. Running large language models locally
   - C. Web scraping
   - D. Database management

4. **Short Answer:** Name two advantages of running LLMs locally with Ollama vs. using cloud APIs.

5. **Multiple Choice:** In Metaflow, which decorator allows parallel execution?
   - A. @step
   - B. @parallel
   - C. @foreach
   - D. Both B and C

6. **True/False:** LCEL chains automatically support streaming and async operations.

7. **Short Answer:** Why might you combine Metaflow with LangChain in a single pipeline?

8. **Code Question:** What Ollama command downloads a model to your local machine?

9. **Multiple Choice:** LCEL stands for:
   - A. LangChain Execution Language
   - B. LangChain Expression Language
   - C. Local Chain Execution Logic
   - D. Language Chain Extension Layer

10. **Multiple Choice:** Which is NOT a benefit of using Metaflow for data preprocessing?
    - A. Automatic versioning of data transformations
    - B. Easy scaling to cloud compute
    - C. Built-in web interface
    - D. Reproducible data lineage

**Answer Key:** 1-B, 2-model, 3-B, 4-Privacy/data control, cost efficiency, no API limits, 5-D, 6-True, 7-Combine structured ML workflows with LLM capabilities, 8-ollama pull model_name, 9-B, 10-C

## Week 3: Supervised Learning with Metaflow Pipelines

### Self-Study Materials (8-10 hours)

- **Core Reading:** ["Hands-On Machine Learning"](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) - Chapter 2 (key sections)
- **Metaflow:** [Machine Learning with Metaflow](https://docs.metaflow.org/metaflow/machine-learning) and [Episode 4: Matrix Math](https://docs.metaflow.org/getting-started/tutorials/season-1-the-local-experience/episode04)
- **LangChain:** [Model Integration](https://python.langchain.com/docs/integrations/llms/) and [Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)

### Hands-On Exercises (8-10 hours)

**Tutorial:** [Metaflow Episode 5: Modeling](https://docs.metaflow.org/getting-started/tutorials/season-1-the-local-experience/episode05) and [Scikit-learn Integration](https://docs.metaflow.org/api/step-decorators/resources)

**Exercise 1: ML Pipeline with Metaflow (4 hours)**
- Dataset: [Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- Build complete ML pipeline: data loading → preprocessing → model training → evaluation
- Implement multiple algorithms: Logistic Regression, Random Forest, XGBoost
- Practice: [Model Training with Metaflow](https://docs.metaflow.org/scaling/remote-tasks/introduction)
- Use @resources decorator for scaling model training

**Exercise 2: LangChain Model Comparison (2 hours)**
- Set up multiple Ollama models (e.g., llama3.2, mistral)
- Create LCEL chains for model comparison
- Build chain that routes inputs to different models based on task type
- Practice: [Model Selection with LangChain](https://python.langchain.com/docs/tutorials/llm_chain/)

**Exercise 3: Hybrid ML + LLM Evaluation (3 hours)**
- Use traditional ML metrics (accuracy, precision, recall)
- Add LLM-powered model explanation and interpretation
- Create Metaflow pipeline that generates both statistical and natural language reports
- Compare model performance using both quantitative and qualitative analysis

### Weekly Planning, Workshops, Office Hours (3.5 hours)

- Monday, 12:00 - 1:00 PM: Team standup, weekly planning
- Wednesday, 1:00 - 2:30 PM: Hands-on tutorial for week's core concepts + peer collaboration
- Friday, 2:00 - 3:00 PM: Review week's work + open office hours for questions

### Deliverables

- Complete supervised learning pipeline in Metaflow with multiple algorithms
- LangChain-based model comparison and routing system
- Hybrid evaluation report combining traditional metrics with LLM insights

### Week 3 Knowledge Check Quiz

1. **Multiple Choice:** In Metaflow, what does the @resources decorator do?
   - A. Manages memory usage
   - B. Specifies computational resources (CPU, memory) for a step
   - C. Handles data resources
   - D. Controls network resources

2. **Short Answer:** Name three classification algorithms you implemented this week.

3. **Multiple Choice:** Which LCEL component handles different output formats?
   - A. Prompt templates
   - B. Language models
   - C. Output parsers
   - D. Retrievers

4. **Code Question:** In Metaflow, how do you access artifacts from a previous step?

5. **Multiple Choice:** What is cross-validation used for?
   - A. Data cleaning
   - B. Model evaluation and preventing overfitting
   - C. Feature selection only
   - D. Data visualization

6. **True/False:** Metaflow automatically versions your trained models along with the code and data.

7. **Short Answer:** Why might you use LLMs to explain traditional ML model results?

8. **Multiple Choice:** In LangChain, what allows you to switch between different LLMs easily?
   - A. Hard-coded model names
   - B. Model abstraction through the unified interface
   - C. Separate installations
   - D. Manual configuration

9. **Code Completion:** Complete this Metaflow artifact access: self.model = _____.model_artifact

10. **Multiple Choice:** Which metric is best for imbalanced classification problems?
    - A. Accuracy only
    - B. F1-score or AUC-ROC
    - C. Mean squared error
    - D. R-squared

**Answer Key:** 1-B, 2-Logistic Regression, Random Forest, XGBoost, 3-C, 4-self.previous_step_name.artifact_name, 5-B, 6-True, 7-Better interpretability and human-readable explanations, 8-B, 9-self.prev_step, 10-B

## Week 4: Advanced ML with Metaflow and LangChain Integration

### Self-Study Materials (8-10 hours)

- **Reading:** [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) and [Metaflow Scaling](https://docs.metaflow.org/scaling/introduction)
- **LangChain:** [Chain Types](https://python.langchain.com/docs/modules/chains/) and [Memory](https://python.langchain.com/docs/modules/memory/)
- **Ollama:** [Model Library](https://ollama.com/library) and [API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Hands-On Exercises (8-10 hours)

**Tutorial:** [Metaflow Remote Execution](https://docs.metaflow.org/scaling/remote-tasks/introduction) and [LangChain Memory Management](https://python.langchain.com/docs/modules/memory/how_to/buffer)

**Exercise 1: Scalable Ensemble Pipeline (3 hours)**
- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Build ensemble methods pipeline with Metaflow
- Use @parallel decorator for concurrent model training
- Practice: [Parallel Execution](https://docs.metaflow.org/metaflow/basics#parallel-steps)
- Compare performance of individual models vs. ensemble

**Exercise 2: Advanced LangChain Workflows (3 hours)**
- Tutorial: [Chain Composition Patterns](https://python.langchain.com/docs/expression_language/how_to/routing)
- Build multi-step LCEL chains with conditional routing
- Implement conversation memory for context retention
- Create chains that combine multiple Ollama models for different tasks

**Exercise 3: Integrated ML + LLM Pipeline (3 hours)**
- Build Metaflow pipeline that trains ensemble models
- Add LangChain-powered result interpretation and reporting
- Create automated model comparison with natural language summaries
- Implement LLM-powered feature importance explanation

**Exercise 4: Text Classification with Transformers (2 hours)**
- Use local models via Ollama for text classification
- Compare transformer-based approaches with traditional ML
- Build LCEL chains for text preprocessing and classification
- Integrate results into Metaflow pipeline for evaluation

### Weekly Planning, Workshops, Office Hours (4.5 hours)

- Monday, 12:00 - 1:00 PM: Team standup, weekly planning
- Wednesday, 1:00 - 2:00 PM: Hands-on tutorial for week's core concepts + peer collaboration
- Wednesday, 2:00 - 2:30 PM: Guest speaker on latest developments in generative AI and future outlook
- Thursday, 3:00 - 5:00 PM: Mid-program review, progress evaluation, capstone project planning

### Deliverables

- Scalable ensemble learning pipeline using Metaflow parallelization
- Advanced LangChain workflows with memory and routing
- Integrated system combining ensemble ML with LLM interpretation

### Week 4 Knowledge Check Quiz

1. **Multiple Choice:** What is the main advantage of ensemble methods?
   - A. Faster training
   - B. Combining multiple models often improves performance and reduces overfitting
   - C. Less memory usage
   - D. Simpler interpretation

2. **Short Answer:** Explain the difference between bagging and boosting in ensemble methods.

3. **Multiple Choice:** In LangChain, what is memory used for?
   - A. Storing model weights
   - B. Maintaining conversation context across interactions
   - C. Caching API responses
   - D. Managing system resources

4. **True/False:** Metaflow's @parallel decorator allows multiple steps to execute simultaneously.

5. **Multiple Choice:** Which is NOT a benefit of using local LLMs with Ollama?
   - A. Data privacy
   - B. No API rate limits
   - C. Always faster than cloud APIs
   - D. Cost control

6. **Code Question:** How do you specify parallel execution in Metaflow for multiple models?

7. **Short Answer:** Why might you combine traditional ML ensembles with LLM explanations?

8. **Multiple Choice:** In LCEL, conditional routing allows:
   - A. Physical network routing
   - B. Directing inputs to different processing chains based on content
   - C. Load balancing only
   - D. Error handling

9. **True/False:** Ollama can run multiple models simultaneously for different tasks.

10. **Multiple Choice:** What is the main trade-off when using transformer models for text classification?
    - A. Accuracy vs Speed
    - B. Memory vs Disk space
    - C. Performance vs Computational requirements
    - D. Simplicity vs Complexity

**Answer Key:** 1-B, 2-Bagging trains models in parallel on different subsets, Boosting trains sequentially where each corrects previous errors, 3-B, 4-True, 5-C, 6-@parallel decorator or @foreach, 7-Combines quantitative performance with human-interpretable insights, 8-B, 9-True, 10-C

## Week 5: Unsupervised Learning and Introduction to LangGraph

### Self-Study Materials (8-10 hours)

- **Reading:** [Clustering Algorithms](https://scikit-learn.org/stable/modules/clustering.html) and [Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html)
- **LangGraph:** [Introduction to LangGraph](https://langchain-ai.github.io/langgraph/) and [Core Concepts](https://langchain-ai.github.io/langgraph/concepts/)
- **Tutorials:** [LangGraph Basics](https://langchain-ai.github.io/langgraph/tutorials/introduction/) and [LangChain Academy](https://academy.langchain.com/)

### Hands-On Exercises (8-10 hours)

**Tutorial:** [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/) and [Metaflow Episode 6: Deployment](https://docs.metaflow.org/getting-started/tutorials/season-1-the-local-experience/episode06)

**Exercise 1: Clustering Pipeline with Metaflow (3 hours)**
- Dataset: [Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Build unsupervised learning pipeline with K-means, DBSCAN, hierarchical clustering
- Use Metaflow for experiment tracking and parameter sweeps
- Practice: [Hyperparameter Tuning](https://docs.metaflow.org/metaflow/basics#foreach)

**Exercise 2: Your First LangGraph Agent (3 hours)**
- Install [LangGraph](https://pypi.org/project/langgraph/)
- Tutorial: [Build a Basic Agent](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- Create simple agent that uses tools for analysis
- Connect agent to local Ollama models
- Practice: [State Management](https://langchain-ai.github.io/langgraph/concepts/low_level/)

**Exercise 3: LangGraph for Data Analysis (3 hours)**
- Build agent that performs exploratory data analysis
- Create multi-step workflow: data loading → analysis → visualization → insights
- Use LangGraph's state management for complex workflows
- Integrate with Metaflow artifacts for data access

**Exercise 4: Hybrid Clustering + LLM Analysis (2 hours)**
- Use traditional clustering algorithms to segment data
- Deploy LangGraph agent to interpret and explain clusters
- Generate natural language descriptions of customer segments
- Create actionable business recommendations using LLM insights

### Weekly Planning, Workshops, Office Hours (3.5 hours)

- Monday, 12:00 - 1:00 PM: Team standup, weekly planning
- Wednesday, 1:00 - 2:30 PM: Hands-on tutorial for week's core concepts + peer collaboration
- Friday, 2:00 - 3:00 PM: Review week's work + open office hours for questions

### Deliverables

- Complete unsupervised learning pipeline with multiple clustering algorithms
- Working LangGraph agent for data analysis tasks
- Hybrid system combining clustering with LLM-powered interpretation

### Week 5 Knowledge Check Quiz

1. **Multiple Choice:** What is LangGraph primarily designed for?
   - A. Data visualization
   - B. Building stateful, multi-step AI agent workflows
   - C. Database management
   - D. Web development

2. **Short Answer:** Name three clustering algorithms and their key differences.

3. **Multiple Choice:** In LangGraph, what is a StateGraph used for?
   - A. Storing model weights
   - B. Defining agent conversation flows and state management
   - C. Visualizing data
   - D. Managing API keys

4. **True/False:** K-means clustering requires you to specify the number of clusters in advance.

5. **Multiple Choice:** What is the main advantage of using LangGraph over simple LangChain chains?
   - A. Faster execution
   - B. Support for complex, stateful workflows with conditional logic
   - C. Better model performance
   - D. Easier installation

6. **Code Question:** In LangGraph, what class is used to create a graph workflow?

7. **Short Answer:** What is the "elbow method" used for in clustering?

8. **Multiple Choice:** LangGraph nodes can:
   - A. Only call language models
   - B. Execute any Python function or tool
   - C. Only process text
   - D. Only work with APIs

9. **True/False:** LangGraph maintains state across different nodes in a workflow.

10. **Multiple Choice:** Which technique is best for visualizing high-dimensional data in 2D?
    - A. Bar charts
    - B. Line plots
    - C. t-SNE or PCA
    - D. Histograms

**Answer Key:** 1-B, 2-K-means (centroid-based), Hierarchical (tree-based), DBSCAN (density-based), 3-B, 4-True, 5-B, 6-StateGraph, 7-Determining optimal number of clusters, 8-B, 9-True, 10-C

## Week 6: Deep Learning with Metaflow and Advanced LangGraph Agents

### Self-Study Materials (8-10 hours)

- **Reading:** [Deep Learning Basics](https://www.deeplearningbook.org/) - neural network fundamentals (free online)
- **Metaflow:** [GPU Computing](https://docs.metaflow.org/scaling/remote-tasks/gpu) and [Metaflow Cards](https://docs.metaflow.org/api/cards)
- **LangGraph:** [Multi-Agent Systems](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/) and [Tool Use](https://langchain-ai.github.io/langgraph/tutorials/tool-calling/)

### Hands-On Exercises (8-10 hours)

**Tutorial:** [Neural Networks with Metaflow](https://github.com/Netflix/metaflow/tree/master/metaflow/tutorials) and [LangGraph Multi-Agent Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/)

**Exercise 1: Deep Learning Pipeline with Metaflow (3 hours)**
- Dataset: [CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10) or [Fashion-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
- Build CNN training pipeline using Metaflow
- Use @resources(gpu=1) for GPU acceleration
- Practice: [Model Checkpointing](https://docs.metaflow.org/api/step-decorators/catch) and [Metaflow Cards](https://docs.metaflow.org/api/cards) for visualization

**Exercise 2: Multi-Agent LangGraph System (3 hours)**
- Build multi-agent system with specialized roles
- Create agents for: data analysis, model training supervision, result interpretation
- Practice: [Agent Communication](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- Integrate agents with Ollama models for different capabilities

**Exercise 3: LangGraph Tools and Function Calling (2 hours)**
- Create custom tools for ML pipeline interaction
- Build agent that can query Metaflow run results
- Practice: [Custom Tools](https://langchain-ai.github.io/langgraph/tutorials/tool-calling/) and API integration
- Enable agents to trigger new Metaflow runs

**Exercise 4: Hybrid Deep Learning + Agent System (3 hours)**
- Combine CNN training pipeline with LangGraph monitoring agents
- Create agent-based experiment management system
- Build automated hyperparameter tuning with agent oversight
- Generate automated training reports using multi-agent workflows

### Weekly Planning, Workshops, Office Hours (3.5 hours)

- Monday, 12:00 - 1:00 PM: Team standup, weekly planning
- Wednesday, 1:00 - 2:00 PM: Hands-on tutorial for week's core concepts + peer collaboration
- Wednesday, 2:00 - 2:30 PM: Guest speaker on responsible AI development and bias mitigation
- Friday, 2:00 - 3:00 PM: Review week's work + open office hours for questions

### Deliverables

- Deep learning pipeline with GPU acceleration using Metaflow
- Multi-agent system for ML pipeline management
- Integrated system combining deep learning training with agent-based oversight

### Week 6 Knowledge Check Quiz

1. **Multiple Choice:** In Metaflow, how do you specify GPU usage for a step?
   - A. @gpu
   - B. @resources(gpu=1)
   - C. @compute(gpu=True)
   - D. @accelerate

2. **Short Answer:** Explain what backpropagation does in neural network training.

3. **Multiple Choice:** In LangGraph, what enables agents to communicate?
   - A. Direct function calls
   - B. Shared state and message passing
   - C. Database connections
   - D. File systems

4. **True/False:** Metaflow Cards provide automatic visualization of training metrics.

5. **Multiple Choice:** What is a key advantage of multi-agent systems?
   - A. Faster execution
   - B. Specialized agents can handle different aspects of complex tasks
   - C. Less memory usage
   - D. Simpler code

6. **Code Question:** How do you create a custom tool in LangGraph?

7. **Short Answer:** What is the main advantage of using CNNs for image classification?

8. **Multiple Choice:** LangGraph tools allow agents to:
   - A. Only process text
   - B. Interact with external systems and APIs
   - C. Only call other agents
   - D. Only access databases

9. **True/False:** Metaflow can automatically save model checkpoints during training.

10. **Multiple Choice:** What is the vanishing gradient problem in deep learning?
    - A. Gradients become too large
    - B. Gradients become too small in deep networks, hindering learning
    - C. Gradients change randomly
    - D. Gradients are calculated incorrectly

**Answer Key:** 1-B, 2-Backpropagation calculates gradients and updates weights to minimize loss, 3-B, 4-True, 5-B, 6-Define function with @tool decorator, 7-Automatic feature extraction and spatial hierarchy learning, 8-B, 9-True, 10-B

## Week 7: Production MLOps/LLMOps and Advanced Agent Systems

### Self-Study Materials (8-10 hours)

- **Reading:** [Metaflow Production](https://docs.metaflow.org/production/introduction) and [LangGraph Platform](https://www.langchain.com/langgraph-platform)
- **LangGraph:** [Agent Architectures](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/) and [Streaming](https://langchain-ai.github.io/langgraph/concepts/streaming/)
- **Articles:** [MLOps Best Practices](https://ml-ops.org/) and [LLMOps Principles](https://blog.langchain.dev/what-is-llmops/)

### Hands-On Exercises (8-10 hours)

**Tutorial:** [Metaflow Deployment](https://docs.metaflow.org/production/scheduling-metaflow-flows/introduction) and [LangGraph Human-in-the-Loop](https://langchain-ai.github.io/langgraph/tutorials/human-in-the-loop/)

**Exercise 1: Production Metaflow Pipeline (3 hours)**
- Deploy ML pipeline to production environment
- Set up automated training schedules
- Practice: [Flow Scheduling](https://docs.metaflow.org/production/scheduling-metaflow-flows/introduction)
- Implement monitoring and alerting for pipeline health

**Exercise 2: Advanced Agent Architectures (3 hours)**
- Build hierarchical agent systems with supervisor patterns
- Implement human-in-the-loop workflows
- Practice: [ReAct Agent Pattern](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
- Create agents with memory and learning capabilities

**Exercise 3: Model Deployment with Agent Monitoring (2 hours)**
- Deploy trained models using Metaflow
- Create LangGraph agents for model monitoring and maintenance
- Implement automated model performance tracking
- Build agent-based alerting system

**Exercise 4: End-to-End MLOps + LLMOps Pipeline (3 hours)**
- Combine all previous learnings into production system
- Build pipeline: data ingestion → training → deployment → monitoring
- Add agent-based oversight and optimization
- Practice: Error handling, rollback strategies, A/B testing

### Weekly Planning, Workshops, Office Hours (3.5 hours)

- Monday, 12:00 - 1:00 PM: Team standup, weekly planning
- Wednesday, 1:00 - 2:30 PM: Hands-on tutorial for week's core concepts + peer collaboration
- Friday, 2:00 - 3:00 PM: Review week's work + open office hours for questions

### Deliverables

- Production-ready ML pipeline with automated deployment
- Advanced agent system with human-in-the-loop capabilities
- Complete MLOps + LLMOps system with monitoring and optimization

### Week 7 Knowledge Check Quiz

1. **Multiple Choice:** What is the main difference between MLOps and LLMOps?
   - A. No difference
   - B. LLMOps specifically addresses challenges of large language models
   - C. MLOps is older
   - D. LLMOps is simpler

2. **Short Answer:** Name three key considerations when deploying ML models to production.

3. **Multiple Choice:** In LangGraph, what is human-in-the-loop?
   - A. Manual code writing
   - B. Allowing human intervention and approval in agent workflows
   - C. Human-only execution
   - D. GUI development

4. **True/False:** Metaflow can automatically schedule and run ML pipelines in production.

5. **Multiple Choice:** What is a ReAct agent pattern?
   - A. Reactive programming
   - B. Reasoning and Acting - agents that reason about actions before taking them
   - C. Real-time agents
   - D. Rapid application development

6. **Code Question:** How do you schedule a Metaflow flow to run daily?

7. **Short Answer:** Why is monitoring important in production ML systems?

8. **Multiple Choice:** LangGraph streaming allows:
   - A. Video streaming
   - B. Real-time visibility into agent reasoning and intermediate results
   - C. Data streaming only
   - D. Audio processing

9. **True/False:** Agents can modify their own behavior based on feedback and performance.

10. **Multiple Choice:** What is A/B testing in the context of ML deployment?
    - A. Testing two algorithms
    - B. Comparing performance of different model versions with real traffic
    - C. Alphabetical testing
    - D. Binary testing

**Answer Key:** 1-B, 2-Monitoring, scaling, versioning, rollback strategies, performance, 3-B, 4-True, 5-B, 6-Using production schedulers like Argo Workflows or cron, 7-Detect performance degradation, ensure reliability, optimize resource usage, 8-B, 9-True, 10-B

## Week 8: Production-Ready AI/ML Capstone Project

### Capstone Project (12-15 hours)

**Objective:** Build a production-ready AI/ML solution using the complete technology stack learned throughout the program.

**Required Technology Integration:**
- **Metaflow:** Complete ML pipeline with versioning, scaling, and deployment
- **LangChain + LCEL:** LLM integration and chain composition
- **LangGraph:** Multi-agent system for complex workflow orchestration
- **Ollama + Commercial APIs:** Hybrid local/cloud model deployment strategy

### Phase 1: AI-Assisted Solution Design (3-4 hours)

**Step 1: Problem Definition and Architecture (2 hours)**
- Choose from project categories below or propose custom solution
- Use LLMs to refine requirements and technical specifications
- Design system architecture integrating all four core technologies
- Create data flow diagrams showing Metaflow → LangChain → LangGraph interactions

**Step 2: Technology Integration Planning (1-2 hours)**
- Plan Metaflow pipeline structure and data flow
- Design LangGraph agent workflows and responsibilities
- Choose appropriate models (local Ollama vs. commercial APIs)
- Create integration strategy and deployment plan

### Phase 2: Implementation Using Full Stack (6-8 hours)

**Component A: Metaflow ML Pipeline (2-3 hours)**

```python
# Production-Ready Metaflow Pipeline Template
from metaflow import FlowSpec, step, Parameter, card, resources
from datetime import datetime

class AICapstoneFlow(FlowSpec):
    """
    Production AI pipeline integrating ML with LLM capabilities
    """
    
    model_type = Parameter('model-type', default='hybrid')
    deployment_env = Parameter('env', default='production')
    
    @step
    def start(self):
        """Initialize pipeline with data validation"""
        print(f"Starting production pipeline: {datetime.now()}")
        self.next(self.data_processing)
    
    @step
    @resources(memory=4096)
    def data_processing(self):
        """Data ingestion and preprocessing with quality checks"""
        # Implement robust data processing
        self.next(self.feature_engineering, self.llm_data_enrichment)
    
    @step
    @resources(cpu=4)
    def feature_engineering(self):
        """Traditional ML feature engineering"""
        # Your ML feature creation logic
        self.next(self.model_training)
    
    @step
    @resources(gpu=1, memory=8192)
    def model_training(self):
        """Scalable model training with monitoring"""
        # Your model training with automatic versioning
        self.next(self.model_evaluation)
    
    @step
    def llm_data_enrichment(self):
        """LLM-powered data analysis and augmentation"""
        # Connect to your LangChain/LangGraph workflows
        self.next(self.model_evaluation)
    
    @step
    @card(type='html')
    def model_evaluation(self, inputs):
        """Comprehensive evaluation with automated reporting"""
        self.merge_artifacts(inputs)
        # Generate evaluation reports and visualizations
        self.next(self.deployment_prep)
    
    @step
    def deployment_prep(self):
        """Production deployment preparation"""
        # Model packaging and API creation
        self.next(self.end)
    
    @step
    def end(self):
        """Pipeline completion with monitoring setup"""
        print("Production pipeline completed successfully")

if __name__ == '__main__':
    AICapstoneFlow()
```

**Component B: LangGraph Agent System (2-3 hours)**

```python
# Advanced LangGraph Multi-Agent System
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    messages: List[str]
    ml_results: dict
    agent_decisions: List[str]
    final_output: str
    human_feedback: str

def ml_supervisor_agent(state: AgentState):
    """Agent that oversees ML pipeline execution"""
    # Monitor Metaflow pipeline status
    # Make decisions about model deployment
    return {"agent_decisions": ["ml_pipeline_approved"]}

def data_analyst_agent(state: AgentState):
    """Agent specialized in data analysis and insights"""
    # Analyze ML results using LLM capabilities
    # Generate business insights
    return {"messages": state["messages"]}

def deployment_agent(state: AgentState):
    """Agent responsible for model deployment decisions"""
    # Evaluate model readiness for production
    # Handle deployment logistics
    return {"agent_decisions": state["agent_decisions"] + ["ready_for_deploy"]}

def human_oversight_agent(state: AgentState):
    """Human-in-the-loop decision point"""
    # Request human approval for critical decisions
    # Incorporate human feedback into workflow
    return {"human_feedback": "approved"}

# Build the multi-agent workflow
workflow = StateGraph(AgentState)
workflow.add_node("ml_supervisor", ml_supervisor_agent)
workflow.add_node("data_analyst", data_analyst_agent)
workflow.add_node("deployment_agent", deployment_agent)
workflow.add_node("human_oversight", human_oversight_agent)

# Define agent communication flow
workflow.set_entry_point("ml_supervisor")
workflow.add_edge("ml_supervisor", "data_analyst")
workflow.add_edge("data_analyst", "deployment_agent")
workflow.add_edge("deployment_agent", "human_oversight")
workflow.add_edge("human_oversight", END)

app = workflow.compile()
```

**Component C: LangChain Integration Layer (2 hours)**

```python
# LangChain LCEL chains for model integration
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

# Local model setup
local_llm = Ollama(model="llama3.2")

# Analysis chain
analysis_prompt = ChatPromptTemplate.from_template(
    "Analyze the ML model results: {results}. Provide insights and recommendations."
)
analysis_chain = analysis_prompt | local_llm | StrOutputParser()

# Report generation chain
report_prompt = ChatPromptTemplate.from_template(
    "Generate executive summary for ML project: {analysis}. Include business impact."
)
report_chain = report_prompt | local_llm | StrOutputParser()

# Combined workflow
full_pipeline = RunnableParallel({
    "analysis": analysis_chain,
    "detailed_report": analysis_chain | report_chain
})
```

### Phase 3: Integration & Production Deployment (2-3 hours)

**Deployment Architecture:**
- FastAPI service integrating all components
- Docker containerization for consistent deployment
- Monitoring and logging for production observability
- Automated testing and validation workflows

### Project Categories

**Choose Your Focus Area:**

1. **Intelligent Document Processing Platform**
   - Metaflow: Document ingestion and classification pipeline
   - LangChain: Text extraction and summarization chains
   - LangGraph: Multi-agent document routing and processing
   - Integration: Automated document workflow with human oversight

2. **Smart Manufacturing Quality Control**
   - Metaflow: Image processing and defect detection pipeline
   - LangChain: Defect explanation and reporting
   - LangGraph: Quality control agent system with escalation
   - Integration: Real-time quality monitoring with agent oversight

3. **Financial Risk Assessment System**
   - Metaflow: Risk modeling and feature engineering pipeline
   - LangChain: Risk report generation and explanation
   - LangGraph: Multi-agent risk analysis with compliance checking
   - Integration: Automated risk assessment with regulatory oversight

4. **Healthcare Diagnosis Support**
   - Metaflow: Medical data processing and model training
   - LangChain: Medical literature synthesis and explanation
   - LangGraph: Multi-specialist agent consultation system
   - Integration: Clinical decision support with human validation

5. **Content Intelligence Platform**
   - Metaflow: Content analysis and recommendation pipeline
   - LangChain: Content generation and optimization
   - LangGraph: Multi-agent content workflow management
   - Integration: Automated content creation with quality control

6. **Customer Service Automation**
   - Metaflow: Customer behavior analysis and prediction
   - LangChain: Response generation and personalization
   - LangGraph: Multi-agent customer service workflow
   - Integration: Intelligent routing with human escalation

### Weekly Planning, Workshops, Office Hours (2.5 hours)

- Monday, 12:00 - 1:00 PM: Team standup, weekly planning
- Wednesday, 1:00 - 2:00 PM: Hands-on tutorial for week's core concepts + peer collaboration
- Wednesday, 2:00 - 2:30 PM: Guest speaker on career insights from hiring professionals in AI

### Final Presentations (3 hours)

- 20-minute comprehensive demonstration of production system
- Technical architecture deep-dive covering all four technologies
- Live deployment demonstration with real-time monitoring
- Q&A covering design decisions, scaling strategies, and lessons learned
- Business impact assessment and ROI analysis
- Peer review and knowledge sharing session

### Deliverables

- Production-ready system integrating Metaflow, LangChain, LangGraph, and Ollama
- Complete technical documentation covering architecture and deployment
- Deployment guide with monitoring and maintenance procedures
- Business case analysis with performance metrics and scalability plan
- Comprehensive presentation showcasing technical depth and practical application

### Week 8 Knowledge Check Quiz (Capstone Assessment)

1. **Multiple Choice:** What makes this capstone different from traditional ML projects?
   - A. Uses more data
   - B. Integrates production-grade MLOps and LLMOps technologies
   - C. Focuses only on accuracy
   - D. Uses only open source tools

2. **Short Answer:** Explain how Metaflow, LangChain, and LangGraph work together in your system.

3. **Multiple Choice:** Why use both Ollama and commercial APIs in the same system?
   - A. Redundancy only
   - B. Cost optimization and task-specific model selection
   - C. To complicate the architecture
   - D. No specific reason

4. **Code Question:** How do you trigger a LangGraph workflow from within a Metaflow step?

5. **Multiple Choice:** What is the primary benefit of multi-agent systems in production?
   - A. Faster execution
   - B. Specialized agents handle different aspects with coordination and oversight
   - C. Simpler code
   - D. Lower cost

6. **Short Answer:** Describe a scenario where human-in-the-loop is essential in your system.

7. **Multiple Choice:** In production AI systems, what is most critical?
   - A. Model accuracy only
   - B. Reliability, monitoring, and maintainability
   - C. Development speed
   - D. Using latest models

8. **True/False:** Your capstone system can automatically scale and handle production workloads.

9. **Short Answer:** Name three production considerations you implemented in your system.

10. **Multiple Choice:** What skill is most important for modern AI/ML practitioners?
    - A. Mathematical expertise only
    - B. Integration of multiple technologies for production-ready solutions
    - C. Programming speed
    - D. Memorizing algorithms

**Answer Key:** 1-B, 2-Metaflow handles ML pipelines and data flow, LangChain manages LLM interactions, LangGraph orchestrates multi-agent workflows, 3-B, 4-Using Python API calls within Metaflow steps, 5-B, 6-Critical decisions, compliance checks, safety validation, 7-B, 8-True, 9-Monitoring, scaling, error handling, versioning, security, etc., 10-B

## Assessment and Evaluation

### Weekly Assessments (40%)

- Technical exercises and coding assignments using core technologies
- Progressive skill building with Metaflow, LangChain, LangGraph, Ollama integration
- Weekly knowledge check quizzes focusing on practical application

### Mid-Program Project (25%)

- Week 4 project demonstrating ensemble ML pipeline with LangChain integration
- Technical presentation comparing traditional and modern MLOps approaches
- Code quality review emphasizing production readiness

### Capstone Project (35%)

- Technical implementation using complete technology stack
- Production deployment and monitoring capabilities
- Innovation in combining MLOps and LLMOps practices
- Business impact and practical applicability

## Resources and Tools

### Required Software

- [Python 3.8+](https://www.python.org/downloads/) with [Anaconda distribution](https://www.anaconda.com/products/distribution)
- [Metaflow](https://pypi.org/project/metaflow/) for ML pipeline management
- [LangChain](https://pypi.org/project/langchain/) and [LangGraph](https://pypi.org/project/langgraph/) for LLM workflows
- [Ollama](https://ollama.com/download) for local model deployment
- [Jupyter Notebook/Lab](https://jupyter.org/install)
- [Git](https://git-scm.com/downloads) and [GitHub account](https://github.com/)
- [Docker](https://www.docker.com/get-started) for containerization

### Core Technology Documentation

- **Metaflow:** [Documentation](https://docs.metaflow.org/) | [GitHub](https://github.com/Netflix/metaflow) | [Tutorials](https://docs.metaflow.org/getting-started/tutorials)
- **LangChain:** [Documentation](https://python.langchain.com/) | [LCEL Guide](https://python.langchain.com/docs/expression_language/) | [Tutorials](https://python.langchain.com/docs/tutorials/)
- **LangGraph:** [Documentation](https://langchain-ai.github.io/langgraph/) | [Tutorials](https://langchain-ai.github.io/langgraph/tutorials/) | [Examples](https://langchain-ai.github.io/langgraph/concepts/)
- **Ollama:** [Documentation](https://ollama.com/) | [Model Library](https://ollama.com/library) | [API Docs](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Supporting Libraries

- **Data & ML:** [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
- **Deep Learning:** [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/)
- **Visualization:** [matplotlib](https://matplotlib.org/), [plotly](https://plotly.com/python/)
- **Deployment:** [FastAPI](https://fastapi.tiangolo.com/), [Streamlit](https://streamlit.io/)

### Essential Reading

1. [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
2. [Building LLM Applications for Production](https://huyenchip.com/2023/04/11/llm-engineering.html)
3. [Metaflow Documentation](https://docs.metaflow.org/)
4. [LangChain Academy Course](https://academy.langchain.com/)

### Online Learning Resources

- [Metaflow Tutorials](https://docs.metaflow.org/getting-started/tutorials) - Complete tutorial series
- [LangChain Academy](https://academy.langchain.com/) - Free structured course on LangGraph
- [Ollama Blog](https://ollama.com/blog) - Latest updates and tutorials
- [MLOps Community](https://ml-ops.org/) - Best practices and case studies

## Mentorship and Support Structure

### Weekly One-on-One Meetings (30 minutes)

- Progress review on technology integration
- Technical problem-solving for complex workflows
- Career guidance in MLOps and LLMOps fields

### Weekly Group Sessions (3.5 hours)

- Weekly standup and planning
- Technology deep-dives and best practices
- Guest speakers from industry
- Peer project showcases and collaborative learning

### Online Community Support

- Google Chat for immediate technical support
- Code review sessions and pair programming
- Integration troubleshooting and optimization tips

## Success Metrics and Outcomes

### Technical Skills Developed

- Production-grade MLOps with Metaflow for scalable ML pipelines
- Advanced LLMOps with LangChain and LangGraph for agent systems
- Local model deployment and management with Ollama
- Integration patterns for hybrid cloud/on-premise AI systems
- End-to-end system design for production AI applications

### Professional Skills Gained

- Modern AI system architecture and design patterns
- Production deployment and monitoring strategies
- Cross-functional collaboration between ML and LLM teams
- Technical communication for complex integrated systems
- Problem-solving with cutting-edge technology stacks

### Career Preparation

- Portfolio demonstrating mastery of current industry-standard tools
- Experience with technology stack used by leading AI companies
- Understanding of production AI system development lifecycle
- Network connections in MLOps and LLMOps communities
- Preparation for senior-level AI engineering roles

### Post-Program Opportunities

- Senior AI/ML Engineer positions at tech companies
- MLOps/LLMOps specialist roles in enterprise organizations
- Startup opportunities building next-generation AI systems
- Open source contributions to Metaflow, LangChain ecosystem
- Speaking opportunities at AI/MLOps conferences and communities

The program graduates will be among the first cohort of practitioners skilled in modern production AI system development, positioning them at the forefront of the rapidly evolving AI infrastructure landscape.

---

## Getting Started Instructions

### New Students
1. **Complete environment setup** using `/setup/` instructions
2. **Join communication channels** (links provided via email)
3. **Review Week 1 materials** in `/weeks/week1/`
4. **Attend Monday kickoff** session

### Returning Students
1. **Check for updates**: `git pull origin main`
2. **Review current week** materials  
3. **Complete any pending** exercises
4. **Prepare questions** for upcoming sessions

---

**Welcome to the INRIVA AI Academy! Let's build the future of AI together! 🚀**

*Questions? Start with `/docs/program-overview/` or reach out via Google Chat.*

---

*© 2025 INRIVA AI Academy. This program content is designed for educational purposes and hands-on learning in modern AI/ML development.*