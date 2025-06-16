# LangChain Cheat Sheet

## 🚀 Quick Setup

```bash
# Install LangChain
pip install langchain langchain-community

# Install Ollama
# MacOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: Download from ollama.com

# Download a model
ollama pull llama3.2
```

## 🔗 LCEL (LangChain Expression Language) Basics

### Core Syntax
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# Basic chain composition
chain = prompt | model | output_parser
```

### Essential Components

#### 1. Prompts
```python
# Simple prompt
prompt = ChatPromptTemplate.from_template("Explain {topic} in simple terms")

# Multi-variable prompt
prompt = ChatPromptTemplate.from_template(
    "Analyze the {data_type} data and provide {analysis_type} insights"
)

# System + Human message
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful data analyst"),
    ("human", "Analyze this data: {data}")
])
```

#### 2. Models
```python
# Ollama local model
model = Ollama(model="llama3.2")

# With parameters
model = Ollama(
    model="llama3.2",
    temperature=0.7,
    num_predict=100
)
```

#### 3. Output Parsers
```python
# String output
output_parser = StrOutputParser()

# JSON output
from langchain_core.output_parsers import JsonOutputParser
json_parser = JsonOutputParser()

# Comma-separated list
from langchain_core.output_parsers import CommaSeparatedListOutputParser
list_parser = CommaSeparatedListOutputParser()
```

## 🔧 Common Chain Patterns

### Basic Analysis Chain
```python
analysis_prompt = ChatPromptTemplate.from_template(
    "Analyze this dataset summary: {summary}\n"
    "Provide 3 key insights in bullet points."
)

analysis_chain = analysis_prompt | model | StrOutputParser()

# Usage
result = analysis_chain.invoke({"summary": "Dataset has 1000 rows, 10 columns..."})
```

### Conditional Routing
```python
from langchain_core.runnables import RunnableBranch

def route_by_type(input_dict):
    data_type = input_dict.get("type", "general")
    if data_type == "numerical":
        return numerical_chain
    elif data_type == "text":
        return text_chain
    else:
        return general_chain

routing_chain = RunnableBranch(
    (lambda x: x["type"] == "numerical", numerical_chain),
    (lambda x: x["type"] == "text", text_chain),
    general_chain  # default
)
```

### Parallel Processing
```python
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "insights": insights_chain,
    "recommendations": recommendations_chain
})
```

## 📊 Metaflow + LangChain Integration

### Basic Integration Pattern
```python
from metaflow import FlowSpec, step
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

class DataAnalysisFlow(FlowSpec):
    
    @step
    def start(self):
        self.data = load_data()
        self.next(self.preprocess)
    
    @step
    def preprocess(self):
        # Traditional preprocessing
        self.processed_data = preprocess(self.data)
        self.next(self.analyze_with_llm)
    
    @step
    def analyze_with_llm(self):
        # LLM-powered analysis
        model = Ollama(model="llama3.2")
        prompt = ChatPromptTemplate.from_template(
            "Analyze this data summary: {summary}"
        )
        chain = prompt | model | StrOutputParser()
        
        summary = self.processed_data.describe().to_string()
        self.llm_insights = chain.invoke({"summary": summary})
        self.next(self.end)
    
    @step
    def end(self):
        print(f"Analysis complete: {self.llm_insights}")
```

## 🛠 Useful Utilities

### Check Ollama Status
```python
import subprocess

def check_ollama():
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama running")
            print("Available models:", result.stdout)
            return True
    except:
        print("❌ Ollama not found")
        return False
```

### Simple Chain Testing
```python
def test_chain(chain, test_inputs):
    """Quick chain testing utility"""
    for i, input_data in enumerate(test_inputs):
        try:
            result = chain.invoke(input_data)
            print(f"Test {i+1}: ✅ Success")
            print(f"Output: {result[:100]}...")
        except Exception as e:
            print(f"Test {i+1}: ❌ Failed - {e}")
```

## 🎯 Common Use Cases

### Data Summarization
```python
summarize_prompt = ChatPromptTemplate.from_template(
    "Summarize this dataset in 2-3 sentences:\n"
    "Columns: {columns}\n"
    "Shape: {shape}\n"
    "Sample data: {sample}"
)

summarize_chain = summarize_prompt | model | StrOutputParser()
```

### Feature Engineering Suggestions
```python
feature_prompt = ChatPromptTemplate.from_template(
    "Given this dataset description:\n{description}\n"
    "Target variable: {target}\n"
    "Suggest 3 new features to create. Be specific about the transformations."
)

feature_chain = feature_prompt | model | StrOutputParser()
```

### Model Interpretation
```python
interpret_prompt = ChatPromptTemplate.from_template(
    "Explain these model results in plain English:\n"
    "Model: {model_type}\n"
    "Accuracy: {accuracy}\n"
    "Top features: {features}\n"
    "What does this mean for business decisions?"
)

interpret_chain = interpret_prompt | model | StrOutputParser()
```

## 🚨 Troubleshooting

### Common Issues

**Ollama not responding:**
```bash
# Restart Ollama service
ollama serve

# Check if model is downloaded
ollama list

# Pull model if missing
ollama pull llama3.2
```

**Chain execution errors:**
```python
# Add error handling
try:
    result = chain.invoke(input_data)
except Exception as e:
    print(f"Chain failed: {e}")
    # Fallback logic here
```

**Memory issues with large models:**
```bash
# Use smaller models
ollama pull llama3.2:1b  # 1B parameter version

# Or adjust Ollama settings
export OLLAMA_NUM_PARALLEL=1
```

## 📚 Quick Reference

### Essential Imports
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch
```

### Key Operators
- `|` - Chain composition (pipe operator)
- `invoke()` - Execute chain with input
- `stream()` - Stream output token by token
- `batch()` - Process multiple inputs

### Model Parameters
```python
Ollama(
    model="llama3.2",
    temperature=0.0,    # Deterministic (0.0) to creative (1.0)
    num_predict=256,    # Max tokens to generate
    top_p=0.9,         # Nucleus sampling
    top_k=40,          # Top-k sampling
)
```

---

**💡 Pro Tips:**
- Always test chains with simple inputs first
- Use temperature=0.0 for consistent outputs in data analysis
- Combine traditional preprocessing with LLM insights for best results
- Keep prompts specific and include examples when possible