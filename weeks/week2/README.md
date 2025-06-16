# Week 2: Data Preprocessing and LangChain Introduction

Welcome to Week 2 of the INRIVA AI Academy! This week introduces advanced data preprocessing with Metaflow and your first steps into the world of Large Language Models with LangChain.

## 🎯 Learning Objectives

By the end of this week, you will:
- ✅ Master advanced data preprocessing techniques in Metaflow pipelines
- ✅ Understand LangChain fundamentals and LCEL (LangChain Expression Language)
- ✅ Set up and work with local LLMs using Ollama
- ✅ Build your first hybrid ML + LLM workflows
- ✅ Integrate text processing and feature engineering at scale

## 🧠 Core Concepts

### Data Preprocessing Mastery
- **Feature Engineering**: Creating meaningful features from raw data
- **Data Validation**: Ensuring data quality throughout pipelines
- **Scaling Techniques**: Handling large datasets with Metaflow
- **Text Processing**: Advanced text cleaning and tokenization

### LangChain Fundamentals
- **LCEL**: LangChain Expression Language for chain composition
- **Prompt Engineering**: Crafting effective prompts for LLMs
- **Local LLMs**: Privacy-focused AI with Ollama
- **Chain Patterns**: Building reusable LLM workflows

## 📚 Prerequisites

### From Week 1
- [x] Metaflow environment working
- [x] Basic pipeline creation skills
- [x] Data exploration fundamentals
- [x] Python and pandas proficiency

### New This Week
- [x] Install LangChain: `pip install langchain`
- [x] Install Ollama: [ollama.com/download](https://ollama.com/download)
- [x] Download a local model: `ollama pull llama3.2`

## 🔧 Quick Setup Check

Run this verification before starting:

```python
# Verify environment
import pandas as pd
import numpy as np
from metaflow import FlowSpec
import langchain
import subprocess

# Check Ollama installation
try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    print("✅ Ollama installed and running")
    print("Available models:", result.stdout)
except:
    print("❌ Ollama not found - install from ollama.com")

print("🎯 Environment ready for Week 2!")
```

## 📁 Week Structure

### 🔥 Workshop Session (`/workshop/`)
- **Step 1**: Advanced Metaflow preprocessing patterns
- **Step 2**: LangChain and LCEL fundamentals  
- **Step 3**: Local LLM setup with Ollama
- **Step 4**: Building hybrid ML + LLM pipelines
- **Step 5**: Text processing and feature engineering

### 📓 Notebooks (`/notebooks/`)
- **week2_workshop.ipynb** - Interactive workshop notebook
- **langchain_intro.ipynb** - LangChain fundamentals
- **data_pipeline_advanced.ipynb** - Advanced preprocessing techniques

### 🌊 Flows (`/flows/`)
- **preprocessing_flow.py** - Complete data preprocessing pipeline
- **text_analysis_flow.py** - Text processing with LLM integration
- **hybrid_pipeline.py** - Combined ML + LLM workflow

### 📊 Data (`/data/`)
- **titanic.csv** - Primary dataset for preprocessing exercises
- **customer_reviews.csv** - Text data for LangChain integration
- **financial_data.json** - Complex structured data for advanced processing

### 🎯 Exercises (`/exercises/`)
- **preprocessing_challenges.md** - Data preprocessing practice
- **langchain_basics.md** - LangChain fundamentals
- **integration_exercises.md** - Hybrid workflow challenges

### 💡 Solutions (`/solutions/`)
- **completed_workshop.ipynb** - Full workshop solutions
- **exercise_solutions.py** - Exercise answers with explanations

### 📚 Resources (`/resources/`)
- **langchain_cheatsheet.md** - Quick LangChain reference
- **ollama_setup.md** - Detailed Ollama installation guide
- **data_preprocessing_guide.md** - Advanced preprocessing techniques

## 🚀 Workshop Progression

### Part 1: Advanced Data Preprocessing (45 minutes)
1. **Missing Data Strategies** - Advanced imputation techniques
2. **Feature Engineering** - Creating predictive features
3. **Scaling and Validation** - Pipeline robustness
4. **Text Data Handling** - NLP preprocessing fundamentals

### Part 2: LangChain Introduction (30 minutes)
1. **Installation and Setup** - LangChain and Ollama
2. **First LCEL Chain** - prompt | model | output_parser
3. **Local LLM Integration** - Working with Ollama models
4. **Chain Composition** - Building complex workflows

### Part 3: Integration Workshop (15 minutes)
1. **Hybrid Pipelines** - Combining Metaflow + LangChain
2. **Text Analysis** - LLM-powered data insights
3. **Production Patterns** - Scalable MLOps + LLMOps

## 📋 Deliverables

### Primary Deliverables
- [ ] **Complete data preprocessing pipeline** using advanced Metaflow features
- [ ] **Working LangChain + Ollama setup** with basic LCEL chains
- [ ] **Hybrid pipeline** combining Metaflow data processing with LangChain analysis

### Bonus Challenges
- [ ] Implement custom feature engineering functions
- [ ] Create a multi-model LLM comparison chain
- [ ] Build automated data quality reporting with LLM summaries

## 🔍 Self-Study Materials (8-10 hours)

### Reading (3-4 hours)
- **["Python for Data Analysis"](https://wesmckinney.com/book/)** - Chapters 5-6 (selected sections)
- **[Metaflow: Working with Data](https://docs.metaflow.org/metaflow/data)** - Data handling patterns
- **[LangChain Introduction](https://python.langchain.com/docs/introduction/)** - Core concepts
- **[LCEL Basics](https://python.langchain.com/docs/expression_language/)** - Expression language fundamentals

### Tutorials (3-4 hours)
- **[Metaflow Episode 3: Playlist Paradise](https://docs.metaflow.org/getting-started/tutorials/season-1-the-local-experience/episode03)**
- **[LangChain Quickstart](https://python.langchain.com/docs/tutorials/llm_chain/)**
- **[LCEL Get Started](https://python.langchain.com/docs/expression_language/get_started)**

### Practice (2-3 hours)
- Set up Ollama and download 2-3 different models
- Complete the preprocessing exercises in `/exercises/`
- Build your first LCEL chain following the tutorials

## 🎯 Success Criteria

### Technical Proficiency
- [ ] Can build complex data preprocessing pipelines in Metaflow
- [ ] Understands LCEL syntax and chain composition
- [ ] Successfully runs local LLMs with Ollama
- [ ] Integrates LangChain into Metaflow workflows

### Practical Skills
- [ ] Handles missing data and outliers effectively
- [ ] Creates meaningful features from raw data
- [ ] Writes clean, maintainable pipeline code
- [ ] Combines traditional ML with LLM capabilities

## 🛠 Troubleshooting

### Common Issues

**Ollama Installation Problems**
```bash
# MacOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows - Download from ollama.com
```

**LangChain Import Errors**
```bash
pip install --upgrade langchain langchain-community
```

**Memory Issues with Local LLMs**
- Use smaller models like `llama3.2:1b`
- Adjust Ollama memory settings
- Check available RAM before running large models

**More help**: See `resources/troubleshooting.md` or ask in Google Chat

## 🎖 Week 2 Knowledge Check

Quick quiz to verify understanding:

1. **LCEL Chain Composition**: What does the `|` operator do in LangChain?
2. **Local vs Cloud LLMs**: Name 2 advantages of using Ollama vs API calls
3. **Metaflow Features**: How do you handle parallel processing in data pipelines?
4. **Integration Patterns**: Why combine Metaflow with LangChain?

**Answers available in `/solutions/quiz_answers.md`**

## 🔗 Additional Resources

### Documentation
- [Metaflow Documentation](https://docs.metaflow.org/)
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.com/docs)

### Community
- [Metaflow Community Slack](https://join.slack.com/t/metaflow-community)
- [LangChain Discord](https://discord.gg/langchain)
- [r/MachineLearning](https://reddit.com/r/MachineLearning)

### Learning Path
- **Week 1**: Foundations complete ✅
- **Week 2**: Data processing + LangChain 🎯
- **Week 3**: Supervised learning with pipelines
- **Week 4**: Advanced ML + LangGraph introduction

---

## 🎯 Next Steps

1. **Complete environment setup** with LangChain and Ollama
2. **Review self-study materials** before workshop
3. **Join Monday's kickoff** session (12:00-1:00 PM)
4. **Attend Wednesday's workshop** (1:00-2:30 PM)
5. **Participate in Friday's showcase** (2:00-3:00 PM)

**Ready to bridge traditional ML with the power of LLMs? Let's dive into Week 2! 🚀**

---

*Questions? Check `/resources/` or reach out via Google Chat.*

*© 2025 INRIVA AI Academy. Designed for hands-on learning in modern AI/ML development.*