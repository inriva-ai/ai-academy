# INRIVA AI Academy - Week 1 Workshop Materials

Welcome to Week 1 of the INRIVA AI Academy! This repository contains all materials for our hands-on workshop covering **Foundations of AI/ML and MLOps with Metaflow**.

## 🎯 Workshop Overview

**Duration**: 90 minutes  
**Date**: Week 1 Wednesday, 1:00-2:30 PM  
**Focus**: Metaflow fundamentals, data exploration, and complete ML pipelines

### What You'll Learn
- ✅ Set up and use Metaflow for MLOps
- ✅ Perform professional data exploration with pandas
- ✅ Create effective data visualizations
- ✅ Build end-to-end ML pipelines
- ✅ Compare and evaluate multiple algorithms

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/inriva-ai/ai-academy.git
cd ai-academy
```

### 2. Set Up Environment
```bash
# Using conda (recommended)
conda env create -f setup/environment.yml
conda activate aiml-academy

# Or using pip
pip install -r setup/requirements.txt
```

### 3. Verify Setup
```bash
python setup/setup_test.py
```

### 4. Start Workshop
```bash
cd ai-academy/weeks/week1
jupyter notebook notebooks/01_environment_verification.ipynb
```

## 📁 Repository Contents

### 📓 Notebooks (`/notebooks/`)
Interactive Jupyter notebooks for the workshop:
- **01_environment_verification.ipynb** - Test your setup
- **02_metaflow_fundamentals.ipynb** - Learn Metaflow basics
- **03_data_exploration.ipynb** - Master pandas exploration
- **04_visualization_basics.ipynb** - Create effective plots
- **05_complete_ml_pipeline.ipynb** - Build production ML workflows

### 🌊 Metaflow Flows (`/flows/`)
Complete Metaflow workflow examples:
- **workshop_intro_flow.py** - Basic Metaflow concepts
- **wine_classification_flow.py** - Complete ML pipeline
- **advanced_example_flow.py** - Advanced techniques

### 📊 Data (`/data/`)
Sample datasets for practice:
- **sample_data.csv** - Custom workshop dataset
- Built-in datasets (iris, wine, etc.) accessed via scikit-learn

### 🎯 Exercises (`/exercises/`)
Additional practice materials:
- **practice_exercises.md** - Self-paced challenges
- **challenge_problems.py** - Advanced exercises

### 💡 Solutions (`/solutions/`)
Complete solutions for reference:
- **completed_notebook.ipynb** - Full workshop notebook
- **exercise_solutions.py** - Exercise answers

### 📚 Resources (`/resources/`)
Quick reference materials:
- **metaflow_cheatsheet.md** - Metaflow command reference
- **pandas_quickref.md** - Essential pandas operations
- **troubleshooting.md** - Common issues and solutions

## 🛠 Technical Requirements

### Software Requirements
- **Python 3.8+**
- **Conda or pip** for package management
- **Jupyter Notebook** or JupyterLab
- **Git** for repository management

### Hardware Requirements
- **4GB RAM minimum** (8GB recommended)
- **2GB free disk space**
- **Stable internet connection**

### Workshop Requirements
- **Google Meet** access
- **Microphone and camera**
- **Two monitors** (recommended for follow-along coding)

## 📋 Pre-Workshop Checklist

- [ ] Repository cloned successfully
- [ ] Environment set up and activated
- [ ] `setup_test.py` runs without errors
- [ ] Jupyter notebook starts correctly
- [ ] Google Meet access confirmed
- [ ] Workshop time in calendar

## 🔧 Troubleshooting

### Common Issues

**Environment Setup Fails**
```bash
# Try updating conda first
conda update conda
conda env create -f setup/environment.yml --force
```

**Import Errors**
```bash
# Reinstall packages
pip install --upgrade metaflow pandas numpy matplotlib seaborn scikit-learn
```

**Jupyter Won't Start**
```bash
# Try different port
jupyter notebook --port=8889
```

**More help**: See `resources/troubleshooting.md` or ask in Google Chat #urgent-help

## 🎓 Learning Path

### Before Workshop
1. Complete environment setup
2. Review Python and pandas basics
3. Read Metaflow introduction primer

### During Workshop
1. Follow along with live coding
2. Ask questions freely
3. Complete exercises as we go
4. Take notes on key concepts

### After Workshop
1. Complete practice exercises
2. Experiment with different parameters
3. Try challenge problems
4. Prepare for Week 2 (LangChain introduction)

## 📞 Support

### During Workshop
- **Live Questions**: Unmute and ask
- **Chat Questions**: Use Google Meet chat
- **Technical Issues**: Google Chat #urgent-help

### After Workshop
- **Questions**: Google Chat or email facilitator
- **Technical Issues**: Google Chat #urgent-help
- **Office Hours**: Friday 2:00-3:00 PM

## 🔗 Additional Resources

- [Metaflow Documentation](https://docs.metaflow.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [INRIVA AI Academy Portal](https://academy.inriva.ai)

## 📅 What's Next

### Week 2 Preview
- **Advanced Data Processing** with Metaflow
- **Introduction to LangChain** and LCEL
- **Combining Traditional ML** with LLM capabilities

### Upcoming Workshops
- **Week 2**: Data preprocessing and LangChain basics
- **Week 3**: Supervised learning and model comparison
- **Week 4**: Advanced ML and LangGraph integration

---

**Happy Learning! 🚀**

*INRIVA AI Academy Team*