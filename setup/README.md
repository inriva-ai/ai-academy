# Setup Instructions - INRIVA AI Academy

This directory contains everything needed to set up your development environment for the 8-week AI Academy program.

## 🎯 Setup Overview

The setup process will install and configure:
- **Python environment** with all required packages
- **Metaflow** for MLOps workflows
- **LangChain & LangGraph** for LLM applications
- **Ollama** for local model deployment (optional)
- **Jupyter** for interactive development
- **API access** for commercial LLM services (optional)

## 📋 Prerequisites

Before starting, ensure you have:

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free disk space
- **Internet**: Stable broadband connection
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

### Software Prerequisites
- **Python 3.8-3.11** (3.9 recommended)
- **Git** for version control
- **Conda** (recommended) OR **pip** for package management

### Check Prerequisites
```bash
# Check Python version
python --version
# Should show: Python 3.8.x, 3.9.x, 3.10.x, or 3.11.x

# Check Git
git --version
# Should show: git version 2.x.x

# Check Conda (if using)
conda --version
# Should show: conda x.x.x

# Check available disk space
# Windows: dir
# macOS/Linux: df -h
```

## 🚀 Quick Setup (Recommended)

### Option 1: Automated Setup with Conda
```bash
# 1. Clone repository
git clone https://github.com/inriva-ai/ai-academy.git
cd ai-academy

# 2. Create environment from file
conda env create -f setup/environment.yml

# 3. Activate environment
conda activate aiml-academy

# 4. Verify installation
python setup/setup_test.py
```

### Option 2: Manual Setup with pip
```bash
# 1. Clone repository  
git clone https://github.com/inriva-ai/ai-academy.git
cd ai-academy

# 2. Create virtual environment
python -m venv aiml-academy

# 3. Activate environment
# On Windows:
aiml-academy\Scripts\activate
# On macOS/Linux:
source aiml-academy/bin/activate

# 4. Install packages
pip install -r setup/requirements.txt

# 5. Verify installation
python setup/setup_test.py
```

## 📦 Package Overview

### Core Framework Packages
- **metaflow** (≥2.7.0) - MLOps workflow management
- **langchain** (≥0.1.0) - LLM application framework
- **langgraph** (≥0.0.30) - Multi-agent workflows
- **langchain-community** - Community integrations

### Data Science Stack
- **pandas** (≥1.5.0) - Data manipulation and analysis
- **numpy** (≥1.21.0) - Numerical computing
- **scikit-learn** (≥1.1.0) - Machine learning algorithms
- **matplotlib** (≥3.5.0) - Plotting and visualization
- **seaborn** (≥0.11.0) - Statistical visualization

### Development Environment
- **jupyter** (≥1.0.0) - Interactive notebooks
- **jupyterlab** (≥3.0.0) - Advanced notebook interface
- **ipywidgets** (≥7.6.0) - Interactive widgets

### Optional Packages
- **ollama** - Local LLM deployment
- **openai** - OpenAI API client
- **anthropic** - Anthropic Claude API client
- **google-generativeai** - Google Gemini API client

## 🔧 Detailed Setup Instructions

### Step 1: Install Prerequisites

#### Install Python (if needed)
**Windows:**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer with "Add Python to PATH" checked
3. Verify: `python --version`

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.9

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-pip
```

#### Install Git (if needed)
**Windows:** Download from [git-scm.com](https://git-scm.com/)
**macOS:** `brew install git` or download from git-scm.com
**Linux:** `sudo apt install git`

#### Install Conda (recommended)
**All Platforms:**
1. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Run installer with default settings
3. Restart terminal/command prompt
4. Verify: `conda --version`

### Step 2: Environment Setup

#### Create Conda Environment
```bash
# Navigate to repository
cd ai-academy

# Create environment with specific Python version
conda env create -f setup/environment.yml

# Alternative: Create manually
conda create -n aiml-academy python=3.9
conda activate aiml-academy
conda install -c conda-forge jupyter pandas numpy matplotlib seaborn scikit-learn
pip install metaflow langchain langgraph
```

#### Create Virtual Environment (Alternative)
```bash
# Create virtual environment
python -m venv aiml-academy

# Activate (Windows)
aiml-academy\Scripts\activate

# Activate (macOS/Linux) 
source aiml-academy/bin/activate

# Install packages
pip install --upgrade pip
pip install -r setup/requirements.txt
```

### Step 3: Verification

#### Run Comprehensive Test
```bash
# Activate your environment first
conda activate aiml-academy
# OR: source aiml-academy/bin/activate

# Run verification script
python setup/setup_test.py

# Expected output:
# ✅ Python version compatible
# ✅ All packages imported successfully
# ✅ Metaflow test successful
# ✅ Data science stack working
# ✅ Jupyter environment ready
# 🎉 Environment ready for workshop!
```

#### Test Jupyter Setup
```bash
# Start Jupyter Notebook
jupyter notebook

# Or start JupyterLab
jupyter lab

# Navigate to weeks/week1/notebooks/01_environment_verification.ipynb
# Run all cells - should complete without errors
```

### Step 4: API Configuration (Optional)

If you plan to use commercial LLM APIs:

#### Create API Keys File
```bash
# Copy template
cp setup/.env.template .env

# Edit with your API keys
# .env file should contain:
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
```

#### API Key Sources
- **OpenAI**: [platform.openai.com](https://platform.openai.com/api-keys)
- **Anthropic**: [console.anthropic.com](https://console.anthropic.com/)
- **Google AI**: [ai.google.dev](https://ai.google.dev/)
- **LangSmith**: [smith.langchain.com](https://smith.langchain.com/)

#### Test API Access
```bash
python setup/api_test.py
```

### Step 5: Ollama Setup (Optional)

For local LLM deployment:

#### Install Ollama
**Windows/macOS:** Download from [ollama.com](https://ollama.com/)
**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Download Models
```bash
# Install a small model for testing
ollama pull llama3.2:3b

# Install larger model for production use
ollama pull llama3.2:8b

# Test installation
ollama run llama3.2:3b "Hello, how are you?"
```

## 🐛 Troubleshooting

### Common Issues

#### Issue: Python Version Incompatible
**Error:** `Python 3.7.x detected, requires 3.8+`
**Solution:**
```bash
# Install newer Python version
# Windows: Download from python.org
# macOS: brew install python@3.9  
# Linux: sudo apt install python3.9

# Create environment with specific version
conda create -n aiml-academy python=3.9
```

#### Issue: Package Installation Fails
**Error:** `ERROR: Failed building wheel for [package]`
**Solutions:**
```bash
# Update pip first
pip install --upgrade pip setuptools wheel

# Clear pip cache
pip cache purge

# Install with no cache
pip install --no-cache-dir -r setup/requirements.txt

# For conda users
conda clean --all
conda update conda
```

#### Issue: Metaflow Import Error
**Error:** `ModuleNotFoundError: No module named 'metaflow'`
**Solutions:**
```bash
# Ensure environment is activated
conda activate aiml-academy
# OR: source aiml-academy/bin/activate

# Reinstall Metaflow
pip install --upgrade metaflow

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Issue: Jupyter Kernel Not Found
**Error:** `Kernel aiml-academy not found`
**Solution:**
```bash
# Install kernel in environment
conda activate aiml-academy
python -m ipykernel install --user --name aiml-academy --display-name "AI Academy"

# Restart Jupyter and select correct kernel
```

#### Issue: Memory/Performance Problems
**Symptoms:** Slow performance, frequent crashes
**Solutions:**
```bash
# Check available memory
# Windows: Task Manager
# macOS: Activity Monitor  
# Linux: htop

# Reduce memory usage
# Set environment variable
export PYTHONHASHSEED=0

# Use smaller datasets for testing
# Close unnecessary applications
```

### Platform-Specific Issues

#### Windows Issues
**Long path problems:**
```bash
# Enable long paths (run as Administrator)
git config --system core.longpaths true
```

**PowerShell execution policy:**
```bash
# Set execution policy (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### macOS Issues  
**Xcode command line tools:**
```bash
# Install if getting compiler errors
xcode-select --install
```

**Permission issues:**
```bash
# Fix homebrew permissions
sudo chown -R $(whoami) /usr/local/lib/python3.9/site-packages
```

#### Linux Issues
**Missing system packages:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential python3-dev python3-venv

# CentOS/RHEL
sudo yum install gcc gcc-c++ python3-devel
```

### Getting Help

#### Self-Diagnosis
1. **Run setup test**: `python setup/setup_test.py`
2. **Check environment**: `conda info --envs` or `which python`
3. **Verify packages**: `pip list | grep metaflow`
4. **Test imports**: `python -c "import metaflow, langchain"`

#### Support Channels
- **🔍 Search first**: `/resources/troubleshooting/`
- **💬 Google Chat**: #urgent-help channel
- **📧 Email**: [technical-support-email]
- **🎫 GitHub Issues**: For persistent problems
- **📞 Office Hours**: Friday 2:00-3:00 PM

#### Information to Include When Asking for Help
```bash
# System information
python --version
pip --version
conda --version  # if using conda

# Environment information  
conda env list  # or: which python
pip list

# Error details
# Copy full error message and stack trace
# Include steps that led to the error
```

## 📚 Additional Setup Options

### Development Tools (Optional)

#### VS Code Setup
1. Install [VS Code](https://code.visualstudio.com/)
2. Install Python extension
3. Install Jupyter extension
4. Configure Python interpreter:
   - Ctrl+Shift+P → "Python: Select Interpreter"
   - Choose `aiml-academy` environment

#### Git Configuration
```bash
# Set user information
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set default editor
git config --global core.editor "code --wait"  # For VS Code
```

### Advanced Configuration

#### Jupyter Extensions
```bash
# Install useful extensions
pip install jupyter-contrib-nbextensions
jupyter contrib nbextension install --user

# Enable extensions
jupyter nbextension enable collapsible_headings/main
jupyter nbextension enable code_folding/main
```

#### Environment Variables
Create `setup/env_setup.sh` (Linux/macOS) or `setup/env_setup.bat` (Windows):
```bash
# env_setup.sh
export METAFLOW_DEFAULT_DATASTORE=local
export METAFLOW_DEFAULT_METADATA=local
export LANGCHAIN_TRACING_V2=true
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Performance Optimization
```bash
# Set environment variables for better performance
export OMP_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

## ✅ Setup Verification Checklist

Complete this checklist to ensure your setup is ready:

### Environment Setup
- [ ] Python 3.8+ installed and accessible
- [ ] Git installed and configured
- [ ] Repository cloned successfully
- [ ] Virtual environment/conda environment created
- [ ] All packages installed without errors

### Package Verification
- [ ] `python setup/setup_test.py` passes all tests
- [ ] Metaflow imports successfully
- [ ] LangChain imports successfully  
- [ ] Jupyter notebook starts correctly
- [ ] Can run cells in verification notebook

### Optional Components
- [ ] API keys configured (if using commercial APIs)
- [ ] Ollama installed and tested (if using local models)
- [ ] Development tools configured (VS Code, etc.)

### Communication Setup
- [ ] Google Chat access confirmed
- [ ] Google Meet tested for workshops
- [ ] Email notifications enabled
- [ ] Calendar events added

## 🎓 Next Steps

Once setup is complete:

1. **📅 Join Week 1 kickoff** - Monday 12:00 PM
2. **📖 Review Week 1 materials** - `/weeks/week1/`
3. **🤝 Introduce yourself** - Google Chat #general
4. **❓ Ask questions** - We're here to help!

## 📞 Support Information

### Immediate Help
- **Google Chat**: #urgent-help (fastest response)
- **Email**: [technical-support]
- **Office Hours**: Friday 2:00-3:00 PM

### Self-Help Resources
- **Troubleshooting Guide**: `/resources/troubleshooting/`
- **FAQ**: `/docs/program-overview/faq.md`
- **Video Tutorials**: [link-to-video-playlist]

---

**Setup complete? Welcome to the AI Academy! 🚀**

*Having issues? Don't worry - we'll get you sorted out. Reach out via Google Chat #urgent-help.*