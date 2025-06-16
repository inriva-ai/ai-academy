# Ollama Installation and Setup Guide

## 🎯 What is Ollama?

Ollama is a powerful tool that allows you to run large language models locally on your machine. This means:
- **Privacy**: Your data never leaves your computer
- **Cost-effective**: No API fees or usage limits
- **Offline capability**: Works without internet connection
- **Control**: Choose exactly which models to use

## 📋 System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 4GB+ free space per model
- **OS**: macOS, Linux, or Windows

### Recommended Requirements
- **RAM**: 16GB+ for larger models
- **GPU**: Optional but significantly faster (NVIDIA/AMD)
- **Storage**: SSD for better performance

## 🔧 Installation

### macOS
```bash
# Option 1: Using Homebrew (recommended)
brew install ollama

# Option 2: Direct download
curl -fsSL https://ollama.com/install.sh | sh

# Option 3: Download from website
# Visit https://ollama.com/download and download the .dmg file
```

### Linux
```bash
# Ubuntu/Debian
curl -fsSL https://ollama.com/install.sh | sh

# Manual installation
curl -L https://ollama.com/download/ollama-linux-amd64 -o ollama
chmod +x ollama
sudo mv ollama /usr/local/bin/
```

### Windows
1. Visit [ollama.com/download](https://ollama.com/download)
2. Download the Windows installer
3. Run the installer and follow the setup wizard
4. Ollama will be available in Command Prompt or PowerShell

## 🚀 Quick Start

### 1. Start Ollama Service
```bash
# Start the Ollama service (runs in background)
ollama serve

# On Windows/macOS, this usually starts automatically
```

### 2. Download Your First Model
```bash
# Download Llama 3.2 (3B parameters, good balance of speed/quality)
ollama pull llama3.2

# For lower memory usage (1B parameters)
ollama pull llama3.2:1b

# For better quality but higher memory usage (8B parameters)
ollama pull llama3.2:8b
```

### 3. Test the Installation
```bash
# Interactive chat
ollama run llama3.2

# Quick test
ollama run llama3.2 "Hello, how are you?"
```

## 📊 Available Models

### Recommended Models for Data Science

#### For Beginners (Low Memory)
```bash
# Llama 3.2 1B - Fast, low memory usage
ollama pull llama3.2:1b

# Phi-3 Mini - Microsoft's efficient model
ollama pull phi3:mini
```

#### For General Use (Medium Memory)
```bash
# Llama 3.2 3B - Great balance of speed and quality
ollama pull llama3.2

# Mistral - Excellent for coding and analysis
ollama pull mistral
```

#### For Advanced Use (High Memory)
```bash
# Llama 3.2 8B - High quality responses
ollama pull llama3.2:8b

# Code Llama - Specialized for programming
ollama pull codellama
```

### Model Comparison
| Model | Size | RAM Needed | Best For |
|-------|------|------------|----------|
| llama3.2:1b | 1B | 4GB+ | Quick tasks, low memory |
| llama3.2 | 3B | 8GB+ | General purpose |
| llama3.2:8b | 8B | 16GB+ | High-quality analysis |
| mistral | 7B | 12GB+ | Code and reasoning |
| codellama | 7B | 12GB+ | Programming tasks |

## 🔧 Configuration

### Basic Configuration
```bash
# List installed models
ollama list

# Show model information
ollama show llama3.2

# Remove a model
ollama rm model_name
```

### Environment Variables
```bash
# Set Ollama host (default: localhost:11434)
export OLLAMA_HOST=0.0.0.0:11434

# Set number of parallel requests
export OLLAMA_NUM_PARALLEL=1

# Set max loaded models
export OLLAMA_MAX_LOADED_MODELS=1

# Keep model in memory (seconds)
export OLLAMA_KEEP_ALIVE=5m
```

### Memory Management
```bash
# Check GPU usage (if available)
ollama ps

# Unload all models from memory
ollama stop

# Unload specific model
ollama stop llama3.2
```

## 🔌 Integration with Python

### Basic Setup
```python
# Install required packages
# pip install langchain-community

from langchain_community.llms import Ollama

# Initialize model
llm = Ollama(model="llama3.2")

# Test connection
response = llm.invoke("Hello, are you working?")
print(response)
```

### Advanced Configuration
```python
from langchain_community.llms import Ollama

llm = Ollama(
    model="llama3.2",
    base_url="http://localhost:11434",  # Ollama server URL
    temperature=0.7,                    # Creativity (0.0-1.0)
    num_predict=256,                   # Max tokens to generate
    top_p=0.9,                         # Nucleus sampling
    top_k=40,                          # Top-k sampling
    repeat_penalty=1.1,                # Avoid repetition
    stop=["Human:", "AI:"],            # Stop tokens
)
```

### Error Handling
```python
import subprocess
import time

def ensure_ollama_running():
    """Ensure Ollama service is running"""
    try:
        # Check if Ollama is responding
        result = subprocess.run(
            ['ollama', 'list'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Try to start Ollama
    try:
        subprocess.Popen(['ollama', 'serve'])
        time.sleep(3)  # Wait for startup
        return True
    except FileNotFoundError:
        print("❌ Ollama not installed")
        return False

# Use in your code
if ensure_ollama_running():
    llm = Ollama(model="llama3.2")
else:
    print("Please install and start Ollama")
```

## 🛠 Troubleshooting

### Common Issues

#### "Connection refused" or "Service unavailable"
```bash
# Start Ollama service
ollama serve

# Check if service is running
ps aux | grep ollama

# Kill existing processes if needed
pkill ollama
ollama serve
```

#### "Model not found"
```bash
# List available models
ollama list

# Pull the model you need
ollama pull llama3.2

# Check model status
ollama ps
```

#### High memory usage
```bash
# Use smaller models
ollama pull llama3.2:1b

# Limit parallel requests
export OLLAMA_NUM_PARALLEL=1

# Reduce keep-alive time
export OLLAMA_KEEP_ALIVE=1m

# Unload models when not in use
ollama stop
```

#### Slow responses
```bash
# Check if GPU is being used
ollama ps

# For NVIDIA GPUs, install CUDA drivers
# For AMD GPUs, install ROCm

# Use smaller models for faster responses
ollama pull phi3:mini
```

### Performance Optimization

#### For CPU Users
```bash
# Use quantized models (smaller, faster)
ollama pull llama3.2:1b

# Adjust thread count
export OMP_NUM_THREADS=4
```

#### For GPU Users
```bash
# Verify GPU is detected
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD

# Pull unquantized models for better quality
ollama pull llama3.2:8b
```

## 📚 Best Practices

### Model Selection
```python
# For data analysis tasks
recommended_models = {
    "quick_analysis": "llama3.2:1b",      # Fast, basic insights
    "general_purpose": "llama3.2",         # Balanced performance
    "detailed_analysis": "llama3.2:8b",    # High-quality insights
    "code_tasks": "codellama",             # Programming and debugging
}
```

### Prompt Engineering
```python
# Good: Specific and clear
prompt = """
Analyze this dataset summary and provide exactly 3 insights:
Rows: 1000
Columns: age, income, education, target
Target distribution: 60% class A, 40% class B

Format your response as:
1. [Insight about data distribution]
2. [Insight about potential patterns]
3. [Insight about modeling considerations]
"""

# Avoid: Vague and open-ended
prompt = "Tell me about this data"
```

### Resource Management
```python
class OllamaManager:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.llm = None
    
    def __enter__(self):
        self.llm = Ollama(model=self.model_name)
        return self.llm
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Unload model to free memory
        subprocess.run(['ollama', 'stop', self.model_name], 
                      capture_output=True)

# Usage
with OllamaManager("llama3.2") as llm:
    response = llm.invoke("Analyze this data...")
    # Model automatically unloaded when done
```

## 🔄 Updating and Maintenance

### Keep Ollama Updated
```bash
# Update Ollama (macOS with Homebrew)
brew upgrade ollama

# Linux - re-run install script
curl -fsSL https://ollama.com/install.sh | sh

# Check version
ollama --version
```

### Model Management
```bash
# Update existing models
ollama pull llama3.2  # Downloads latest version

# Clean up old models
ollama list  # See what's installed
ollama rm old_model_name

# Check disk usage
du -sh ~/.ollama/models/*
```

## 📞 Getting Help

### Official Resources
- **Website**: [ollama.com](https://ollama.com)
- **GitHub**: [github.com/ollama/ollama](https://github.com/ollama/ollama)
- **Discord**: Join the Ollama community Discord

### Command Line Help
```bash
# General help
ollama --help

# Command-specific help
ollama pull --help
ollama run --help
```

### Verify Installation
```python
# Complete verification script
import subprocess
import sys

def verify_ollama_setup():
    """Complete Ollama setup verification"""
    print("🔍 Verifying Ollama installation...")
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True)
        print(f"✅ Ollama version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Ollama not found. Please install from ollama.com")
        return False
    
    # Check if service is running
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama service is running")
            
            # List available models
            models = result.stdout.strip()
            if models and "NAME" in models:
                print("✅ Available models:")
                print(models)
            else:
                print("⚠️  No models installed")
                print("💡 Run: ollama pull llama3.2")
        else:
            print("❌ Ollama service not responding")
            print("💡 Run: ollama serve")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Ollama service timeout")
        return False
    
    # Test LangChain integration
    try:
        from langchain_community.llms import Ollama
        print("✅ LangChain integration available")
    except ImportError:
        print("⚠️  LangChain not installed")
        print("💡 Run: pip install langchain-community")
    
    print("\n🎉 Ollama setup verification complete!")
    return True

if __name__ == "__main__":
    verify_ollama_setup()
```

---

**🎯 Quick Start Summary:**
1. Install Ollama: `brew install ollama` (macOS) or visit ollama.com
2. Pull a model: `ollama pull llama3.2`
3. Test: `ollama run llama3.2 "Hello!"`
4. Use in Python: `Ollama(model="llama3.2")`

**💡 Need help?** Run the verification script above or check the troubleshooting section!