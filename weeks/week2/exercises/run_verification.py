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
