#!/usr/bin/env python3
"""
INRIVA AI Academy 2025 - Complete Setup Verification
=====================================================

This script verifies that your environment is correctly set up
for the entire 8-week AI/ML internship program.

Run this script before starting any week to ensure everything works!
"""

import sys
import subprocess
import importlib
import warnings
warnings.filterwarnings('ignore')

def print_header():
    """Print program header"""
    print("=" * 70)
    print("🎓 INRIVA AI Academy 2025 - Environment Setup Verification")
    print("   8-Week AI/ML & Generative AI Internship Program")
    print("=" * 70)

def test_python_version():
    """Test Python version compatibility."""
    print("\n🐍 Testing Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.9+")
        return False

def test_core_imports():
    """Test core package imports for all weeks."""
    print("\n📦 Testing core package imports...")
    
    core_packages = {
        # Week 1-2: Foundation
        'metaflow': 'Metaflow MLOps framework',
        'pandas': 'Data manipulation library',
        'numpy': 'Numerical computing library',
        'matplotlib': 'Plotting library',
        'seaborn': 'Statistical visualization',
        'sklearn': 'Machine learning library',
        
        # Week 2-4: LangChain ecosystem
        'langchain': 'LangChain LLM framework',
        'langchain_community': 'LangChain community integrations',
        
        # Week 4-6: Advanced AI
        'langgraph': 'LangGraph agent framework',
        
        # Development tools
        'jupyter': 'Jupyter notebook environment'
    }
    
    failed_imports = []
    
    for package, description in core_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"   ✅ {package} v{version} - {description}")
        except ImportError as e:
            print(f"   ❌ {package} - {description} (FAILED: {e})")
            failed_imports.append(package)
    
    return len(failed_imports) == 0, failed_imports

def test_optional_imports():
    """Test optional package imports."""
    print("\n📦 Testing optional packages...")
    
    optional_packages = {
        'xgboost': 'Gradient boosting framework',
        'lightgbm': 'Light gradient boosting',
        'plotly': 'Interactive plotting',
        'streamlit': 'Web app framework',
        'requests': 'HTTP library',
        'mlflow': 'ML experiment tracking'
    }
    
    optional_available = []
    
    for package, description in optional_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"   ✅ {package} v{version} - {description}")
            optional_available.append(package)
        except ImportError:
            print(f"   ⚠️  {package} - {description} (Optional - not installed)")
    
    return optional_available

def test_metaflow_functionality():
    """Test Metaflow basic functionality."""
    print("\n🌊 Testing Metaflow functionality...")
    
    try:
        from metaflow import FlowSpec, step, Parameter
        
        class TestFlow(FlowSpec):
            test_param = Parameter('test_param', default='academy')
            
            @step
            def start(self):
                self.message = f"Hello {self.test_param}!"
                self.next(self.end)
            
            @step  
            def end(self):
                print(f"   ✅ Metaflow test successful: {self.message}")
        
        # Test flow creation
        flow = TestFlow()
        print("   ✅ Metaflow FlowSpec creation successful")
        print("   ✅ Parameter handling working")
        return True
        
    except Exception as e:
        print(f"   ❌ Metaflow test failed: {e}")
        return False

def test_langchain_functionality():
    """Test LangChain basic functionality."""
    print("\n🔗 Testing LangChain functionality...")
    
    try:
        # Test core LangChain imports
        from langchain.schema import HumanMessage, AIMessage
        from langchain.prompts import ChatPromptTemplate
        
        # Test prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            ("human", "Hello {name}!")
        ])
        
        messages = prompt.format_messages(name="Academy Student")
        assert len(messages) == 2
        
        print("   ✅ LangChain core imports successful")
        print("   ✅ Prompt template creation working")
        
        # Test LangGraph if available
        try:
            from langgraph.graph import StateGraph
            print("   ✅ LangGraph available for advanced weeks")
        except ImportError:
            print("   ⚠️  LangGraph not available (install for weeks 4-8)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LangChain test failed: {e}")
        return False

def test_ml_stack():
    """Test machine learning stack."""
    print("\n🤖 Testing ML/Data Science stack...")
    
    try:
        # Test pandas
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert len(df) == 3
        print("   ✅ Pandas DataFrame operations")
        
        # Test numpy
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        print("   ✅ NumPy array operations")
        
        # Test matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.close(fig)
        print("   ✅ Matplotlib plotting")
        
        # Test scikit-learn
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        print(f"   ✅ Scikit-learn ML pipeline (accuracy: {accuracy:.3f})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ML stack test failed: {e}")
        return False

def test_jupyter_environment():
    """Test Jupyter environment."""
    print("\n📓 Testing Jupyter environment...")
    
    try:
        result = subprocess.run(['jupyter', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   ✅ Jupyter command accessible")
            
            # Check for JupyterLab
            result_lab = subprocess.run(['jupyter-lab', '--version'], 
                                      capture_output=True, text=True, timeout=10)
            if result_lab.returncode == 0:
                print("   ✅ JupyterLab available")
            
            return True
        else:
            print("   ❌ Jupyter command failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ❌ Jupyter command timeout")
        return False
    except FileNotFoundError:
        print("   ❌ Jupyter not found in PATH")
        return False
    except Exception as e:
        print(f"   ❌ Jupyter test failed: {e}")
        return False

def test_week_readiness():
    """Test readiness for specific weeks."""
    print("\n📅 Testing week-specific readiness...")
    
    week_tests = {
        "Weeks 1-2 (Foundation)": lambda: test_week_1_2(),
        "Weeks 3-4 (Advanced ML)": lambda: test_week_3_4(),
        "Weeks 5-6 (Deep Learning)": lambda: test_week_5_6(),
        "Weeks 7-8 (Production)": lambda: test_week_7_8()
    }
    
    results = {}
    for week_range, test_func in week_tests.items():
        try:
            results[week_range] = test_func()
        except Exception as e:
            print(f"   ❌ {week_range} test failed: {e}")
            results[week_range] = False
    
    return results

def test_week_1_2():
    """Test Week 1-2 readiness (Foundation)."""
    try:
        import metaflow
        import pandas as pd
        import sklearn
        print("   ✅ Week 1-2 ready: Metaflow + Traditional ML")
        return True
    except ImportError:
        print("   ❌ Week 1-2 not ready: Missing foundation packages")
        return False

def test_week_3_4():
    """Test Week 3-4 readiness (LangChain integration)."""
    try:
        import langchain
        import metaflow
        print("   ✅ Week 3-4 ready: LangChain + Metaflow integration")
        return True
    except ImportError:
        print("   ❌ Week 3-4 not ready: Missing LangChain")
        return False

def test_week_5_6():
    """Test Week 5-6 readiness (Advanced agents)."""
    try:
        import langgraph
        import langchain
        print("   ✅ Week 5-6 ready: LangGraph + Advanced agents")
        return True
    except ImportError:
        print("   ⚠️  Week 5-6 partially ready: Install LangGraph for full functionality")
        return False

def test_week_7_8():
    """Test Week 7-8 readiness (Production)."""
    try:
        import streamlit
        import metaflow
        import langchain
        print("   ✅ Week 7-8 ready: Production deployment tools")
        return True
    except ImportError:
        print("   ⚠️  Week 7-8 partially ready: Some deployment tools missing")
        return False

def run_complete_test():
    """Run comprehensive setup test."""
    print_header()
    
    # Core tests
    tests = [
        ("Python Version", test_python_version),
        ("Core Packages", lambda: test_core_imports()[0]),
        ("Metaflow Functionality", test_metaflow_functionality),
        ("LangChain Functionality", test_langchain_functionality),
        ("ML/Data Science Stack", test_ml_stack),
        ("Jupyter Environment", test_jupyter_environment)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Optional packages
    optional_available = test_optional_imports()
    
    # Week-specific readiness
    week_results = test_week_readiness()
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SETUP VERIFICATION SUMMARY")
    print("=" * 70)
    
    # Core tests summary
    passed = 0
    total = len(results)
    
    print("\n🔧 Core Requirements:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status:8} | {test_name}")
        if result:
            passed += 1
    
    # Week readiness summary
    print(f"\n📅 Week Readiness:")
    for week_range, ready in week_results.items():
        status = "✅ READY" if ready else "⚠️  PARTIAL"
        print(f"  {status:9} | {week_range}")
    
    # Optional packages summary
    print(f"\n📦 Optional Packages Available: {len(optional_available)}")
    
    print("-" * 70)
    print(f"CORE TESTS: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 CONGRATULATIONS! Your environment is ready for the AI Academy!")
        print("\n🚀 Next Steps:")
        print("1. 📅 Check your program calendar for upcoming sessions")
        print("2. 📚 Navigate to the current week's materials")
        print("3. 💻 Keep this environment activated for all work")
        print("4. 🙋 Join the Google Chat workspace for communication")
        print("5. 🎯 Start building amazing AI/ML solutions!")
        
        if len(optional_available) < 5:
            print(f"\n💡 Tip: You have {len(optional_available)} optional packages.")
            print("   Consider installing more for enhanced functionality:")
            print("   pip install plotly streamlit xgboost lightgbm")
        
    else:
        print(f"\n⚠️  WARNING: {total - passed} core tests failed!")
        print("\nPlease fix the issues above before starting the program.")
        print("\n🆘 Get help:")
        print("- 💬 Google Chat #urgent-help")
        print("- 📧 Email the program coordinator")
        print("- 📚 Check setup/troubleshooting.md")
        
        # Show failed imports
        _, failed_imports = test_core_imports()
        if failed_imports:
            print(f"\n📦 To fix import issues, try:")
            print(f"   pip install {' '.join(failed_imports)}")
            print("   OR")
            print("   conda env create -f setup/environment.yml --force")
    
    print("\n" + "=" * 70)
    return passed == total

if __name__ == "__main__":
    success = run_complete_test()
    sys.exit(0 if success else 1)
