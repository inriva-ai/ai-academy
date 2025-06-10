# INRIVA AI Academy 2025 - 8-Week AI/ML & Generative AI Program

Welcome to the INRIVA AI Academy! This repository contains all materials for our comprehensive 8-week internship program covering **AI/ML fundamentals, MLOps with Metaflow, and Generative AI with LangChain and LangGraph**.

## 🎯 Program Overview

**Duration**: 8 weeks, 20 hours/week  
**Focus**: Production-ready AI/ML solutions using modern MLOps and LLMOps frameworks  
**Technology Stack**: Metaflow, LangChain, LangGraph, Ollama + Commercial APIs

### Program Goals
- ✅ Build foundational knowledge in traditional ML and generative AI
- ✅ Gain practical experience with production-grade tools and frameworks
- ✅ Develop problem-solving skills through hands-on projects
- ✅ Learn from experienced professionals through mentorship
- ✅ Create a portfolio-worthy capstone project

### Core Technology Stack
- **🌊 Metaflow**: Scalable data science workflows and MLOps
- **🦜 LangChain + LCEL**: LLM integration and chain composition  
- **🕸️ LangGraph**: Multi-agent system workflows
- **🦙 Ollama**: Privacy-focused local LLM deployment
- **☁️ Commercial APIs**: OpenAI GPT, Google Gemini, Anthropic Claude

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/inriva-ai/ai-academy.git
cd ai-academy
```

### 2. Environment Setup
```bash
# Create and activate environment
conda env create -f setup/environment.yml
conda activate aiml-academy

# Or using pip
pip install -r setup/requirements.txt
```

### 3. Verify Installation
```bash
python setup/setup_test.py
```

### 4. Start Learning
```bash
# Navigate to current week
cd weeks/week1
jupyter notebook notebooks/
```

## 📁 Repository Structure

### 🏗️ Setup (`/setup/`)
Environment configuration and verification:
- **environment.yml** - Conda environment specification
- **requirements.txt** - Pip package requirements  
- **setup_test.py** - Comprehensive setup verification
- **README.md** - Detailed setup instructions

### 📅 Weekly Materials (`/weeks/`)
Complete materials for each week:
```
weeks/
├── week1/          # Foundations & Metaflow
├── week2/          # Data Processing & LangChain
├── week3/          # Supervised Learning
├── week4/          # Advanced ML & LangGraph
├── week5/          # Unsupervised Learning
├── week6/          # Deep Learning & Advanced Agents
├── week7/          # Production MLOps/LLMOps
└── week8/          # Capstone Project
```

Each week contains:
- **📓 notebooks/** - Interactive Jupyter notebooks
- **🌊 flows/** - Metaflow workflow examples
- **🎯 exercises/** - Practice problems and challenges
- **💡 solutions/** - Complete solutions and explanations

### 📊 Data (`/data/`)
Curated datasets for the program:
- **raw/** - Original, unprocessed datasets
- **processed/** - Cleaned and prepared data
- **external/** - External APIs and data sources

### 📚 Resources (`/resources/`)
Reference materials and guides:
- **cheatsheets/** - Quick reference cards (Metaflow, pandas, LangChain, etc.)
- **primers/** - In-depth introductory materials
- **troubleshooting/** - Common issues and solutions

### 🎨 Projects (`/projects/`)
Major project work:
- **capstone/** - Week 8 capstone project templates and examples

### 📖 Documentation (`/docs/`)
Program documentation:
- **program-overview/** - Detailed syllabus and learning objectives
- **weekly-guides/** - Week-by-week instructor and student guides
- **assessment-rubrics/** - Evaluation criteria and grading rubrics

## 📋 Weekly Learning Path

### Week 1: Foundations of AI/ML and MLOps with Metaflow
- **Focus**: Metaflow basics, data exploration, first ML pipeline
- **Key Skills**: Python fundamentals, pandas, basic ML workflows
- **Deliverables**: Working environment, simple classification pipeline

### Week 2: Data Preprocessing and LangChain Introduction  
- **Focus**: Advanced data processing, LangChain basics with LCEL
- **Key Skills**: Feature engineering, text processing, LLM integration
- **Deliverables**: Data pipeline with LLM analysis components

### Week 3: Supervised Learning with Metaflow Pipelines
- **Focus**: Classification and regression with multiple algorithms
- **Key Skills**: Model comparison, hyperparameter tuning, evaluation
- **Deliverables**: Comprehensive ML comparison pipeline

### Week 4: Advanced ML and LangChain Integration
- **Focus**: Ensemble methods, LangGraph introduction
- **Key Skills**: Advanced ML techniques, agent-based workflows
- **Deliverables**: Hybrid ML + LLM analysis system

### Week 5: Unsupervised Learning and LangGraph Agents
- **Focus**: Clustering, dimensionality reduction, multi-agent systems
- **Key Skills**: Pattern discovery, agent communication, complex workflows
- **Deliverables**: Intelligent data analysis agent system

### Week 6: Deep Learning and Advanced Agent Systems
- **Focus**: Neural networks, hierarchical agents, human-in-the-loop
- **Key Skills**: CNN/RNN implementation, advanced agent architectures
- **Deliverables**: Deep learning pipeline with agent monitoring

### Week 7: Production MLOps/LLMOps and Agent Systems
- **Focus**: Deployment, monitoring, production best practices
- **Key Skills**: Pipeline automation, system monitoring, scalability
- **Deliverables**: Production-ready ML/LLM system

### Week 8: Production-Ready AI/ML Capstone Project
- **Focus**: End-to-end system development using full technology stack
- **Key Skills**: System integration, problem-solving, presentation
- **Deliverables**: Complete AI solution with business impact analysis

## 🛠 Technical Requirements

### Software Requirements
- **Python 3.8+** with scientific computing stack
- **Conda or pip** for package management
- **Jupyter Notebook/Lab** for interactive development
- **Git** for version control
- **Docker** (optional, for advanced deployment)

### Hardware Requirements  
- **8GB RAM minimum** (16GB recommended)
- **5GB free disk space**
- **Stable internet connection** for API access
- **Modern web browser** for Google Meet workshops

### API Access (Optional)
- **OpenAI API** key for GPT models
- **Anthropic API** key for Claude models  
- **Google AI** key for Gemini models
- **Local alternatives**: Ollama for privacy-focused development

## 🎓 Learning Support

### Weekly Structure
- **Monday (1 hour)**: Week kickoff and planning session
- **Wednesday (1.5 hours)**: Technical workshop with live coding
- **Friday (1 hour)**: Showcase and office hours
- **Individual**: 30-minute weekly mentorship sessions

### Communication Channels
- **🎥 Google Meet**: Live workshops and office hours
- **💬 Google Chat**: Daily async check-ins and Q&A
- **📧 Email**: Direct communication with instructors
- **📱 Discord**: Community learning and peer support

### Assessment & Progress
- **Weekly Assessments (40%)**: Technical exercises and knowledge checks
- **Mid-Program Project (25%)**: Week 4 ensemble ML + LangChain project
- **Capstone Project (35%)**: Week 8 comprehensive AI solution

## 🔧 Setup Instructions

### Prerequisites Check
```bash
# Verify Python version
python --version  # Should be 3.8+

# Check available space
df -h  # Should have 5GB+ free

# Test internet connection
curl -I https://api.openai.com  # Should return 200/401
```

### Environment Setup
```bash
# Clone repository
git clone https://github.com/inriva-ai/ai-academy.git
cd ai-academy

# Setup environment (choose one)
# Option 1: Conda (recommended)
conda env create -f setup/environment.yml
conda activate aiml-academy

# Option 2: Virtual environment + pip
python -m venv aiml-academy
source aiml-academy/bin/activate  # On Windows: aiml-academy\Scripts\activate
pip install -r setup/requirements.txt

# Verify setup
python setup/setup_test.py
```

### API Configuration (Optional)
```bash
# Create .env file for API keys
cp setup/.env.template .env
# Edit .env with your API keys

# Test API access
python setup/api_test.py
```

## 📞 Support & Help

### During Program Hours
- **Live Questions**: Unmute during workshops
- **Urgent Technical Issues**: Google Chat #urgent-help
- **General Questions**: Google Chat #general

### Outside Program Hours  
- **Async Questions**: Google Chat or email
- **Technical Issues**: Create GitHub issue
- **Emergency Contact**: [emergency-contact]

### Self-Help Resources
1. **📚 Check relevant primer** in `/resources/primers/`
2. **🔍 Search troubleshooting guide** in `/resources/troubleshooting/`
3. **💡 Review week solutions** in `/weeks/weekN/solutions/`
4. **🤝 Ask peers** in Google Chat or Discord

## 🔗 Additional Resources

### Official Documentation
- [Metaflow Documentation](https://docs.metaflow.org/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama Documentation](https://ollama.com/)

### Learning Materials
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

### Community & Forums
- [Metaflow Community Slack](https://join.slack.com/t/metaflow-community)
- [LangChain Discord](https://discord.gg/langchain)
- [r/MachineLearning](https://reddit.com/r/MachineLearning)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/metaflow+langchain)

## 📈 Program Outcomes

### Technical Skills
- **MLOps Proficiency**: Production ML pipelines with Metaflow
- **LLMOps Expertise**: Advanced LLM applications with LangChain/LangGraph
- **System Integration**: Combining traditional ML with generative AI
- **Production Deployment**: Scalable, monitored AI systems

### Professional Skills  
- **Problem-Solving**: Complex AI challenges across domains
- **Collaboration**: Team-based development and peer learning
- **Communication**: Technical presentations and documentation
- **Project Management**: End-to-end system development

### Career Preparation
- **Portfolio Projects**: Production-ready AI solutions
- **Industry Connections**: Mentorship and networking opportunities
- **Modern Skills**: Latest tools and frameworks used by leading companies
- **Practical Experience**: Real-world problem solving with AI

## 📅 Important Dates

### Program Milestones
- **Week 1**: Environment setup and first Metaflow pipeline
- **Week 2**: LangChain integration and hybrid workflows  
- **Week 4**: Mid-program project and assessment
- **Week 6**: Advanced agent systems and deep learning
- **Week 8**: Capstone project presentations

### Workshops & Events
- **Every Monday 12:00-1:00 PM**: Weekly kickoff
- **Every Wednesday 1:00-2:30 PM**: Technical workshop
- **Every Friday 2:00-3:00 PM**: Showcase and office hours
- **Week 4 Thursday**: Mid-program review (2 hours)
- **Week 8 Friday**: Final presentations (3 hours)

## 🏆 Success Criteria

To successfully complete the program:
- ✅ **Attendance**: Participate in 90%+ of workshops
- ✅ **Weekly Progress**: Complete all weekly deliverables  
- ✅ **Mid-Program Project**: Demonstrate ML + LLM integration
- ✅ **Capstone Project**: Build production-ready AI solution
- ✅ **Technical Proficiency**: Pass all knowledge assessments
- ✅ **Collaboration**: Actively participate in peer learning

## 🎯 Next Steps

### New Participants
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