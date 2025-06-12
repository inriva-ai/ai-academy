# LangChain Primer for AI/ML Interns

## What is LangChain?

LangChain is a powerful framework designed to simplify the development of applications powered by Large Language Models (LLMs). It provides a standardized interface for working with different LLMs and offers tools to build complex, data-aware, and agentic applications.

**Key Benefits for AI/ML Development:**
- **LLM Agnostic**: Switch between OpenAI, Anthropic, local models, and others with minimal code changes
- **Chain Complex Operations**: Connect multiple LLM calls and data processing steps
- **Memory Management**: Handle conversation history and context across interactions
- **Document Processing**: Build RAG (Retrieval-Augmented Generation) systems easily
- **Agent Framework**: Create AI agents that can use tools and make decisions
- **Production Ready**: Built-in monitoring, debugging, and deployment features

**Why LangChain for This Program:**
- Complements traditional ML by adding LLM capabilities
- Essential for modern AI applications combining ML and generative AI
- Industry standard for building production LLM applications
- Bridges the gap between research prototypes and real-world systems

---

## Installation and Setup

### Step 1: Basic Installation

```bash
# Install core LangChain
pip install langchain

# Install specific integrations (choose based on your needs)
pip install langchain-openai        # For OpenAI models
pip install langchain-anthropic     # For Claude models
pip install langchain-community     # Community integrations
pip install langchain-experimental  # Experimental features

# Essential additional packages
pip install langchain-chroma        # Vector database
pip install langchain-text-splitters # Document processing
pip install langsmith               # Monitoring and debugging
```

### Step 2: Environment Setup

Create a `.env` file in your project directory:

```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=your_project_name
```

### Step 3: Development Environment Setup

```python
# setup.py - Run this first to configure your environment
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify setup
def verify_setup():
    """Verify that LangChain is properly configured."""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_community.vectorstores import Chroma
        
        print("✅ LangChain imports successful")
        
        # Test basic LLM connection (with mock if no API key)
        if os.getenv("OPENAI_API_KEY"):
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            print("✅ OpenAI connection configured")
        
        if os.getenv("ANTHROPIC_API_KEY"):
            llm = ChatAnthropic(model="claude-3-sonnet-20240229")
            print("✅ Anthropic connection configured")
        
        print("✅ LangChain setup complete!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

if __name__ == "__main__":
    verify_setup()
```

### Step 4: Optional: Local Model Setup

For development without API costs:

```bash
# Install Ollama for local models
pip install langchain-ollama

# Or use Hugging Face models
pip install langchain-huggingface
```

---

## Basic LangChain Concepts

### 1. Core Components

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# LLMs - The foundation models
class LLMBasics:
    """Demonstrate basic LLM usage in LangChain."""
    
    def __init__(self):
        # Initialize different LLM providers
        self.openai_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        )
        
        # Mock LLM for development without API keys
        self.mock_llm = self.create_mock_llm()
    
    def create_mock_llm(self):
        """Create a mock LLM for development."""
        class MockLLM:
            def invoke(self, messages):
                # Simple mock response based on input
                if isinstance(messages, list):
                    content = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
                else:
                    content = str(messages)
                
                return AIMessage(content=f"Mock response to: {content[:50]}...")
        
        return MockLLM()
    
    def basic_llm_call(self, prompt: str):
        """Make a basic LLM call."""
        messages = [HumanMessage(content=prompt)]
        
        # Use mock LLM for demonstration
        response = self.mock_llm.invoke(messages)
        return response.content
    
    def conversation_example(self):
        """Demonstrate conversation with context."""
        messages = [
            SystemMessage(content="You are a helpful data science tutor."),
            HumanMessage(content="What is the difference between supervised and unsupervised learning?"),
        ]
        
        response = self.mock_llm.invoke(messages)
        print(f"AI: {response.content}")
        
        # Continue conversation
        messages.append(response)
        messages.append(HumanMessage(content="Can you give me an example of each?"))
        
        response2 = self.mock_llm.invoke(messages)
        print(f"AI: {response2.content}")

# Demonstrate basic usage
llm_demo = LLMBasics()
basic_response = llm_demo.basic_llm_call("Explain machine learning in simple terms")
print(f"Basic response: {basic_response}")

llm_demo.conversation_example()
```

### 2. Prompts and Prompt Templates

```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate

class PromptManagement:
    """Demonstrate advanced prompt management."""
    
    def __init__(self):
        self.llm = LLMBasics().mock_llm
    
    def basic_prompt_template(self):
        """Basic prompt templating."""
        template = """
        You are an expert data scientist. Analyze the following dataset description and provide insights.
        
        Dataset: {dataset_name}
        Size: {num_rows} rows, {num_columns} columns
        Domain: {domain}
        
        Provide a brief analysis including:
        1. Potential use cases
        2. Likely challenges
        3. Recommended preprocessing steps
        """
        
        prompt = PromptTemplate(
            input_variables=["dataset_name", "num_rows", "num_columns", "domain"],
            template=template
        )
        
        # Format prompt with specific data
        formatted_prompt = prompt.format(
            dataset_name="Customer Purchase History",
            num_rows=10000,
            num_columns=15,
            domain="E-commerce"
        )
        
        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
        return response.content
    
    def chat_prompt_template(self):
        """Chat-specific prompt templating."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant specializing in {domain}."),
            ("human", "I have a dataset with {description}. What should I do?"),
            ("ai", "Based on the dataset description, I recommend..."),
            ("human", "{follow_up_question}")
        ])
        
        messages = prompt.format_messages(
            domain="machine learning",
            description="customer transaction data with missing values",
            follow_up_question="How do I handle the missing values?"
        )
        
        return messages
    
    def few_shot_prompting(self):
        """Demonstrate few-shot learning with examples."""
        # Examples for the model to learn from
        examples = [
            {
                "input": "Dataset: House prices, 500 rows, 10 features",
                "output": "Use case: Regression for price prediction. Challenge: Small dataset size. Preprocessing: Handle outliers, feature scaling."
            },
            {
                "input": "Dataset: Email spam, 10000 rows, text data",
                "output": "Use case: Text classification. Challenge: Text preprocessing. Preprocessing: Tokenization, TF-IDF vectorization."
            }
        ]
        
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="Input: {input}\nOutput: {output}"
        )
        
        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Analyze datasets and provide structured insights:",
            suffix="Input: {input}\nOutput:",
            input_variables=["input"]
        )
        
        formatted = prompt.format(input="Dataset: Customer reviews, 5000 rows, sentiment labels")
        return formatted

# Demonstrate prompt management
prompt_demo = PromptManagement()
analysis = prompt_demo.basic_prompt_template()
print(f"Dataset analysis: {analysis}")

chat_messages = prompt_demo.chat_prompt_template()
few_shot_example = prompt_demo.few_shot_prompting()
print(f"Few-shot prompt: {few_shot_example}")
```

### 3. Chains - Connecting Operations

```python
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.chains.summarize import load_summarize_chain

class ChainOperations:
    """Demonstrate chain operations for complex workflows."""
    
    def __init__(self):
        self.llm = LLMBasics().mock_llm
    
    def simple_chain(self):
        """Basic chain operation."""
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="Write a brief explanation of {topic} for beginners:"
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Mock chain execution
        result = "Mock explanation of the requested topic for beginners..."
        return result
    
    def sequential_chain_example(self):
        """Chain multiple operations in sequence."""
        
        # Step 1: Generate dataset description
        description_prompt = PromptTemplate(
            input_variables=["domain"],
            template="Describe a realistic dataset for {domain} analysis:"
        )
        description_chain = LLMChain(llm=self.llm, prompt=description_prompt, output_key="description")
        
        # Step 2: Create analysis plan
        analysis_prompt = PromptTemplate(
            input_variables=["description"],
            template="Based on this dataset: {description}\nCreate a step-by-step analysis plan:"
        )
        analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt, output_key="analysis_plan")
        
        # Step 3: Suggest ML models
        model_prompt = PromptTemplate(
            input_variables=["analysis_plan"],
            template="Given this analysis plan: {analysis_plan}\nRecommend appropriate ML models:"
        )
        model_chain = LLMChain(llm=self.llm, prompt=model_prompt, output_key="model_recommendations")
        
        # Combine chains
        overall_chain = SequentialChain(
            chains=[description_chain, analysis_chain, model_chain],
            input_variables=["domain"],
            output_variables=["description", "analysis_plan", "model_recommendations"],
            verbose=True
        )
        
        # Mock execution
        result = {
            "description": "Mock dataset description for the domain",
            "analysis_plan": "Mock step-by-step analysis plan",
            "model_recommendations": "Mock ML model recommendations"
        }
        
        return result

# Demonstrate chains
chain_demo = ChainOperations()
simple_result = chain_demo.simple_chain()
sequential_result = chain_demo.sequential_chain_example()

print(f"Simple chain result: {simple_result}")
print(f"Sequential chain results: {sequential_result}")
```

---

## Document Processing and RAG Systems

### 1. Document Loading and Processing

```python
from langchain_community.document_loaders import TextLoader, PDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class DocumentProcessor:
    """Handle document loading, splitting, and embedding."""
    
    def __init__(self):
        # Use mock embeddings for development
        self.embeddings = self.create_mock_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def create_mock_embeddings(self):
        """Create mock embeddings for development."""
        class MockEmbeddings:
            def embed_documents(self, texts):
                # Return random embeddings for demonstration
                import random
                return [[random.random() for _ in range(1536)] for _ in texts]
            
            def embed_query(self, text):
                import random
                return [random.random() for _ in range(1536)]
        
        return MockEmbeddings()
    
    def create_sample_documents(self):
        """Create sample documents for demonstration."""
        documents = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data.",
            "Neural networks are inspired by biological neurons and consist of interconnected nodes that process information.",
            "Deep learning uses multi-layer neural networks to learn complex patterns in large datasets.",
            "Feature engineering is the process of selecting and transforming variables for machine learning models.",
            "Cross-validation is a technique used to assess model performance and prevent overfitting.",
            "Random forests combine multiple decision trees to improve prediction accuracy and reduce overfitting.",
            "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models."
        ]
        
        # Convert to Document objects
        from langchain.schema import Document
        return [Document(page_content=text, metadata={"source": f"doc_{i}"}) for i, text in enumerate(documents)]
    
    def process_documents(self, documents):
        """Process documents into chunks and embeddings."""
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        # Create embeddings and vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        return vectorstore
    
    def load_csv_data(self):
        """Example of loading and processing CSV data."""
        # Create sample CSV content
        import pandas as pd
        import tempfile
        import os
        
        # Create sample data
        data = {
            'question': [
                'What is machine learning?',
                'How does supervised learning work?',
                'What are neural networks?',
                'What is deep learning?'
            ],
            'answer': [
                'Machine learning is a subset of AI that learns from data.',
                'Supervised learning uses labeled examples to train models.',
                'Neural networks are computational models inspired by the brain.',
                'Deep learning uses multi-layer neural networks for complex tasks.'
            ],
            'category': ['basics', 'basics', 'advanced', 'advanced']
        }
        
        df = pd.DataFrame(data)
        
        # Save to temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_file = f.name
        
        try:
            # Load CSV using LangChain
            loader = CSVLoader(file_path=csv_file)
            documents = loader.load()
            
            print(f"Loaded {len(documents)} documents from CSV")
            return documents
        
        finally:
            # Clean up temporary file
            os.unlink(csv_file)

# Demonstrate document processing
doc_processor = DocumentProcessor()
sample_docs = doc_processor.create_sample_documents()
vectorstore = doc_processor.process_documents(sample_docs)
csv_docs = doc_processor.load_csv_data()

print("Document processing completed")
```

### 2. Building RAG (Retrieval-Augmented Generation) Systems

```python
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

class RAGSystem:
    """Implement Retrieval-Augmented Generation system."""
    
    def __init__(self, vectorstore=None):
        self.llm = LLMBasics().mock_llm
        self.vectorstore = vectorstore or self.create_mock_vectorstore()
    
    def create_mock_vectorstore(self):
        """Create a mock vector store for development."""
        class MockVectorStore:
            def __init__(self):
                self.documents = [
                    "Machine learning algorithms learn patterns from data to make predictions.",
                    "Supervised learning requires labeled training data with input-output pairs.",
                    "Unsupervised learning finds hidden patterns in data without labels.",
                    "Deep learning uses neural networks with multiple hidden layers.",
                    "Feature engineering involves creating meaningful variables for models."
                ]
            
            def similarity_search(self, query, k=4):
                from langchain.schema import Document
                # Return mock relevant documents
                return [Document(page_content=doc, metadata={"score": 0.8}) 
                       for doc in self.documents[:k]]
            
            def as_retriever(self, **kwargs):
                return MockRetriever(self)
        
        class MockRetriever:
            def __init__(self, vectorstore):
                self.vectorstore = vectorstore
            
            def get_relevant_documents(self, query):
                return self.vectorstore.similarity_search(query)
        
        return MockVectorStore()
    
    def basic_rag_query(self, question: str):
        """Perform basic RAG query."""
        # Retrieve relevant documents
        relevant_docs = self.vectorstore.similarity_search(question, k=3)
        
        # Combine documents for context
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt with context
        prompt = f"""
        Based on the following context, answer the question.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        # Mock response
        response = f"Based on the provided context, here's what I know about {question}: [Mock RAG response based on retrieved documents]"
        
        return {
            "question": question,
            "answer": response,
            "source_documents": relevant_docs
        }
    
    def advanced_rag_chain(self):
        """Create an advanced RAG chain with custom processing."""
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Mock QA chain
        class MockQAChain:
            def __init__(self, llm, retriever):
                self.llm = llm
                self.retriever = retriever
            
            def run(self, query):
                docs = self.retriever.get_relevant_documents(query)
                context = "\n".join([doc.page_content for doc in docs])
                return f"Mock answer based on {len(docs)} retrieved documents about: {query}"
        
        qa_chain = MockQAChain(self.llm, retriever)
        
        return qa_chain
    
    def multi_query_rag(self, question: str):
        """Use multiple query variations for better retrieval."""
        
        # Generate query variations
        query_variations = [
            question,
            f"What is {question}?",
            f"How does {question} work?",
            f"Explain {question} in simple terms"
        ]
        
        all_docs = []
        for query in query_variations:
            docs = self.vectorstore.similarity_search(query, k=2)
            all_docs.extend(docs)
        
        # Remove duplicates and combine
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        
        context = "\n".join([doc.page_content for doc in unique_docs[:5]])
        
        response = f"Multi-query RAG response using {len(unique_docs)} unique documents for: {question}"
        
        return {
            "question": question,
            "query_variations": query_variations,
            "answer": response,
            "documents_used": len(unique_docs)
        }

# Demonstrate RAG system
rag_system = RAGSystem()

basic_result = rag_system.basic_rag_query("What is machine learning?")
print(f"Basic RAG: {basic_result['answer']}")

advanced_chain = rag_system.advanced_rag_chain()
advanced_result = advanced_chain.run("supervised learning")
print(f"Advanced RAG: {advanced_result}")

multi_query_result = rag_system.multi_query_rag("neural networks")
print(f"Multi-query RAG: {multi_query_result}")
```

---

## Memory and Conversation Management

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

class ConversationManager:
    """Manage conversation memory and context."""
    
    def __init__(self):
        self.llm = LLMBasics().mock_llm
        self.setup_memory_types()
    
    def setup_memory_types(self):
        """Initialize different memory types."""
        
        # Buffer memory - stores all messages
        self.buffer_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Window memory - stores last N messages
        self.window_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,  # Keep last 5 exchanges
            return_messages=True
        )
        
        # Summary memory - summarizes old conversations
        self.summary_memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True
        )
    
    def buffer_memory_example(self):
        """Demonstrate buffer memory usage."""
        conversation = [
            ("Human", "What is machine learning?"),
            ("AI", "Machine learning is a subset of AI that learns from data."),
            ("Human", "Can you give me an example?"),
            ("AI", "Sure! Email spam detection learns from examples of spam and legitimate emails."),
            ("Human", "What about supervised vs unsupervised learning?"),
            ("AI", "Supervised learning uses labeled data, unsupervised finds patterns without labels.")
        ]
        
        # Add messages to memory
        for role, content in conversation:
            if role == "Human":
                self.buffer_memory.chat_memory.add_user_message(content)
            else:
                self.buffer_memory.chat_memory.add_ai_message(content)
        
        # Retrieve conversation history
        history = self.buffer_memory.load_memory_variables({})
        return history
    
    def window_memory_example(self):
        """Demonstrate window memory for long conversations."""
        
        # Simulate a long conversation
        long_conversation = [
            ("Human", "Tell me about data science"),
            ("AI", "Data science combines statistics, programming, and domain expertise."),
            ("Human", "What programming languages are used?"),
            ("AI", "Python and R are the most popular for data science."),
            ("Human", "What about databases?"),
            ("AI", "SQL databases store structured data, NoSQL handles unstructured data."),
            ("Human", "How do I learn machine learning?"),
            ("AI", "Start with basic statistics, then learn Python and practice with datasets."),
            ("Human", "What's the difference between ML and AI?"),  # This will be remembered
            ("AI", "AI is broader, ML is a subset focused on learning from data."),  # This will be remembered
        ]
        
        # Add all messages
        for role, content in long_conversation:
            if role == "Human":
                self.window_memory.chat_memory.add_user_message(content)
            else:
                self.window_memory.chat_memory.add_ai_message(content)
        
        # Window memory only keeps last 5 exchanges (10 messages)
        history = self.window_memory.load_memory_variables({})
        return history
    
    def conversation_with_memory_chain(self):
        """Create a conversation chain with memory."""
        from langchain.chains import ConversationChain
        
        # Mock conversation chain
        class MockConversationChain:
            def __init__(self, llm, memory):
                self.llm = llm
                self.memory = memory
            
            def predict(self, input_text):
                # Get conversation history
                history = self.memory.load_memory_variables({})
                
                # Add current input to memory
                self.memory.chat_memory.add_user_message(input_text)
                
                # Generate response (mock)
                response = f"Mock response to '{input_text}' considering conversation history"
                
                # Add response to memory
                self.memory.chat_memory.add_ai_message(response)
                
                return response
        
        conversation = MockConversationChain(
            llm=self.llm,
            memory=self.buffer_memory
        )
        
        # Simulate conversation
        responses = []
        questions = [
            "What is the difference between classification and regression?",
            "Can you give me an example of classification?",
            "What about regression examples?",
            "How do I choose between them?"
        ]
        
        for question in questions:
            response = conversation.predict(question)
            responses.append({"question": question, "response": response})
        
        return responses

# Demonstrate conversation management
conv_manager = ConversationManager()

buffer_history = conv_manager.buffer_memory_example()
print(f"Buffer memory conversations: {len(buffer_history['chat_history'])} messages")

window_history = conv_manager.window_memory_example()
print(f"Window memory conversations: {len(window_history['chat_history'])} messages")

conversation_responses = conv_manager.conversation_with_memory_chain()
print(f"Conversation chain: {len(conversation_responses)} exchanges")
```

---

## Advanced Examples

### 1. Building an AI Data Analysis Agent

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from typing import Optional
import pandas as pd
import numpy as np

class DataAnalysisTool(BaseTool):
    """Custom tool for data analysis operations."""
    
    name = "data_analyzer"
    description = "Analyzes datasets and provides statistical insights"
    
    def _run(self, query: str) -> str:
        """Execute data analysis based on query."""
        
        # Mock dataset for demonstration
        np.random.seed(42)
        data = {
            'sales': np.random.normal(1000, 200, 100),
            'profit': np.random.normal(200, 50, 100),
            'customers': np.random.poisson(50, 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
        }
        df = pd.DataFrame(data)
        
        # Parse query and perform analysis
        if "summary" in query.lower():
            return f"Dataset summary: {df.describe().to_string()}"
        elif "correlation" in query.lower():
            return f"Correlations: {df.corr().to_string()}"
        elif "group" in query.lower() or "region" in query.lower():
            return f"Group analysis: {df.groupby('region').mean().to_string()}"
        else:
            return f"Basic info: Shape {df.shape}, Columns: {list(df.columns)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of the tool."""
        return self._run(query)

class MLModelTool(BaseTool):
    """Tool for ML model operations."""
    
    name = "ml_model"
    description = "Trains and evaluates machine learning models"
    
    def _run(self, query: str) -> str:
        """Execute ML operations."""
        
        if "train" in query.lower():
            return "Mock: Trained a Random Forest model with 85% accuracy"
        elif "predict" in query.lower():
            return "Mock: Generated predictions for the dataset"
        elif "evaluate" in query.lower():
            return "Mock: Model evaluation - Accuracy: 85%, Precision: 82%, Recall: 88%"
        else:
            return "Available ML operations: train, predict, evaluate"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class DataAnalysisAgent:
    """AI agent for comprehensive data analysis."""
    
    def __init__(self):
        self.llm = LLMBasics().mock_llm
        self.tools = [
            DataAnalysisTool(),
            MLModelTool(),
        ]
        self.setup_agent()
    
    def setup_agent(self):
        """Set up the agent with tools."""
        
        # Mock agent for demonstration
        class MockAgent:
            def __init__(self, tools, llm):
                self.tools = {tool.name: tool for tool in tools}
                self.llm = llm
            
            def run(self, input_text):
                # Simple tool selection logic
                if "analyze" in input_text.lower() or "data" in input_text.lower():
                    tool_result = self.tools["data_analyzer"]._run(input_text)
                    return f"Data Analysis Result: {tool_result[:200]}..."
                elif "model" in input_text.lower() or "train" in input_text.lower():
                    tool_result = self.tools["ml_model"]._run(input_text)
                    return f"ML Model Result: {tool_result}"
                else:
                    return "I can help with data analysis and ML modeling. What would you like to do?"
        
        self.agent = MockAgent(self.tools, self.llm)
    
    def analyze_dataset(self, request: str):
        """Analyze dataset based on natural language request."""
        return self.agent.run(request)
    
    def complete_analysis_workflow(self):
        """Demonstrate complete analysis workflow."""
        workflow_steps = [
            "Analyze the dataset and provide a summary",
            "Show correlation between variables",
            "Group data by region and analyze",
            "Train a machine learning model",
            "Evaluate the model performance"
        ]
        
        results = []
        for step in workflow_steps:
            result = self.agent.run(step)
            results.append({"step": step, "result": result})
        
        return results

# Demonstrate data analysis agent
agent = DataAnalysisAgent()

# Single analysis
single_result = agent.analyze_dataset("Analyze the sales data and show summary statistics")
print(f"Single analysis: {single_result}")

# Complete workflow
workflow_results = agent.complete_analysis_workflow()
for i, step_result in enumerate(workflow_results):
    print(f"Step {i+1}: {step_result['result']}")
```

### 2. Custom Chain for Research and Analysis

```python
from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun
from typing import Dict, Any, List

class ResearchAnalysisChain(Chain):
    """Custom chain for research and analysis workflows."""
    
    llm: Any
    vectorstore: Any
    
    @property
    def input_keys(self) -> List[str]:
        return ["research_topic", "analysis_depth"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["research_summary", "key_findings", "recommendations", "sources"]
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute the research and analysis chain."""
        
        topic = inputs["research_topic"]
        depth = inputs.get("analysis_depth", "standard")
        
        # Step 1: Research phase
        research_queries = [
            f"What is {topic}?",
            f"Current trends in {topic}",
            f"Applications of {topic}",
            f"Challenges in {topic}"
        ]
        
        research_results = []
        for query in research_queries:
            # Mock research using vectorstore
            docs = self.vectorstore.similarity_search(query, k=3)
            research_results.extend([doc.page_content for doc in docs])
        
        # Step 2: Analysis phase
        combined_research = "\n".join(research_results)
        
        # Mock analysis
        key_findings = [
            f"Finding 1: {topic} is an important area with growing applications",
            f"Finding 2: Current challenges include scalability and implementation",
            f"Finding 3: Future trends point toward increased automation"
        ]
        
        recommendations = [
            f"Recommendation 1: Focus on practical applications of {topic}",
            f"Recommendation 2: Address current limitations through research",
            f"Recommendation 3: Prepare for future developments"
        ]
        
        research_summary = f"Comprehensive analysis of {topic} based on current research and trends."
        
        return {
            "research_summary": research_summary,
            "key_findings": key_findings,
            "recommendations": recommendations,
            "sources": [f"Source {i+1}" for i in range(len(research_results))]
        }

class AdvancedWorkflowManager:
    """Manage complex AI workflows combining multiple capabilities."""
    
    def __init__(self):
        self.llm = LLMBasics().mock_llm
        self.vectorstore = RAGSystem().vectorstore
        self.research_chain = ResearchAnalysisChain(
            llm=self.llm,
            vectorstore=self.vectorstore
        )
    
    def research_and_analyze(self, topic: str):
        """Perform comprehensive research and analysis."""
        
        result = self.research_chain({
            "research_topic": topic,
            "analysis_depth": "comprehensive"
        })
        
        return result
    
    def comparative_analysis(self, topics: List[str]):
        """Compare multiple topics using research chain."""
        
        comparisons = {}
        for topic in topics:
            analysis = self.research_chain({
                "research_topic": topic,
                "analysis_depth": "standard"
            })
            comparisons[topic] = analysis
        
        # Generate comparison summary
        comparison_summary = {
            "topics_analyzed": topics,
            "common_themes": "Mock common themes across topics",
            "key_differences": "Mock key differences between topics",
            "individual_analyses": comparisons
        }
        
        return comparison_summary
    
    def end_to_end_ml_workflow(self, problem_description: str):
        """Complete ML workflow from problem to solution."""
        
        workflow_steps = {
            "problem_analysis": f"Analyzed problem: {problem_description}",
            "data_requirements": "Identified required data sources and features",
            "methodology": "Recommended ML approaches and algorithms",
            "implementation_plan": "Created step-by-step implementation plan",
            "evaluation_strategy": "Defined metrics and validation approach"
        }
        
        return workflow_steps

# Demonstrate advanced workflows
workflow_manager = AdvancedWorkflowManager()

# Research analysis
research_result = workflow_manager.research_and_analyze("deep learning")
print(f"Research analysis: {research_result['research_summary']}")

# Comparative analysis
comparison_result = workflow_manager.comparative_analysis(["machine learning", "deep learning", "reinforcement learning"])
print(f"Comparative analysis: {comparison_result['common_themes']}")

# End-to-end ML workflow
ml_workflow = workflow_manager.end_to_end_ml_workflow("Predict customer churn for e-commerce company")
print(f"ML workflow: {ml_workflow}")
```

---

## Integration with Traditional ML

### Combining LangChain with Scikit-learn

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

class MLLangChainIntegration:
    """Integrate traditional ML with LangChain for enhanced workflows."""
    
    def __init__(self):
        self.llm = LLMBasics().mock_llm
        self.trained_models = {}
    
    def create_sample_ml_dataset(self):
        """Create sample dataset for ML integration demo."""
        np.random.seed(42)
        
        # Generate synthetic customer data
        n_samples = 1000
        
        data = {
            'age': np.random.normal(35, 10, n_samples),
            'income': np.random.normal(50000, 15000, n_samples),
            'spending': np.random.normal(2000, 500, n_samples),
            'tenure': np.random.exponential(3, n_samples),
            'support_calls': np.random.poisson(2, n_samples)
        }
        
        # Create target variable (churn) with logical relationships
        churn_probability = (
            0.1 +
            0.3 * (data['support_calls'] > 3).astype(int) +
            0.2 * (data['spending'] < 1500).astype(int) +
            0.1 * (data['tenure'] < 1).astype(int)
        )
        
        data['churn'] = np.random.binomial(1, churn_probability, n_samples)
        
        return pd.DataFrame(data)
    
    def train_ml_model_with_llm_insights(self, df):
        """Train ML model and get LLM insights about the process."""
        
        # Prepare data
        features = ['age', 'income', 'spending', 'tenure', 'support_calls']
        X = df[features]
        y = df['churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get predictions and metrics
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        
        # Get feature importance
        feature_importance = dict(zip(features, model.feature_importances_))
        
        # Use LLM to interpret results
        interpretation_prompt = f"""
        I trained a Random Forest model to predict customer churn with the following results:
        
        Accuracy: {accuracy:.3f}
        Feature Importance: {feature_importance}
        
        Please provide insights about:
        1. Model performance interpretation
        2. Feature importance analysis
        3. Business recommendations
        """
        
        # Mock LLM interpretation
        llm_insights = f"""
        Mock LLM Insights:
        1. Model Performance: {accuracy:.1%} accuracy indicates good predictive capability
        2. Feature Analysis: Most important features suggest customer service and spending patterns drive churn
        3. Business Recommendations: Focus on improving customer support and engagement strategies
        """
        
        # Store model
        self.trained_models['churn_prediction'] = model
        
        return {
            'model': model,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'llm_insights': llm_insights,
            'test_predictions': y_pred
        }
    
    def explain_predictions_with_llm(self, model_name: str, sample_data: dict):
        """Use LLM to explain individual predictions."""
        
        if model_name not in self.trained_models:
            return "Model not found"
        
        model = self.trained_models[model_name]
        
        # Make prediction
        features = np.array(list(sample_data.values())).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Get feature importance for this model
        feature_names = list(sample_data.keys())
        importance = dict(zip(feature_names, model.feature_importances_))
        
        # Create explanation prompt
        explanation_prompt = f"""
        Customer Profile: {sample_data}
        Prediction: {'Will Churn' if prediction == 1 else 'Will Not Churn'}
        Probability: {probability[1]:.3f} (churn), {probability[0]:.3f} (no churn)
        
        Feature Importance in Model: {importance}
        
        Explain this prediction in business terms.
        """
        
        # Mock explanation
        explanation = f"""
        Mock Explanation: Based on the customer profile, the model predicts {'churn' if prediction == 1 else 'retention'} 
        with {max(probability):.1%} confidence. Key factors influencing this prediction include the customer's 
        support call history and spending patterns.
        """
        
        return {
            'prediction': prediction,
            'probability': probability.tolist(),
            'explanation': explanation,
            'customer_profile': sample_data
        }
    
    def generate_ml_report(self, model_results: dict):
        """Generate comprehensive ML report using LLM."""
        
        report_prompt = f"""
        Generate a comprehensive machine learning model report based on:
        
        Model Type: Random Forest Classifier
        Accuracy: {model_results['accuracy']:.3f}
        Feature Importance: {model_results['feature_importance']}
        
        Include:
        1. Executive Summary
        2. Technical Details
        3. Business Impact
        4. Recommendations for Improvement
        5. Next Steps
        """
        
        # Mock comprehensive report
        report = f"""
        MACHINE LEARNING MODEL REPORT
        ============================
        
        Executive Summary:
        The churn prediction model achieves {model_results['accuracy']:.1%} accuracy, providing reliable 
        customer retention insights for business decision-making.
        
        Technical Details:
        - Algorithm: Random Forest Classifier
        - Features: {len(model_results['feature_importance'])} customer attributes
        - Performance: {model_results['accuracy']:.1%} accuracy on test set
        
        Business Impact:
        - Enables proactive customer retention strategies
        - Identifies at-risk customers for targeted interventions
        - Potential ROI through reduced churn rates
        
        Recommendations:
        1. Deploy model for real-time churn scoring
        2. Integrate with customer service systems
        3. Develop targeted retention campaigns
        
        Next Steps:
        - Monitor model performance in production
        - Collect additional features for improvement
        - Regular model retraining schedule
        """
        
        return report

# Demonstrate ML-LangChain integration
ml_integration = MLLangChainIntegration()

# Create dataset and train model
df = ml_integration.create_sample_ml_dataset()
model_results = ml_integration.train_ml_model_with_llm_insights(df)

print("ML Model Training Results:")
print(f"Accuracy: {model_results['accuracy']:.3f}")
print(f"LLM Insights: {model_results['llm_insights']}")

# Explain individual prediction
sample_customer = {
    'age': 45,
    'income': 60000,
    'spending': 1200,
    'tenure': 0.5,
    'support_calls': 5
}

prediction_explanation = ml_integration.explain_predictions_with_llm('churn_prediction', sample_customer)
print(f"\nPrediction Explanation: {prediction_explanation['explanation']}")

# Generate comprehensive report
ml_report = ml_integration.generate_ml_report(model_results)
print(f"\nML Report:\n{ml_report}")
```

---

## Best Practices and Production Considerations

### 1. Error Handling and Monitoring

```python
import logging
from functools import wraps
import time
from typing import Dict, Any

class LangChainProductionManager:
    """Best practices for production LangChain applications."""
    
    def __init__(self):
        self.setup_logging()
        self.setup_monitoring()
    
    def setup_logging(self):
        """Configure logging for LangChain applications."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('langchain_app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_monitoring(self):
        """Set up monitoring and metrics collection."""
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'errors': []
        }
    
    def error_handler(self, max_retries=3, backoff_factor=2):
        """Decorator for handling LLM API errors with retry logic."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        
                        # Update metrics
                        response_time = time.time() - start_time
                        self.update_metrics(success=True, response_time=response_time)
                        
                        return result
                        
                    except Exception as e:
                        self.logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                        self.update_metrics(success=False, error=str(e))
                        
                        if attempt == max_retries - 1:
                            raise e
                        
                        # Exponential backoff
                        time.sleep(backoff_factor ** attempt)
                
            return wrapper
        return decorator
    
    def update_metrics(self, success: bool, response_time: float = 0, error: str = None):
        """Update application metrics."""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
            # Update rolling average
            current_avg = self.metrics['average_response_time']
            total_successful = self.metrics['successful_requests']
            self.metrics['average_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
        else:
            self.metrics['failed_requests'] += 1
            if error:
                self.metrics['errors'].append({
                    'timestamp': time.time(),
                    'error': error
                })
    
    @error_handler(max_retries=3)
    def safe_llm_call(self, prompt: str, **kwargs):
        """Make LLM call with error handling."""
        # Mock LLM call that might fail
        import random
        
        if random.random() < 0.1:  # 10% failure rate for demo
            raise Exception("Mock API error")
        
        return f"Mock response to: {prompt[:50]}..."
    
    def validate_inputs(self, inputs: Dict[str, Any], required_fields: list):
        """Validate inputs before processing."""
        missing_fields = [field for field in required_fields if field not in inputs]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Additional validation
        for field, value in inputs.items():
            if isinstance(value, str) and len(value.strip()) == 0:
                raise ValueError(f"Field '{field}' cannot be empty")
        
        return True
    
    def rate_limit_check(self, requests_per_minute=60):
        """Simple rate limiting implementation."""
        current_time = time.time()
        
        # Simple in-memory rate limiting (use Redis in production)
        if not hasattr(self, 'request_times'):
            self.request_times = []
        
        # Remove old requests (older than 1 minute)
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) >= requests_per_minute:
            raise Exception("Rate limit exceeded")
        
        self.request_times.append(current_time)
        return True
    
    def get_metrics_report(self):
        """Generate metrics report."""
        total = self.metrics['total_requests']
        success_rate = (self.metrics['successful_requests'] / total) if total > 0 else 0
        
        return {
            'total_requests': total,
            'success_rate': success_rate,
            'average_response_time': self.metrics['average_response_time'],
            'recent_errors': self.metrics['errors'][-10:],  # Last 10 errors
            'uptime_status': 'healthy' if success_rate > 0.95 else 'degraded'
        }

# Demonstrate production best practices
prod_manager = LangChainProductionManager()

# Test error handling
for i in range(10):
    try:
        result = prod_manager.safe_llm_call(f"Test prompt {i}")
        print(f"Request {i}: Success")
    except Exception as e:
        print(f"Request {i}: Failed - {e}")

# Show metrics
metrics = prod_manager.get_metrics_report()
print(f"\nMetrics Report: {metrics}")
```

### 2. Configuration Management

```python
import os
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class LangChainConfig:
    """Configuration for LangChain applications."""
    
    # LLM Settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # Vector Store Settings
    vector_store_type: str = "chroma"
    vector_store_path: str = "./vector_store"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Application Settings
    max_retries: int = 3
    timeout: int = 30
    rate_limit: int = 60
    debug_mode: bool = False
    
    # Monitoring
    enable_monitoring: bool = True
    log_level: str = "INFO"
    metrics_interval: int = 300  # 5 minutes

class ConfigManager:
    """Manage application configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> LangChainConfig:
        """Load configuration from file and environment."""
        
        # Default configuration
        config_dict = {}
        
        # Load from YAML file if it exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                config_dict.update(file_config)
        
        # Override with environment variables
        env_mapping = {
            'OPENAI_API_KEY': 'openai_api_key',
            'ANTHROPIC_API_KEY': 'anthropic_api_key',
            'MODEL_NAME': 'model_name',
            'TEMPERATURE': 'temperature',
            'MAX_TOKENS': 'max_tokens',
            'VECTOR_STORE_PATH': 'vector_store_path',
            'DEBUG_MODE': 'debug_mode',
            'LOG_LEVEL': 'log_level'
        }
        
        for env_var, config_key in env_mapping.items():
            if os.getenv(env_var):
                value = os.getenv(env_var)
                # Convert string values to appropriate types
                if config_key in ['temperature']:
                    value = float(value)
                elif config_key in ['max_tokens', 'chunk_size', 'chunk_overlap', 'max_retries', 'timeout', 'rate_limit']:
                    value = int(value)
                elif config_key in ['debug_mode', 'enable_monitoring']:
                    value = value.lower() in ['true', '1', 'yes']
                
                config_dict[config_key] = value
        
        return LangChainConfig(**config_dict)
    
    def save_config(self):
        """Save current configuration to file."""
        config_dict = {
            'model_name': self.config.model_name,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'vector_store_type': self.config.vector_store_type,
            'vector_store_path': self.config.vector_store_path,
            'chunk_size': self.config.chunk_size,
            'chunk_overlap': self.config.chunk_overlap,
            'max_retries': self.config.max_retries,
            'timeout': self.config.timeout,
            'rate_limit': self.config.rate_limit,
            'debug_mode': self.config.debug_mode,
            'enable_monitoring': self.config.enable_monitoring,
            'log_level': self.config.log_level
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def update_config(self, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.save_config()

# Example configuration usage
config_manager = ConfigManager()
print(f"Current config: {config_manager.config}")
```

---

## Integration with Your Internship Program

### Week-by-Week LangChain Integration

#### **Week 4: Introduction to LangChain**
- Install and configure LangChain
- Basic LLM interactions and prompt templates
- Simple chains for text processing
- Compare with direct API usage

#### **Week 5: Document Processing and RAG**
- Build document processing pipelines
- Create vector stores for company/project data
- Implement basic RAG for question-answering
- Combine with traditional ML for enhanced insights

#### **Week 6: Advanced Chains and Memory**
- Implement conversation management
- Build complex sequential chains
- Create custom tools for data analysis
- Memory management for long conversations

#### **Week 7: Agents and Production**
- Build AI agents that use ML models
- Implement error handling and monitoring
- Production deployment considerations
- Ethics and safety in LLM applications

#### **Week 8: Capstone Integration**
- Use LangChain in capstone projects
- Combine traditional ML with LLM capabilities
- Build end-to-end applications
- Present complete AI solutions

### Project Ideas Using LangChain

1. **Intelligent Data Analysis Assistant**
   - RAG system with company data
   - Natural language querying of datasets
   - Automated insight generation

2. **ML Model Explainer**
   - LLM explanations of model predictions
   - Natural language model documentation
   - Interactive model exploration

3. **Research Assistant**
   - Multi-source document analysis
   - Automated literature review
   - Hypothesis generation from data

4. **Customer Service Agent**
   - RAG with product documentation
   - Integration with existing ML models
   - Conversation memory and context

---

## Additional Resources

### Learning Resources
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [LangSmith for Monitoring](https://smith.langchain.com/)
- [LangChain Academy](https://academy.langchain.com/)

### Community and Support
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [Discord Community](https://discord.gg/langchain)
- [Twitter: @LangChainAI](https://twitter.com/langchainai)

### Advanced Topics
- [Custom Tool Development](https://python.langchain.com/docs/modules/agents/tools/custom_tools)
- [Production Deployment](https://python.langchain.com/docs/guides/productionization/)
- [LangGraph for Complex Workflows](https://langchain-ai.github.io/langgraph/)

LangChain bridges the gap between traditional ML and modern LLM applications, making it an essential tool for building sophisticated AI systems. Start with basic concepts and gradually build up to complex agents and production deployments!