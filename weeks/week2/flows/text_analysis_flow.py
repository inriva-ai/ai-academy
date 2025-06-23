"""
Text Analysis Flow with LLM Integration - Metaflow Implementation
================================================================

A comprehensive text processing pipeline using Metaflow for orchestration
and LLM integration for advanced analysis.

Usage:
    python text_analysis_flow.py run
    python text_analysis_flow.py run --text "Your text here"
    python text_analysis_flow.py run --input_file data/documents.csv
    python text_analysis_flow.py show
"""

from metaflow import FlowSpec, step, Parameter, current, catch, retry
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np

# NLP Libraries
try:
    import spacy
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from textstat import flesch_reading_ease, flesch_kincaid_grade
except ImportError as e:
    print(f"Warning: Some NLP libraries not available: {e}")

# LLM Libraries
try:
    import openai
except ImportError:
    print("Warning: OpenAI library not available")

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class TextType(Enum):
    """Types of text content"""
    ARTICLE = "article"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    ACADEMIC = "academic"
    LEGAL = "legal"
    TECHNICAL = "technical"
    GENERAL = "general"

@dataclass
class TextMetrics:
    """Text readability and complexity metrics"""
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_words_per_sentence: float = 0.0
    flesch_score: float = 0.0
    flesch_kincaid_grade: float = 0.0
    complexity_level: str = "unknown"

@dataclass
class AnalysisResult:
    """Results from text analysis"""
    text_id: str
    original_text: str
    processed_text: str
    text_type: str
    sentiment: Dict[str, float] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    metrics: Optional[Dict] = None
    llm_analysis: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class TextAnalysisFlow(FlowSpec):
    """
    A comprehensive text analysis workflow using Metaflow.
    
    This flow processes text through multiple stages:
    1. Data loading and preprocessing
    2. Traditional NLP analysis (sentiment, entities, metrics)
    3. LLM-powered analysis and insights
    4. Results aggregation and export
    """
    
    # Flow Parameters
    text = Parameter(
        'text',
        help='Single text to analyze',
        default=None
    )
    
    input_file = Parameter(
        'input_file',
        help='CSV file with texts to analyze (should have "text" column)',
        default=None
    )
    
    output_format = Parameter(
        'output_format',
        help='Output format: json, csv, or both',
        default='json'
    )
    
    llm_model = Parameter(
        'llm_model',
        help='LLM model to use for analysis',
        default='gpt-3.5-turbo'
    )
    
    enable_llm = Parameter(
        'enable_llm',
        help='Enable LLM analysis (requires API key)',
        default=False,
        type=bool
    )
    
    max_texts = Parameter(
        'max_texts',
        help='Maximum number of texts to process',
        default=100,
        type=int
    )
    
    spacy_model = Parameter(
        'spacy_model',
        help='spaCy model for NLP processing',
        default='en_core_web_sm'
    )

    @step
    def start(self):
        """
        Initialize the text analysis workflow and prepare data
        """
        print("🚀 Starting Text Analysis Flow")
        print(f"Run ID: {current.run_id}")
        print(f"Flow name: {current.flow_name}")
        
        # Initialize flow metadata
        self.flow_start_time = datetime.now()
        self.total_texts_processed = 0
        self.errors_encountered = []
        
        # Prepare input texts
        self.texts_to_process = []
        
        if self.text:
            # Single text provided
            self.texts_to_process = [
                {"id": "single_text", "text": self.text}
            ]
            print(f"Processing single text: {self.text[:100]}...")
            
        elif self.input_file:
            # Load from file
            try:
                df = pd.read_csv(self.input_file)
                if 'text' not in df.columns:
                    raise ValueError("Input CSV must have a 'text' column")
                
                # Add IDs if not present
                if 'id' not in df.columns:
                    df['id'] = df.index.astype(str)
                
                # Convert to list of dicts and limit
                self.texts_to_process = df[['id', 'text']].head(self.max_texts).to_dict('records')
                print(f"Loaded {len(self.texts_to_process)} texts from {self.input_file}")
                
            except Exception as e:
                error_msg = f"Error loading input file: {e}"
                self.errors_encountered.append(error_msg)
                print(f"❌ {error_msg}")
                self.texts_to_process = []
        else:
            # Use sample texts
            sample_texts = [
                {
                    "id": "sample_1",
                    "text": """
                    Artificial intelligence has revolutionized the way we process and analyze text data.
                    Modern NLP techniques, combined with large language models, enable sophisticated
                    understanding of human language. This technology has applications in sentiment analysis,
                    content generation, and automated summarization.
                    """
                },
                {
                    "id": "sample_2", 
                    "text": """
                    Dear valued customer,
                    
                    We are writing to inform you about an important update to our service terms.
                    Please review the attached documentation carefully. If you have any questions,
                    don't hesitate to contact our support team.
                    
                    Best regards,
                    Customer Service Team
                    """
                }
            ]
            self.texts_to_process = sample_texts
            print("Using sample texts for demonstration")
        
        print(f"Total texts to process: {len(self.texts_to_process)}")
        
        # Store as artifact for debugging
        self.input_summary = {
            "total_texts": len(self.texts_to_process),
            "source": "parameter" if self.text else "file" if self.input_file else "sample",
            "enable_llm": self.enable_llm,
            "model": self.llm_model
        }
        
        self.next(self.setup_nlp_models)

    @catch(var='setup_errors')
    @step
    def setup_nlp_models(self):
        """
        Initialize NLP models and tools
        """
        print("🔧 Setting up NLP models...")
        
        self.nlp_models = {}
        self.setup_errors = []
        
        # Initialize spaCy model
        try:
            import spacy
            self.nlp_models['spacy'] = spacy.load(self.spacy_model)
            print(f"✅ Loaded spaCy model: {self.spacy_model}")
        except Exception as e:
            error_msg = f"Could not load spaCy model {self.spacy_model}: {e}"
            self.setup_errors.append(error_msg)
            print(f"⚠️ {error_msg}")
            self.nlp_models['spacy'] = None
        
        # Initialize VADER sentiment analyzer
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.nlp_models['sentiment'] = SentimentIntensityAnalyzer()
            print("✅ Initialized VADER sentiment analyzer")
        except Exception as e:
            error_msg = f"Could not initialize sentiment analyzer: {e}"
            self.setup_errors.append(error_msg)
            print(f"⚠️ {error_msg}")
            self.nlp_models['sentiment'] = None
        
        # Initialize LLM client if enabled
        if self.enable_llm:
            try:
                import openai
                self.nlp_models['llm_client'] = openai.OpenAI()
                print("✅ Initialized OpenAI client")
            except Exception as e:
                error_msg = f"Could not initialize LLM client: {e}"
                self.setup_errors.append(error_msg)
                print(f"⚠️ {error_msg}")
                self.nlp_models['llm_client'] = None
        
        self.model_status = {
            'spacy_available': self.nlp_models['spacy'] is not None,
            'sentiment_available': self.nlp_models['sentiment'] is not None,
            'llm_available': self.enable_llm and self.nlp_models.get('llm_client') is not None
        }
        
        print(f"Model status: {self.model_status}")
        
        self.next(self.preprocess_texts)

    @step
    def preprocess_texts(self):
        """
        Clean and preprocess all input texts
        """
        print("🧹 Preprocessing texts...")
        
        self.preprocessed_texts = []
        
        for text_item in self.texts_to_process:
            try:
                text_id = text_item['id']
                original_text = text_item['text']
                
                # Basic text cleaning
                import re
                cleaned_text = re.sub(r'\s+', ' ', original_text).strip()
                cleaned_text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', cleaned_text)
                
                # Detect text type
                text_type = self._detect_text_type(cleaned_text)
                
                self.preprocessed_texts.append({
                    'id': text_id,
                    'original_text': original_text,
                    'processed_text': cleaned_text,
                    'text_type': text_type.value,
                    'preprocessing_time': datetime.now().isoformat()
                })
                
            except Exception as e:
                error_msg = f"Error preprocessing text {text_item.get('id', 'unknown')}: {e}"
                self.errors_encountered.append(error_msg)
                print(f"❌ {error_msg}")
        
        print(f"Preprocessed {len(self.preprocessed_texts)} texts")
        
        self.next(self.analyze_traditional_nlp, foreach='preprocessed_texts')

    @catch(var='nlp_errors')
    @retry(times=2)
    @step
    def analyze_traditional_nlp(self):
        """
        Perform traditional NLP analysis on each text
        """
        # Access the current text being processed
        text_data = self.input
        text_id = text_data['id']
        processed_text = text_data['processed_text']
        
        print(f"🔍 Analyzing text {text_id} with traditional NLP...")
        
        analysis_start_time = datetime.now()
        self.nlp_errors = []
        
        # Initialize results
        self.nlp_results = {
            'text_id': text_id,
            'sentiment': {},
            'entities': [],
            'keywords': [],
            'metrics': {}
        }
        
        # Sentiment analysis
        if self.nlp_models.get('sentiment'):
            try:
                scores = self.nlp_models['sentiment'].polarity_scores(processed_text)
                self.nlp_results['sentiment'] = {
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu'],
                    'compound': scores['compound']
                }
            except Exception as e:
                self.nlp_errors.append(f"Sentiment analysis failed: {e}")
        
        # Named entity recognition
        if self.nlp_models.get('spacy'):
            try:
                doc = self.nlp_models['spacy'](processed_text)
                
                # Extract entities
                entities = []
                for ent in doc.ents:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'description': spacy.explain(ent.label_),
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
                self.nlp_results['entities'] = entities
                
                # Extract keywords (simplified)
                keywords = [
                    token.lemma_.lower() for token in doc
                    if (token.is_alpha and 
                        not token.is_stop and 
                        not token.is_punct and
                        len(token.text) > 2)
                ]
                
                # Get top keywords by frequency
                from collections import Counter
                keyword_freq = Counter(keywords)
                self.nlp_results['keywords'] = [
                    word for word, _ in keyword_freq.most_common(10)
                ]
                
            except Exception as e:
                self.nlp_errors.append(f"spaCy analysis failed: {e}")
        
        # Text metrics
        try:
            metrics = self._calculate_text_metrics(processed_text)
            self.nlp_results['metrics'] = asdict(metrics)
        except Exception as e:
            self.nlp_errors.append(f"Metrics calculation failed: {e}")
        
        # Calculate processing time
        processing_time = (datetime.now() - analysis_start_time).total_seconds()
        self.nlp_results['processing_time'] = processing_time
        
        print(f"✅ Completed NLP analysis for {text_id} in {processing_time:.2f}s")
        
        self.next(self.analyze_with_llm)

    @catch(var='llm_errors')
    @retry(times=2)
    @step  
    def analyze_with_llm(self):
        """
        Perform LLM-based analysis on the text
        """
        text_data = self.input
        text_id = text_data['id']
        processed_text = text_data['processed_text']
        text_type = text_data['text_type']
        
        print(f"🤖 Analyzing text {text_id} with LLM...")
        
        self.llm_errors = []
        self.llm_results = {
            'text_id': text_id,
            'analysis': {},
            'summary': None,
            'insights': []
        }
        
        if not self.enable_llm or not self.nlp_models.get('llm_client'):
            print(f"⏭️ LLM analysis disabled for {text_id}")
            self.llm_results['note'] = "LLM analysis disabled"
            self.next(self.join_analysis_results)
            return
        
        try:
            # Create analysis prompt
            prompt = self._build_analysis_prompt(processed_text, text_type)
            
            # Call LLM
            response = self.nlp_models['llm_client'].chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse response
            llm_content = response.choices[0].message.content
            self.llm_results['raw_response'] = llm_content
            
            # Try to extract structured data
            try:
                import json
                import re
                json_match = re.search(r'\{.*\}', llm_content, re.DOTALL)
                if json_match:
                    parsed_analysis = json.loads(json_match.group())
                    self.llm_results['analysis'] = parsed_analysis
                else:
                    self.llm_results['analysis'] = {'raw_text': llm_content}
            except:
                self.llm_results['analysis'] = {'raw_text': llm_content}
            
            # Generate summary
            summary_prompt = f"Provide a concise 2-sentence summary of this text:\n\n{processed_text[:2000]}"
            
            summary_response = self.nlp_models['llm_client'].chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            self.llm_results['summary'] = summary_response.choices[0].message.content.strip()
            
            print(f"✅ Completed LLM analysis for {text_id}")
            
        except Exception as e:
            error_msg = f"LLM analysis failed for {text_id}: {e}"
            self.llm_errors.append(error_msg)
            print(f"❌ {error_msg}")
        
        self.next(self.join_analysis_results)

    @step
    def join_analysis_results(self, inputs):
        """
        Combine analysis results from all texts
        """
        print("🔗 Joining analysis results...")
        
        self.all_results = []
        self.processing_summary = {
            'total_processed': 0,
            'successful_nlp': 0,
            'successful_llm': 0,
            'errors': []
        }
        
        self.flow_start_time = None

        for inp in inputs:
            try:
                if self.flow_start_time is None:
                    self.flow_start_time = inp.flow_start_time
                else:
                    self.flow_start_time = min(self.flow_start_time, inp.flow_start_time)    
                # Get the original text data
                text_data = inp.input
                
                # Combine NLP and LLM results
                result = AnalysisResult(
                    text_id=text_data['id'],
                    original_text=text_data['original_text'],
                    processed_text=text_data['processed_text'],
                    text_type=text_data['text_type'],
                    sentiment=inp.nlp_results.get('sentiment', {}),
                    entities=inp.nlp_results.get('entities', []),
                    keywords=inp.nlp_results.get('keywords', []),
                    metrics=inp.nlp_results.get('metrics', {}),
                    summary=inp.llm_results.get('summary'),
                    llm_analysis=inp.llm_results.get('analysis', {}),
                    processing_time=inp.nlp_results.get('processing_time', 0),
                    timestamp=datetime.now().isoformat()
                )
                
                self.all_results.append(result)
                self.processing_summary['total_processed'] += 1
                
                # Count successes
                if inp.nlp_results.get('sentiment') or inp.nlp_results.get('entities'):
                    self.processing_summary['successful_nlp'] += 1
                
                if inp.llm_results.get('summary') or inp.llm_results.get('analysis'):
                    self.processing_summary['successful_llm'] += 1
                
                # Collect errors
                if hasattr(inp, 'nlp_errors') and inp.nlp_errors:
                    self.processing_summary['errors'].extend(inp.nlp_errors)
                if hasattr(inp, 'llm_errors') and inp.llm_errors:
                    self.processing_summary['errors'].extend(inp.llm_errors)
                
            except Exception as e:
                error_msg = f"Error joining results: {e}"
                self.processing_summary['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        print(f"✅ Joined results for {len(self.all_results)} texts")
        print(f"Processing summary: {self.processing_summary}")
        
        self.next(self.generate_insights)

    @step
    def generate_insights(self):
        """
        Generate aggregate insights from all analyses
        """
        print("💡 Generating aggregate insights...")
        
        if not self.all_results:
            self.aggregate_insights = {"error": "No results to analyze"}
            self.next(self.export_results)
            return
        
        # Calculate aggregate metrics
        self.aggregate_insights = {
            'total_texts_analyzed': len(self.all_results),
            'text_type_distribution': {},
            'sentiment_overview': {},
            'common_keywords': [],
            'average_metrics': {},
            'entity_types': {},
            'processing_stats': {}
        }
        
        # Text type distribution
        text_types = [result.text_type for result in self.all_results]
        self.aggregate_insights['text_type_distribution'] = {
            text_type: text_types.count(text_type) for text_type in set(text_types)
        }
        
        # Sentiment overview
        sentiments = [result.sentiment for result in self.all_results if result.sentiment]
        if sentiments:
            avg_sentiment = {
                'positive': np.mean([s.get('positive', 0) for s in sentiments]),
                'negative': np.mean([s.get('negative', 0) for s in sentiments]),
                'neutral': np.mean([s.get('neutral', 0) for s in sentiments]),
                'compound': np.mean([s.get('compound', 0) for s in sentiments])
            }
            self.aggregate_insights['sentiment_overview'] = avg_sentiment
        
        # Common keywords across all texts
        all_keywords = []
        for result in self.all_results:
            all_keywords.extend(result.keywords)
        
        from collections import Counter
        keyword_freq = Counter(all_keywords)
        self.aggregate_insights['common_keywords'] = keyword_freq.most_common(20)
        
        # Average text metrics
        metrics_list = [result.metrics for result in self.all_results if result.metrics]
        if metrics_list:
            numeric_metrics = ['word_count', 'sentence_count', 'flesch_score', 'flesch_kincaid_grade']
            avg_metrics = {}
            for metric in numeric_metrics:
                values = [m.get(metric, 0) for m in metrics_list if m.get(metric)]
                if values:
                    avg_metrics[f'avg_{metric}'] = np.mean(values)
            self.aggregate_insights['average_metrics'] = avg_metrics
        
        # Entity type distribution
        all_entities = []
        for result in self.all_results:
            all_entities.extend([e.get('label', 'UNKNOWN') for e in result.entities])
        
        entity_freq = Counter(all_entities)
        self.aggregate_insights['entity_types'] = dict(entity_freq.most_common(15))
        
        # Processing statistics
        processing_times = [result.processing_time for result in self.all_results if result.processing_time]
        self.aggregate_insights['processing_stats'] = {
            'total_processing_time': sum(processing_times),
            'average_processing_time': np.mean(processing_times) if processing_times else 0,
            'texts_per_second': len(self.all_results) / sum(processing_times) if processing_times else 0
        }
        
        print(f"✅ Generated insights for {len(self.all_results)} texts")
        
        self.next(self.export_results)

    @step
    def export_results(self):
        """
        Export results in requested format(s)
        """
        print(f"📤 Exporting results in {self.output_format} format...")
        
        self.export_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Prepare data for export
            export_data = {
                'flow_metadata': {
                    'run_id': current.run_id,
                    'flow_name': current.flow_name,
                    'execution_time': (datetime.now() - self.flow_start_time).total_seconds(),
                    'parameters': {
                        'llm_model': self.llm_model,
                        'enable_llm': self.enable_llm,
                        'spacy_model': self.spacy_model,
                        'max_texts': self.max_texts
                    }
                },
                # 'input_summary': self.input_summary,
                'processing_summary': self.processing_summary,
                'aggregate_insights': self.aggregate_insights,
                'detailed_results': [asdict(result) for result in self.all_results]
            }
            
            # Export as JSON
            if self.output_format in ['json', 'both']:
                json_path = f"text_analysis_results_{timestamp}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                self.export_paths.append(json_path)
                print(f"✅ Exported JSON results to {json_path}")
            
            # Export as CSV
            if self.output_format in ['csv', 'both']:
                # Create a flattened DataFrame for CSV export
                csv_data = []
                for result in self.all_results:
                    row = {
                        'text_id': result.text_id,
                        'text_type': result.text_type,
                        'word_count': result.metrics.get('word_count', 0) if result.metrics else 0,
                        'sentiment_positive': result.sentiment.get('positive', 0),
                        'sentiment_negative': result.sentiment.get('negative', 0),
                        'sentiment_compound': result.sentiment.get('compound', 0),
                        'entity_count': len(result.entities),
                        'keyword_count': len(result.keywords),
                        'has_summary': bool(result.summary),
                        'processing_time': result.processing_time,
                        'keywords': ', '.join(result.keywords[:5]),  # Top 5 keywords
                        'top_entities': ', '.join([e.get('text', '') for e in result.entities[:3]])  # Top 3 entities
                    }
                    csv_data.append(row)
                
                csv_path = f"text_analysis_summary_{timestamp}.csv"
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False)
                self.export_paths.append(csv_path)
                print(f"✅ Exported CSV summary to {csv_path}")
            
            # Store final summary
            self.final_summary = {
                'total_texts_processed': len(self.all_results),
                'successful_analyses': self.processing_summary['successful_nlp'],
                'export_paths': self.export_paths,
                'total_execution_time': (datetime.now() - self.flow_start_time).total_seconds(),
                'key_insights': {
                    'most_common_text_type': max(self.aggregate_insights['text_type_distribution'].items(), 
                                                key=lambda x: x[1])[0] if self.aggregate_insights['text_type_distribution'] else 'unknown',
                    'average_sentiment': self.aggregate_insights['sentiment_overview'].get('compound', 0),
                    'most_common_keyword': self.aggregate_insights['common_keywords'][0][0] if self.aggregate_insights['common_keywords'] else 'none'
                }
            }
            
        except Exception as e:
            error_msg = f"Export failed: {e}"
            print(f"❌ {error_msg}")
            self.export_paths = []
            self.final_summary = {'error': error_msg}
        
        self.next(self.end)

    @step
    def end(self):
        """
        Finalize the workflow and display summary
        """
        total_time = (datetime.now() - self.flow_start_time).total_seconds()
        
        print("\n" + "="*60)
        print("🎉 TEXT ANALYSIS FLOW COMPLETED")
        print("="*60)
        print(f"Run ID: {current.run_id}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Texts processed: {len(self.all_results)}")
        print(f"Export files: {self.export_paths}")
        
        if hasattr(self, 'final_summary') and 'key_insights' in self.final_summary:
            insights = self.final_summary['key_insights']
            print(f"Most common text type: {insights['most_common_text_type']}")
            print(f"Average sentiment: {insights['average_sentiment']:.3f}")
            print(f"Top keyword: {insights['most_common_keyword']}")
        
        if self.processing_summary.get('errors'):
            print(f"\n⚠️ Errors encountered: {len(self.processing_summary['errors'])}")
            for error in self.processing_summary['errors'][:3]:  # Show first 3 errors
                print(f"  - {error}")
        
        print("\n💡 To view results:")
        print(f"  metaflow run {current.flow_name}/{current.run_id} list artifacts")
        print("="*60)
    
    # Helper methods
    def _detect_text_type(self, text: str) -> TextType:
        """Detect the type of text content"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['subject:', 'from:', 'to:', 'dear', 'sincerely']):
            return TextType.EMAIL
        elif any(keyword in text_lower for keyword in ['abstract', 'methodology', 'references']):
            return TextType.ACADEMIC
        elif any(keyword in text_lower for keyword in ['whereas', 'hereby', 'pursuant']):
            return TextType.LEGAL
        elif any(pattern in text for pattern in ['#', '@', 'RT']):
            return TextType.SOCIAL_MEDIA
        elif any(keyword in text_lower for keyword in ['algorithm', 'implementation', 'system']):
            return TextType.TECHNICAL
        else:
            return TextType.GENERAL
    
    def _calculate_text_metrics(self, text: str) -> TextMetrics:
        """Calculate text readability metrics"""
        try:
            words = word_tokenize(text)
            sentences = sent_tokenize(text)
            paragraphs = text.split('\n\n')
            
            word_count = len(words)
            sentence_count = len(sentences)
            paragraph_count = len([p for p in paragraphs if p.strip()])
            
            avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
            
            try:
                flesch_score = flesch_reading_ease(text)
                fk_grade = flesch_kincaid_grade(text)
            except:
                flesch_score = 0.0
                fk_grade = 0.0
            
            # Determine complexity level
            if flesch_score >= 90:
                complexity = "very_easy"
            elif flesch_score >= 80:
                complexity = "easy"
            elif flesch_score >= 70:
                complexity = "fairly_easy"
            elif flesch_score >= 60:
                complexity = "standard"
            elif flesch_score >= 50:
                complexity = "fairly_difficult"
            elif flesch_score >= 30:
                complexity = "difficult"
            else:
                complexity = "very_difficult"
            
            return TextMetrics(
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                avg_words_per_sentence=avg_words_per_sentence,
                flesch_score=flesch_score,
                flesch_kincaid_grade=fk_grade,
                complexity_level=complexity
            )
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return TextMetrics()
    
    def _build_analysis_prompt(self, text: str, text_type: str) -> str:
        """Build analysis prompt for LLM"""
        return f"""
        Analyze the following {text_type} text and provide insights in JSON format:
        
        Text: "{text[:2000]}..."
        
        Please provide analysis with these keys:
        - "themes": main topics and themes (list)
        - "tone": overall tone (string)
        - "insights": key insights or takeaways (list)
        - "audience": target audience assessment (string)
        - "classification": content category (string)
        
        Respond with valid JSON only.
        """

if __name__ == '__main__':
    TextAnalysisFlow()
