"""
Week 2: Hybrid ML + LLM Pipeline with Metaflow and LangChain
===========================================================

This flow demonstrates integration patterns for combining traditional ML 
preprocessing with LLM-powered analysis and insights.

Features:
- Metaflow data processing workflows
- LangChain LLM integration for analysis
- Hybrid traditional + AI-powered insights
- Error handling and fallback mechanisms
- Structured reporting with natural language

Usage:
    python hybrid_pipeline.py run
    python hybrid_pipeline.py run --model_name mistral --use_llm_analysis True
"""

from metaflow import FlowSpec, step, Parameter, catch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# LangChain imports with error handling
try:
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import Ollama
    from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
    from langchain_core.runnables import RunnablePassthrough
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️ LangChain not available - running in traditional mode only")
    LANGCHAIN_AVAILABLE = False


class StructuredInsightParser(BaseOutputParser):
    """
    Custom parser to extract structured insights from LLM responses
    """
    
    def parse(self, text: str) -> dict:
        """Parse LLM output into structured insights"""
        try:
            lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
            
            insights = {
                'summary': '',
                'key_findings': [],
                'recommendations': [],
                'concerns': [],
                'raw_response': text
            }
            
            current_section = 'summary'
            
            for line in lines:
                line_lower = line.lower()
                
                # Section detection
                if any(keyword in line_lower for keyword in ['findings', 'patterns', 'observations']):
                    current_section = 'key_findings'
                    continue
                elif any(keyword in line_lower for keyword in ['recommend', 'suggest', 'should']):
                    current_section = 'recommendations'
                    continue
                elif any(keyword in line_lower for keyword in ['concern', 'warning', 'issue', 'problem']):
                    current_section = 'concerns'
                    continue
                
                # Content extraction
                if current_section == 'summary' and not insights['summary']:
                    insights['summary'] = line
                elif current_section == 'key_findings':
                    if line and not any(keyword in line_lower for keyword in ['findings', 'patterns']):
                        insights['key_findings'].append(line.strip('- '))
                elif current_section == 'recommendations':
                    if line and not any(keyword in line_lower for keyword in ['recommend']):
                        insights['recommendations'].append(line.strip('- '))
                elif current_section == 'concerns':
                    if line and not any(keyword in line_lower for keyword in ['concern']):
                        insights['concerns'].append(line.strip('- '))
            
            # Fallback if no summary found
            if not insights['summary'] and lines:
                insights['summary'] = lines[0]
            
            return insights
            
        except Exception as e:
            return {
                'error': str(e),
                'raw_response': text,
                'summary': 'Error parsing LLM response',
                'key_findings': [],
                'recommendations': [],
                'concerns': []
            }


class HybridMLLMFlow(FlowSpec):
    """
    Hybrid pipeline combining Metaflow data processing with LangChain LLM analysis.
    Demonstrates best practices for integrating traditional ML with generative AI.
    """
    
    # Configuration parameters
    data_file = Parameter('data_file',
                         help='Path to input data file',
                         default='../data/titanic.csv')
    
    model_name = Parameter('model_name',
                          help='Ollama model for LLM analysis',
                          default='llama3.2')
    
    use_llm_analysis = Parameter('use_llm_analysis',
                                help='Enable LLM-powered analysis',
                                default=True)
    
    llm_temperature = Parameter('llm_temperature',
                               help='LLM temperature for analysis',
                               default=0.3)
    
    test_size = Parameter('test_size',
                         help='Test split ratio',
                         default=0.2)
    
    random_state = Parameter('random_state',
                           help='Random seed',
                           default=42)
    
    @step
    def start(self):
        """
        Initialize hybrid pipeline
        """
        print("🌊🦜 Starting Hybrid ML + LLM Pipeline")
        print("=" * 45)
        print(f"📋 Configuration:")
        print(f"   Data file: {self.data_file}")
        print(f"   LLM model: {self.model_name}")
        print(f"   LLM analysis: {'Enabled' if self.use_llm_analysis else 'Disabled'}")
        print(f"   LLM temperature: {self.llm_temperature}")
        print(f"   Test size: {self.test_size}")
        
        # Load data with fallback to synthetic
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Data loaded from file: {self.df.shape}")
        except FileNotFoundError:
            print(f"⚠️ Data file not found: {self.data_file}")
            print("   Creating synthetic dataset...")
            self.df = self._create_sample_dataset()
            print(f"✅ Synthetic data created: {self.df.shape}")
        
        self.next(self.preprocess_data)
    
    def _create_sample_dataset(self):
        """Create sample Titanic-like dataset"""
        np.random.seed(self.random_state)
        n_samples = 891
        
        # Create realistic data
        ages = np.random.normal(30, 15, n_samples)
        ages = np.clip(ages, 0, 80)
        missing_age_mask = np.random.random(n_samples) < 0.20
        ages[missing_age_mask] = np.nan
        
        sexes = np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35])
        pclasses = np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55])
        embarked = np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
        
        # Add missing values
        missing_embarked_mask = np.random.random(n_samples) < 0.002
        embarked[missing_embarked_mask] = None
        
        sibsp = np.random.poisson(0.5, n_samples)
        parch = np.random.poisson(0.4, n_samples)
        
        # Correlated fares
        fare_base = {1: 80, 2: 20, 3: 10}
        fares = [np.random.lognormal(np.log(fare_base[pc]), 0.5) for pc in pclasses]
        
        # Realistic survival patterns
        survival_prob = 0.3
        survival_prob += (sexes == 'female') * 0.4
        survival_prob += (ages < 16) * 0.3
        survival_prob += (pclasses == 1) * 0.3
        survival_prob += (pclasses == 2) * 0.15
        survived = np.random.binomial(1, survival_prob)
        
        return pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Survived': survived,
            'Pclass': pclasses,
            'Sex': sexes,
            'Age': ages,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fares,
            'Embarked': embarked
        })
    
    @step
    def preprocess_data(self):
        """
        Quick but comprehensive data preprocessing
        """
        print("🔧 Preprocessing data...")
        
        df_clean = self.df.copy()
        preprocessing_log = {}
        
        # Store original statistics
        original_stats = {
            'total_rows': len(df_clean),
            'total_columns': len(df_clean.columns),
            'missing_values': df_clean.isnull().sum().to_dict(),
            'data_types': df_clean.dtypes.to_dict()
        }
        
        # Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        
        # Age: Group-based median
        if 'Age' in df_clean.columns and df_clean['Age'].isnull().any():
            if 'Sex' in df_clean.columns and 'Pclass' in df_clean.columns:
                age_median = df_clean.groupby(['Sex', 'Pclass'])['Age'].transform('median')
                df_clean['Age'].fillna(age_median, inplace=True)
            else:
                df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
        
        # Embarked: Most frequent
        if 'Embarked' in df_clean.columns and df_clean['Embarked'].isnull().any():
            most_frequent = df_clean['Embarked'].mode()[0] if not df_clean['Embarked'].mode().empty else 'S'
            df_clean['Embarked'].fillna(most_frequent, inplace=True)
        
        # Fare: Class-based median  
        if 'Fare' in df_clean.columns and df_clean['Fare'].isnull().any():
            if 'Pclass' in df_clean.columns:
                fare_median = df_clean.groupby('Pclass')['Fare'].transform('median')
                df_clean['Fare'].fillna(fare_median, inplace=True)
            else:
                df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
        
        missing_after = df_clean.isnull().sum().sum()
        preprocessing_log['missing_values'] = {
            'before': missing_before,
            'after': missing_after,
            'handled': missing_before - missing_after
        }
        
        # Basic feature engineering
        if 'SibSp' in df_clean.columns and 'Parch' in df_clean.columns:
            df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
            df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)
            preprocessing_log['features_created'] = ['FamilySize', 'IsAlone']
        
        # Generate comprehensive statistics for analysis
        self.data_statistics = {
            'basic_info': {
                'total_passengers': len(df_clean),
                'survival_rate': df_clean['Survived'].mean() if 'Survived' in df_clean.columns else None,
                'average_age': df_clean['Age'].mean() if 'Age' in df_clean.columns else None,
                'average_fare': df_clean['Fare'].mean() if 'Fare' in df_clean.columns else None
            },
            'distributions': {
                'class_distribution': df_clean['Pclass'].value_counts().to_dict() if 'Pclass' in df_clean.columns else {},
                'gender_distribution': df_clean['Sex'].value_counts().to_dict() if 'Sex' in df_clean.columns else {},
                'embarkation_distribution': df_clean['Embarked'].value_counts().to_dict() if 'Embarked' in df_clean.columns else {}
            },
            'survival_analysis': self._calculate_survival_statistics(df_clean),
            'data_quality': {
                'missing_data_handled': preprocessing_log['missing_values'],
                'outliers': self._detect_outliers(df_clean),
                'completeness_score': ((len(df_clean.columns) * len(df_clean)) - missing_after) / (len(df_clean.columns) * len(df_clean)) * 100
            }
        }
        
        self.df_processed = df_clean
        self.preprocessing_log = preprocessing_log
        
        print(f"   Missing values: {missing_before} → {missing_after}")
        print(f"   Features added: {len(preprocessing_log.get('features_created', []))}")
        print(f"   Data completeness: {self.data_statistics['data_quality']['completeness_score']:.1f}%")
        
        # Route to appropriate analysis
        if self.use_llm_analysis and LANGCHAIN_AVAILABLE:
            self.next(self.llm_analysis)
        else:
            self.next(self.traditional_analysis)
    
    def _calculate_survival_statistics(self, df):
        """Calculate detailed survival statistics"""
        if 'Survived' not in df.columns:
            return {"error": "No survival data available"}
        
        survival_stats = {
            'overall_rate': df['Survived'].mean(),
            'by_gender': df.groupby('Sex')['Survived'].mean().to_dict() if 'Sex' in df.columns else {},
            'by_class': df.groupby('Pclass')['Survived'].mean().to_dict() if 'Pclass' in df.columns else {},
            'by_embarkation': df.groupby('Embarked')['Survived'].mean().to_dict() if 'Embarked' in df.columns else {}
        }
        
        # Age group analysis
        if 'Age' in df.columns:
            age_groups = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young Adult', 'Middle Age', 'Senior'])
            survival_stats['by_age_group'] = df.groupby(age_groups)['Survived'].mean().to_dict()
        
        return survival_stats
    
    def _detect_outliers(self, df):
        """Detect outliers in numerical columns"""
        outlier_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['PassengerId', 'Survived']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_info[col] = {
                    'count': outliers,
                    'percentage': (outliers / len(df)) * 100
                }
        
        return outlier_info
    
    @catch(var='llm_error')
    @step
    def llm_analysis(self):
        """
        LLM-powered data analysis with structured output
        """
        print("🧠 Running LLM analysis...")
        
        try:
            # Create advanced analysis prompt
            analysis_prompt = PromptTemplate.from_template(
                """You are an expert data scientist analyzing passenger data from the Titanic disaster.

DATA OVERVIEW:
- Total passengers: {total_passengers}
- Overall survival rate: {survival_rate:.1%}
- Average age: {average_age:.1f} years
- Average fare: ${average_fare:.2f}

DISTRIBUTIONS:
- Class distribution: {class_distribution}
- Gender distribution: {gender_distribution}
- Embarkation ports: {embarkation_distribution}

SURVIVAL PATTERNS:
- By gender: {survival_by_gender}
- By class: {survival_by_class}
- By age group: {survival_by_age_group}

DATA QUALITY:
- Data completeness: {completeness_score:.1f}%
- Missing data handled: {missing_data_handled}
- Outliers detected: {outliers_summary}

Please provide a comprehensive analysis including:

KEY FINDINGS:
- What are the 3 most important survival patterns?
- Which factors show the strongest correlation with survival?

RECOMMENDATIONS:
- What insights would be most valuable for understanding this disaster?
- What additional data would strengthen the analysis?

CONCERNS:
- Are there any data quality issues that could affect conclusions?
- What potential biases should be considered?

Please structure your response clearly with the above sections."""
            )
            
            # Prepare input data
            stats = self.data_statistics
            analysis_input = {
                'total_passengers': stats['basic_info']['total_passengers'],
                'survival_rate': stats['basic_info']['survival_rate'],
                'average_age': stats['basic_info']['average_age'],
                'average_fare': stats['basic_info']['average_fare'],
                'class_distribution': stats['distributions']['class_distribution'],
                'gender_distribution': stats['distributions']['gender_distribution'],
                'embarkation_distribution': stats['distributions']['embarkation_distribution'],
                'survival_by_gender': stats['survival_analysis']['by_gender'],
                'survival_by_class': stats['survival_analysis']['by_class'],
                'survival_by_age_group': stats['survival_analysis'].get('by_age_group', {}),
                'completeness_score': stats['data_quality']['completeness_score'],
                'missing_data_handled': stats['data_quality']['missing_data_handled'],
                'outliers_summary': {k: v['count'] for k, v in stats['data_quality']['outliers'].items()}
            }
            
            # Create LLM chain
            llm = Ollama(model=self.model_name, temperature=self.llm_temperature)
            parser = StructuredInsightParser()
            
            analysis_chain = analysis_prompt | llm | parser
            
            # Run analysis
            print(f"   Using model: {self.model_name}")
            print(f"   Temperature: {self.llm_temperature}")
            
            llm_result = analysis_chain.invoke(analysis_input)
            
            # Validate result
            if 'error' in llm_result:
                print(f"   ⚠️ LLM parsing error: {llm_result['error']}")
                print("   Falling back to traditional analysis...")
                self.next(self.traditional_analysis)
                return
            
            self.analysis_result = {
                'type': 'llm_analysis',
                'model_used': self.model_name,
                'temperature': self.llm_temperature,
                'insights': llm_result,
                'analysis_quality': self._assess_llm_quality(llm_result)
            }
            
            print("✅ LLM analysis complete")
            print(f"   Insights generated: {len(llm_result.get('key_findings', []))} findings")
            print(f"   Recommendations: {len(llm_result.get('recommendations', []))}")
            print(f"   Concerns identified: {len(llm_result.get('concerns', []))}")
            
        except Exception as e:
            print(f"❌ LLM analysis failed: {e}")
            print("   Falling back to traditional analysis...")
            self.llm_error = str(e)
            self.next(self.traditional_analysis)
            return
        
        self.next(self.generate_insights)
    
    def _assess_llm_quality(self, llm_result):
        """Assess the quality of LLM analysis"""
        quality_score = 0
        
        # Check completeness
        if llm_result.get('summary'):
            quality_score += 25
        if llm_result.get('key_findings'):
            quality_score += 25
        if llm_result.get('recommendations'):
            quality_score += 25
        if llm_result.get('concerns'):
            quality_score += 25
        
        # Content quality indicators
        total_content = len(llm_result.get('summary', '')) + \
                       sum(len(item) for item in llm_result.get('key_findings', [])) + \
                       sum(len(item) for item in llm_result.get('recommendations', [])) + \
                       sum(len(item) for item in llm_result.get('concerns', []))
        
        return {
            'completeness_score': quality_score,
            'content_length': total_content,
            'sections_completed': sum(1 for section in ['summary', 'key_findings', 'recommendations', 'concerns'] 
                                    if llm_result.get(section))
        }
    
    @step
    def traditional_analysis(self):
        """
        Traditional statistical analysis as fallback
        """
        print("📊 Running traditional statistical analysis...")
        
        stats = self.data_statistics
        
        # Generate insights using statistical rules
        findings = []
        recommendations = []
        concerns = []
        
        # Survival rate analysis
        if stats['basic_info']['survival_rate'] is not None:
            survival_rate = stats['basic_info']['survival_rate']
            findings.append(f"Overall survival rate was {survival_rate:.1%}, indicating high mortality in the disaster")
            
            if survival_rate < 0.4:
                concerns.append("Low survival rate suggests severe emergency response challenges")
        
        # Gender analysis
        if stats['survival_analysis']['by_gender']:
            gender_survival = stats['survival_analysis']['by_gender']
            if 'female' in gender_survival and 'male' in gender_survival:
                female_rate = gender_survival['female']
                male_rate = gender_survival['male']
                findings.append(f"Women had {female_rate:.1%} survival rate vs {male_rate:.1%} for men")
                
                if female_rate > male_rate + 0.2:
                    findings.append("Strong 'women and children first' protocol evident")
        
        # Class analysis
        if stats['survival_analysis']['by_class']:
            class_survival = stats['survival_analysis']['by_class']
            if len(class_survival) >= 3:
                first_class = class_survival.get(1, 0)
                third_class = class_survival.get(3, 0)
                findings.append(f"First class: {first_class:.1%} survival vs Third class: {third_class:.1%}")
                
                if first_class > third_class + 0.2:
                    concerns.append("Significant class-based survival disparities observed")
        
        # Data quality insights
        completeness = stats['data_quality']['completeness_score']
        if completeness > 95:
            findings.append(f"High data completeness ({completeness:.1f}%) enables reliable analysis")
        else:
            concerns.append(f"Data completeness only {completeness:.1f}% - may affect analysis reliability")
        
        # Missing data
        missing_handled = stats['data_quality']['missing_data_handled']['handled']
        if missing_handled > 0:
            recommendations.append(f"Successfully handled {missing_handled} missing values using appropriate imputation")
        
        # Outliers
        outlier_counts = [info['count'] for info in stats['data_quality']['outliers'].values()]
        if any(count > len(self.df_processed) * 0.1 for count in outlier_counts):
            concerns.append("High outlier presence in some variables may require additional investigation")
        
        # General recommendations
        recommendations.append("Consider additional feature engineering for age groups and family relationships")
        recommendations.append("Validate findings with historical disaster response literature")
        
        # Create structured result
        traditional_result = {
            'summary': f"Statistical analysis of {stats['basic_info']['total_passengers']} passengers reveals significant survival patterns by gender, class, and other factors",
            'key_findings': findings,
            'recommendations': recommendations,
            'concerns': concerns
        }
        
        self.analysis_result = {
            'type': 'traditional_analysis',
            'insights': traditional_result,
            'statistical_summary': {
                'survival_by_gender': stats['survival_analysis']['by_gender'],
                'survival_by_class': stats['survival_analysis']['by_class'],
                'data_quality_score': completeness
            }
        }
        
        print("✅ Traditional analysis complete")
        print(f"   Statistical findings: {len(findings)}")
        print(f"   Recommendations: {len(recommendations)}")
        print(f"   Concerns: {len(concerns)}")
        
        self.next(self.generate_insights)
    
    @step
    def generate_insights(self):
        """
        Generate combined insights and actionable recommendations
        """
        print("💡 Generating comprehensive insights...")
        
        # Combine preprocessing insights with analysis results
        insights = self.analysis_result['insights']
        
        # Add preprocessing-specific insights
        preprocessing_insights = []
        
        if hasattr(self, 'preprocessing_log'):
            missing_handled = self.preprocessing_log['missing_values']['handled']
            if missing_handled > 0:
                preprocessing_insights.append(f"Successfully processed {missing_handled} missing values")
            
            features_created = len(self.preprocessing_log.get('features_created', []))
            if features_created > 0:
                preprocessing_insights.append(f"Created {features_created} additional features for analysis")
        
        # Create actionable recommendations
        actionable_recommendations = []
        
        # Add technical recommendations
        actionable_recommendations.append("Implement stratified sampling for model training to handle class imbalance")
        actionable_recommendations.append("Consider ensemble methods to capture complex survival patterns")
        
        # Add domain-specific recommendations
        if self.data_statistics['basic_info']['survival_rate'] < 0.5:
            actionable_recommendations.append("Focus on emergency response protocols and evacuation procedures")
        
        # Combine all insights
        self.comprehensive_insights = {
            'analysis_type': self.analysis_result['type'],
            'primary_insights': insights,
            'preprocessing_insights': preprocessing_insights,
            'actionable_recommendations': actionable_recommendations,
            'data_statistics': self.data_statistics,
            'quality_assessment': self._generate_quality_assessment()
        }
        
        print(f"✅ Insights generated: {self.analysis_result['type']}")
        print(f"   Primary insights: {len(insights.get('key_findings', []))}")
        print(f"   Actionable recommendations: {len(actionable_recommendations)}")
        
        self.next(self.create_final_report)
    
    def _generate_quality_assessment(self):
        """Generate overall quality assessment"""
        stats = self.data_statistics
        
        quality_metrics = {
            'data_completeness': stats['data_quality']['completeness_score'],
            'sample_size': stats['basic_info']['total_passengers'],
            'feature_coverage': len(self.df_processed.columns),
            'analysis_reliability': 'high' if stats['data_quality']['completeness_score'] > 90 else 'medium'
        }
        
        # Add LLM quality if available
        if hasattr(self, 'analysis_result') and 'analysis_quality' in self.analysis_result:
            quality_metrics['llm_analysis_quality'] = self.analysis_result['analysis_quality']
        
        return quality_metrics
    
    @step
    def create_final_report(self):
        """
        Create comprehensive final report
        """
        print("📋 Creating final comprehensive report...")
        
        # Compile complete report
        self.final_report = {
            'executive_summary': self._create_executive_summary(),
            'data_overview': {
                'source': self.data_file,
                'shape': self.df.shape,
                'processed_shape': self.df_processed.shape,
                'preprocessing_applied': self.preprocessing_log
            },
            'analysis_results': self.comprehensive_insights,
            'technical_details': {
                'missing_value_strategy': 'group-based imputation',
                'feature_engineering': self.preprocessing_log.get('features_created', []),
                'analysis_method': self.analysis_result['type'],
                'model_used': self.model_name if self.analysis_result['type'] == 'llm_analysis' else 'statistical_analysis'
            },
            'recommendations': {
                'immediate_actions': self.comprehensive_insights['actionable_recommendations'][:3],
                'further_analysis': self.comprehensive_insights['actionable_recommendations'][3:],
                'data_improvements': self._suggest_data_improvements()
            },
            'quality_metrics': self.comprehensive_insights['quality_assessment']
        }
        
        print("✅ Final report compiled")
        print(f"   Report sections: {len(self.final_report)}")
        print(f"   Quality score: {self.final_report['quality_metrics']['data_completeness']:.1f}%")
        
        self.next(self.end)
    
    def _create_executive_summary(self):
        """Create executive summary of key findings"""
        insights = self.comprehensive_insights['primary_insights']
        stats = self.data_statistics['basic_info']
        
        summary_parts = []
        
        # Data overview
        summary_parts.append(f"Analysis of {stats['total_passengers']} passenger records")
        
        # Key finding
        if stats['survival_rate']:
            summary_parts.append(f"with {stats['survival_rate']:.1%} overall survival rate")
        
        # Top insight
        if insights.get('key_findings') and len(insights['key_findings']) > 0:
            summary_parts.append(f"reveals {insights['key_findings'][0].lower()}")
        
        # Analysis quality
        analysis_type = "AI-powered" if self.analysis_result['type'] == 'llm_analysis' else "statistical"
        summary_parts.append(f"using {analysis_type} analysis methods")
        
        return ". ".join(summary_parts) + "."
    
    def _suggest_data_improvements(self):
        """Suggest data collection improvements"""
        suggestions = []
        
        # Based on missing data patterns
        missing_values = self.data_statistics['data_quality']['missing_data_handled']
        if missing_values['before'] > 0:
            suggestions.append("Improve data collection completeness for passenger demographics")
        
        # Based on analysis type
        if self.analysis_result['type'] == 'traditional_analysis':
            suggestions.append("Consider implementing LLM analysis for deeper insights")
        
        # General suggestions
        suggestions.extend([
            "Collect additional contextual data about emergency procedures",
            "Include more detailed passenger background information",
            "Add timeline data for evacuation sequence analysis"
        ])
        
        return suggestions
    
    @step
    def end(self):
        """
        Pipeline completion with comprehensive output
        """
        print("\n🎉 Hybrid ML + LLM Pipeline Complete!")
        print("=" * 50)
        
        # Display key results
        report = self.final_report
        
        print("\n📊 EXECUTIVE SUMMARY:")
        print(f"   {report['executive_summary']}")
        
        print(f"\n🧠 ANALYSIS METHOD: {report['analysis_results']['analysis_type'].upper()}")
        if self.analysis_result['type'] == 'llm_analysis':
            print(f"   Model: {self.model_name}")
            print(f"   Temperature: {self.llm_temperature}")
        
        print(f"\n📈 DATA QUALITY:")
        quality = report['quality_metrics']
        print(f"   Completeness: {quality['data_completeness']:.1f}%")
        print(f"   Sample size: {quality['sample_size']} records")
        print(f"   Features: {quality['feature_coverage']} columns")
        print(f"   Reliability: {quality['analysis_reliability']}")
        
        print(f"\n🎯 KEY FINDINGS:")
        insights = report['analysis_results']['primary_insights']
        findings = insights.get('key_findings', [])
        for i, finding in enumerate(findings[:3], 1):  # Show top 3
            print(f"   {i}. {finding}")
        
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations']['immediate_actions'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n⚠️ CONCERNS:")
        concerns = insights.get('concerns', [])
        for i, concern in enumerate(concerns[:2], 1):  # Show top 2
            print(f"   {i}. {concern}")
        
        print(f"\n🎯 PIPELINE ARTIFACTS:")
        print("   ✅ Cleaned and processed dataset")
        print("   ✅ Comprehensive data statistics")
        print("   ✅ AI-powered or statistical analysis")
        print("   ✅ Structured insights and recommendations")
        print("   ✅ Quality assessment and validation")
        print("   ✅ Executive-ready final report")
        
        print(f"\n🚀 SUCCESS METRICS:")
        print(f"   - Data processed: {report['data_overview']['shape'][0]} → {report['data_overview']['processed_shape'][0]} rows")
        print(f"   - Features engineered: {len(report['technical_details']['feature_engineering'])}")
        print(f"   - Insights generated: {len(findings)} findings")
        print(f"   - Recommendations: {len(report['recommendations']['immediate_actions']) + len(report['recommendations']['further_analysis'])}")
        
        print(f"\n🎓 This pipeline demonstrates:")
        print("   🌊 Advanced Metaflow data processing")
        print("   🦜 LangChain LLM integration") 
        print("   🔗 Hybrid traditional + AI workflows")
        print("   📊 Structured insight generation")
        print("   📋 Production-ready reporting")
        
        print(f"\n🎉 Ready for deployment and scaling!")


if __name__ == '__main__':
    HybridMLLMFlow()