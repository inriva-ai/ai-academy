# Week 3 Exercises: Hybrid ML + LLM Evaluation

## 🎯 Learning Objectives

By completing these exercises, you will:
- Master hybrid evaluation systems combining quantitative ML metrics with qualitative LLM insights
- Implement sophisticated LangChain integration patterns for model interpretation
- Build natural language reporting systems for ML models
- Create business-friendly AI explanation interfaces
- Develop advanced prompt engineering for model analysis

## 📋 Prerequisites

- Completed Week 3 Workshop and ML pipelines
- Understanding of LangChain framework and prompt engineering
- Basic knowledge of model interpretation techniques
- Familiarity with business stakeholder communication

---

## 🧠 Exercise 1: Advanced Model Interpretation with LLM Integration

### Challenge
Build a comprehensive model interpretation system that combines traditional ML explainability techniques with LLM-powered natural language explanations.

### Scenario
Enterprise ML model deployment where stakeholders need both technical metrics and business-friendly explanations for model decisions and performance.

### Tasks

#### Task 1.1: Multi-Modal Explanation System
```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema import BaseOutputParser
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import shap
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ModelExplanation(BaseModel):
    """Structured model explanation output."""
    technical_summary: str = Field(description="Technical performance summary")
    business_impact: str = Field(description="Business impact explanation")
    key_features: List[str] = Field(description="Most important features")
    confidence_level: str = Field(description="Model confidence assessment")
    recommendations: List[str] = Field(description="Actionable recommendations")
    risk_factors: List[str] = Field(description="Potential risks or limitations")

class HybridModelExplainer:
    """
    Advanced model explanation system combining ML and LLM techniques.
    """
    
    def __init__(self, llm_model="llama3.2", use_local_llm=True):
        """
        Initialize hybrid explainer with LLM configuration.
        
        Args:
            llm_model: Name of the LLM model to use
            use_local_llm: Whether to use local Ollama or external service
        """
        print(f"🧠 Initializing Hybrid Model Explainer")
        print(f"   LLM Model: {llm_model}")
        print(f"   Local LLM: {use_local_llm}")
        
        # TODO: Initialize LLM with error handling
        try:
            if use_local_llm:
                self.llm = Ollama(model=llm_model)
                self.llm_available = True
            else:
                # TODO: Add support for external LLM services
                raise NotImplementedError("External LLM services not implemented")
                
        except Exception as e:
            print(f"   ⚠️ LLM initialization failed: {e}")
            self.llm = None
            self.llm_available = False
        
        # Initialize explanation templates
        self.explanation_templates = self._create_explanation_templates()
        
        # Initialize output parser
        self.explanation_parser = PydanticOutputParser(pydantic_object=ModelExplanation)
    
    def _create_explanation_templates(self):
        """Create comprehensive explanation templates."""
        # TODO: Implement sophisticated explanation templates
        
        # Technical explanation template
        technical_template = PromptTemplate(
            input_variables=["model_type", "accuracy", "precision", "recall", "f1_score", 
                           "feature_importance", "dataset_info", "cross_validation"],
            template="""
            Provide a technical analysis of this machine learning model:
            
            Model Details:
            - Algorithm: {model_type}
            - Accuracy: {accuracy:.3f}
            - Precision: {precision:.3f}
            - Recall: {recall:.3f}
            - F1-Score: {f1_score:.3f}
            - Cross-validation: {cross_validation}
            
            Dataset Information:
            {dataset_info}
            
            Feature Importance (Top 5):
            {feature_importance}
            
            Provide analysis covering:
            1. Model performance assessment (excellent/good/fair/poor)
            2. Statistical significance and reliability
            3. Feature importance interpretation
            4. Potential overfitting or underfitting concerns
            5. Model strengths and limitations
            
            Keep the explanation technical but accessible to data scientists.
            """
        )
        
        # Business explanation template
        business_template = PromptTemplate(
            input_variables=["model_type", "business_context", "performance_metrics", 
                           "key_features", "impact_analysis"],
            template="""
            Explain this machine learning model's performance and implications for business stakeholders:
            
            Business Context: {business_context}
            Model Type: {model_type}
            Performance: {performance_metrics}
            Key Predictive Factors: {key_features}
            Impact Analysis: {impact_analysis}
            
            Provide a business-focused explanation covering:
            1. What this model does in plain English
            2. How accurate and reliable it is for business decisions
            3. Which factors most influence the predictions
            4. Potential business risks and opportunities
            5. Recommended actions for implementation
            
            Use business language, avoid technical jargon, and focus on actionable insights.
            """
        )
        
        # Comparative analysis template
        comparative_template = PromptTemplate(
            input_variables=["model_comparison", "performance_differences", "use_cases"],
            template="""
            Compare these machine learning models and provide selection guidance:
            
            Model Comparison:
            {model_comparison}
            
            Performance Differences:
            {performance_differences}
            
            Intended Use Cases:
            {use_cases}
            
            Provide comparative analysis covering:
            1. Relative strengths and weaknesses of each model
            2. Performance trade-offs (accuracy vs speed vs interpretability)
            3. Recommended model for different scenarios
            4. Risk assessment for each option
            5. Implementation considerations
            
            Focus on helping stakeholders make informed decisions.
            """
        )
        
        # Structured explanation template
        structured_template = ChatPromptTemplate.from_template(
            """
            Generate a comprehensive, structured explanation of this ML model:
            
            Model: {model_type}
            Performance: Accuracy {accuracy:.3f}, F1-Score {f1_score:.3f}
            Top Features: {top_features}
            Business Context: {business_context}
            
            {format_instructions}
            
            Provide a complete analysis with technical summary, business impact, 
            key features, confidence assessment, recommendations, and risk factors.
            """
        )
        
        return {
            'technical': technical_template,
            'business': business_template,
            'comparative': comparative_template,
            'structured': structured_template
        }
    
    def explain_model_performance(self, model, X_test, y_test, feature_names, 
                                business_context="", explanation_type="structured"):
        """
        Generate comprehensive model explanation.
        
        Args:
            model: Trained ML model
            X_test: Test features
            y_test: Test targets
            feature_names: List of feature names
            business_context: Business context description
            explanation_type: Type of explanation (technical/business/structured)
        
        Returns:
            Dictionary containing various explanation formats
        """
        print(f"🔍 Generating {explanation_type} model explanation...")
        
        # TODO: Calculate comprehensive performance metrics
        performance_metrics = self._calculate_performance_metrics(model, X_test, y_test)
        
        # TODO: Extract feature importance
        feature_importance = self._extract_feature_importance(model, X_test, y_test, feature_names)
        
        # TODO: Generate technical insights
        technical_insights = self._generate_technical_insights(model, performance_metrics, feature_importance)
        
        if not self.llm_available:
            return self._generate_fallback_explanation(performance_metrics, feature_importance, business_context)
        
        try:
            if explanation_type == "structured":
                explanation = self._generate_structured_explanation(
                    model, performance_metrics, feature_importance, business_context
                )
            elif explanation_type == "technical":
                explanation = self._generate_technical_explanation(
                    model, performance_metrics, feature_importance
                )
            elif explanation_type == "business":
                explanation = self._generate_business_explanation(
                    model, performance_metrics, feature_importance, business_context
                )
            else:
                raise ValueError(f"Unknown explanation type: {explanation_type}")
            
            return {
                'explanation_type': explanation_type,
                'llm_explanation': explanation,
                'technical_metrics': performance_metrics,
                'feature_importance': feature_importance,
                'technical_insights': technical_insights,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"   ⚠️ LLM explanation failed: {e}")
            return self._generate_fallback_explanation(performance_metrics, feature_importance, business_context)
    
    def _calculate_performance_metrics(self, model, X_test, y_test):
        """Calculate comprehensive performance metrics."""
        # TODO: Implement comprehensive metrics calculation
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.model_selection import cross_val_score
        
        y_pred = model.predict(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation metrics
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'sample_size': len(y_test)
        }
    
    def _extract_feature_importance(self, model, X_test, y_test, feature_names):
        """Extract feature importance using multiple methods."""
        # TODO: Implement multi-method feature importance extraction
        importance_methods = {}
        
        # Method 1: Model-specific importance
        if hasattr(model, 'feature_importances_'):
            importance_methods['tree_importance'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_methods['linear_importance'] = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
        
        # Method 2: Permutation importance
        try:
            perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
            importance_methods['permutation_importance'] = perm_importance.importances_mean
        except Exception as e:
            print(f"   ⚠️ Permutation importance failed: {e}")
        
        # Method 3: SHAP values (if available)
        try:
            import shap
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model, X_test[:100])  # Sample for efficiency
                shap_values = explainer(X_test[:10])
                importance_methods['shap_importance'] = np.abs(shap_values.values).mean(axis=0)
        except Exception as e:
            print(f"   ⚠️ SHAP importance failed: {e}")
        
        # Combine importance scores
        if importance_methods:
            # Use the first available method as primary
            primary_method = list(importance_methods.keys())[0]
            primary_importance = importance_methods[primary_method]
            
            # Create feature importance ranking
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': primary_importance
            }).sort_values('importance', ascending=False)
            
            return {
                'method_used': primary_method,
                'rankings': importance_df.to_dict('records'),
                'top_5_features': importance_df.head(5)['feature'].tolist(),
                'all_methods': importance_methods
            }
        else:
            return {
                'method_used': 'none_available',
                'rankings': [],
                'top_5_features': feature_names[:5],
                'all_methods': {}
            }
    
    def _generate_technical_insights(self, model, performance_metrics, feature_importance):
        """Generate technical insights without LLM."""
        # TODO: Implement rule-based technical insights
        insights = []
        
        # Performance insights
        accuracy = performance_metrics['accuracy']
        if accuracy > 0.95:
            insights.append("Excellent model performance with very high accuracy")
        elif accuracy > 0.90:
            insights.append("Very good model performance suitable for production")
        elif accuracy > 0.80:
            insights.append("Good model performance with room for improvement")
        else:
            insights.append("Model performance needs significant improvement")
        
        # Stability insights
        cv_std = performance_metrics['cv_std']
        if cv_std < 0.02:
            insights.append("Highly stable model with consistent performance")
        elif cv_std < 0.05:
            insights.append("Moderately stable model performance")
        else:
            insights.append("Model shows high variance - consider regularization")
        
        # Feature insights
        if feature_importance['method_used'] != 'none_available':
            top_feature = feature_importance['top_5_features'][0]
            insights.append(f"Most predictive feature: {top_feature}")
        
        return insights
    
    def _generate_structured_explanation(self, model, performance_metrics, feature_importance, business_context):
        """Generate structured explanation using LLM."""
        # TODO: Implement structured explanation generation
        template = self.explanation_templates['structured']
        
        # Prepare input data
        input_data = {
            'model_type': type(model).__name__,
            'accuracy': performance_metrics['accuracy'],
            'f1_score': performance_metrics['f1_score'],
            'top_features': ', '.join(feature_importance['top_5_features']),
            'business_context': business_context or "General prediction task",
            'format_instructions': self.explanation_parser.get_format_instructions()
        }
        
        # Create chain
        chain = template | self.llm | self.explanation_parser
        
        # Generate explanation
        explanation = chain.invoke(input_data)
        
        return explanation
    
    def _generate_technical_explanation(self, model, performance_metrics, feature_importance):
        """Generate technical explanation using LLM."""
        # TODO: Implement technical explanation generation
        template = self.explanation_templates['technical']
        
        # Prepare detailed input data
        input_data = {
            'model_type': type(model).__name__,
            'accuracy': performance_metrics['accuracy'],
            'precision': performance_metrics['precision'],
            'recall': performance_metrics['recall'],
            'f1_score': performance_metrics['f1_score'],
            'cross_validation': f"{performance_metrics['cv_mean']:.3f} ± {performance_metrics['cv_std']:.3f}",
            'feature_importance': self._format_feature_importance(feature_importance),
            'dataset_info': f"Test set size: {performance_metrics['sample_size']} samples"
        }
        
        # Create chain and generate
        chain = template | self.llm | StrOutputParser()
        explanation = chain.invoke(input_data)
        
        return explanation
    
    def _generate_business_explanation(self, model, performance_metrics, feature_importance, business_context):
        """Generate business-focused explanation using LLM."""
        # TODO: Implement business explanation generation
        template = self.explanation_templates['business']
        
        # Prepare business-focused input data
        input_data = {
            'model_type': type(model).__name__,
            'business_context': business_context or "Automated decision making system",
            'performance_metrics': f"Accuracy: {performance_metrics['accuracy']:.1%}, Reliability: {performance_metrics['cv_mean']:.1%}",
            'key_features': ', '.join(feature_importance['top_5_features']),
            'impact_analysis': self._generate_impact_analysis(performance_metrics)
        }
        
        # Create chain and generate
        chain = template | self.llm | StrOutputParser()
        explanation = chain.invoke(input_data)
        
        return explanation
    
    def _format_feature_importance(self, feature_importance):
        """Format feature importance for display."""
        if not feature_importance['rankings']:
            return "Feature importance not available"
        
        formatted = []
        for i, feature_data in enumerate(feature_importance['rankings'][:5], 1):
            formatted.append(f"{i}. {feature_data['feature']}: {feature_data['importance']:.3f}")
        
        return "\\n".join(formatted)
    
    def _generate_impact_analysis(self, performance_metrics):
        """Generate impact analysis for business context."""
        accuracy = performance_metrics['accuracy']
        cv_std = performance_metrics['cv_std']
        
        if accuracy > 0.90 and cv_std < 0.03:
            return "High reliability suitable for automated decision making"
        elif accuracy > 0.80:
            return "Good reliability suitable for decision support with human oversight"
        else:
            return "Moderate reliability requiring significant human oversight"
    
    def _generate_fallback_explanation(self, performance_metrics, feature_importance, business_context):
        """Generate fallback explanation when LLM is not available."""
        # TODO: Implement comprehensive fallback explanation
        accuracy = performance_metrics['accuracy']
        f1 = performance_metrics['f1_score']
        cv_mean = performance_metrics['cv_mean']
        cv_std = performance_metrics['cv_std']
        
        fallback_explanation = f"""
        📊 Model Performance Summary (Fallback Mode):
        
        🎯 Performance Metrics:
        • Accuracy: {accuracy:.1%} - {'Excellent' if accuracy > 0.95 else 'Very Good' if accuracy > 0.90 else 'Good' if accuracy > 0.80 else 'Needs Improvement'}
        • F1-Score: {f1:.3f}
        • Cross-validation: {cv_mean:.3f} ± {cv_std:.3f} - {'Stable' if cv_std < 0.02 else 'Moderate' if cv_std < 0.05 else 'Variable'}
        
        🔍 Key Features:
        {', '.join(feature_importance['top_5_features'])}
        
        💼 Business Impact:
        {'This model demonstrates high reliability suitable for production deployment.' if accuracy > 0.90 else 'This model shows good performance but may benefit from additional optimization.' if accuracy > 0.80 else 'This model requires improvement before production deployment.'}
        
        ⚠️ Note: Detailed LLM-powered analysis not available. Consider enabling LLM integration for comprehensive insights.
        """
        
        return fallback_explanation

# Example usage and testing
def test_hybrid_explainer():
    """Test the hybrid model explainer with sample data."""
    # TODO: Implement comprehensive testing
    print("🧪 Testing Hybrid Model Explainer")
    
    # Create sample data
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # Train a sample model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize explainer
    explainer = HybridModelExplainer()
    
    # Generate explanations
    business_context = "Wine quality classification for premium wine selection"
    
    structured_explanation = explainer.explain_model_performance(
        model, X_test, y_test, data.feature_names, 
        business_context, explanation_type="structured"
    )
    
    technical_explanation = explainer.explain_model_performance(
        model, X_test, y_test, data.feature_names,
        explanation_type="technical"
    )
    
    business_explanation = explainer.explain_model_performance(
        model, X_test, y_test, data.feature_names,
        business_context, explanation_type="business"
    )
    
    return {
        'structured': structured_explanation,
        'technical': technical_explanation,
        'business': business_explanation
    }

if __name__ == "__main__":
    test_results = test_hybrid_explainer()
    print("✅ Hybrid Model Explainer testing complete!")
```

#### Task 1.2: Advanced Prompt Engineering for Model Analysis
```python
class AdvancedModelPromptEngine:
    """
    Advanced prompt engineering system for model analysis and interpretation.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt_library = self._build_prompt_library()
    
    def _build_prompt_library(self):
        """Build comprehensive prompt library for different analysis types."""
        # TODO: Implement advanced prompt library
        
        # Performance analysis prompts
        performance_prompts = {
            'accuracy_analysis': PromptTemplate(
                input_variables=["accuracy", "baseline_accuracy", "industry_standard"],
                template="""
                Analyze this model's accuracy performance:
                
                Current Accuracy: {accuracy:.3f}
                Baseline Accuracy: {baseline_accuracy:.3f}
                Industry Standard: {industry_standard:.3f}
                
                Provide detailed analysis of:
                1. How this accuracy compares to baseline and industry standards
                2. Whether this accuracy is sufficient for production deployment
                3. Potential business impact of this accuracy level
                4. Recommendations for improvement if needed
                
                Consider both statistical significance and practical significance.
                """
            ),
            
            'bias_analysis': PromptTemplate(
                input_variables=["fairness_metrics", "protected_attributes", "model_type"],
                template="""
                Analyze potential bias in this {model_type} model:
                
                Fairness Metrics:
                {fairness_metrics}
                
                Protected Attributes: {protected_attributes}
                
                Provide comprehensive bias analysis covering:
                1. Evidence of differential performance across groups
                2. Potential sources of bias in the model
                3. Legal and ethical implications
                4. Mitigation strategies and recommendations
                5. Monitoring recommendations for production
                
                Focus on actionable insights for responsible AI deployment.
                """
            ),
            
            'uncertainty_analysis': PromptTemplate(
                input_variables=["confidence_intervals", "prediction_uncertainty", "calibration_metrics"],
                template="""
                Analyze the uncertainty and calibration of this model:
                
                Confidence Intervals: {confidence_intervals}
                Prediction Uncertainty: {prediction_uncertainty}
                Calibration Metrics: {calibration_metrics}
                
                Provide uncertainty analysis covering:
                1. How well-calibrated the model's confidence estimates are
                2. Reliability of uncertainty estimates for decision making
                3. Scenarios where uncertainty is highest
                4. Recommendations for handling uncertain predictions
                5. Communication strategies for uncertainty to stakeholders
                
                Focus on practical implications for model deployment.
                """
            )
        }
        
        # Business impact prompts
        business_prompts = {
            'roi_analysis': PromptTemplate(
                input_variables=["cost_savings", "implementation_cost", "risk_reduction", "revenue_impact"],
                template="""
                Analyze the business ROI of this ML model deployment:
                
                Projected Cost Savings: {cost_savings}
                Implementation Cost: {implementation_cost}
                Risk Reduction: {risk_reduction}
                Revenue Impact: {revenue_impact}
                
                Provide ROI analysis covering:
                1. Quantitative ROI calculation and payback period
                2. Qualitative benefits beyond financial metrics
                3. Risk factors that could impact ROI
                4. Sensitivity analysis for key assumptions
                5. Recommendations for maximizing business value
                
                Present findings in executive-friendly format.
                """
            ),
            
            'stakeholder_communication': PromptTemplate(
                input_variables=["model_purpose", "performance_summary", "key_features", "limitations"],
                template="""
                Create stakeholder communication for this ML model:
                
                Model Purpose: {model_purpose}
                Performance Summary: {performance_summary}
                Key Predictive Features: {key_features}
                Known Limitations: {limitations}
                
                Generate communication materials for:
                1. Executive summary (2-3 key points)
                2. Technical team briefing (implementation details)
                3. End-user guidance (how to interpret and use)
                4. Compliance documentation (audit trail)
                5. FAQ for common questions
                
                Tailor language appropriately for each audience.
                """
            )
        }
        
        # Model comparison prompts
        comparison_prompts = {
            'algorithm_comparison': PromptTemplate(
                input_variables=["model_results", "evaluation_criteria", "use_case_context"],
                template="""
                Compare these ML algorithms for the given use case:
                
                Model Results:
                {model_results}
                
                Evaluation Criteria: {evaluation_criteria}
                Use Case Context: {use_case_context}
                
                Provide comprehensive comparison covering:
                1. Relative performance across evaluation criteria
                2. Trade-offs between different approaches
                3. Suitability for the specific use case
                4. Implementation complexity and resource requirements
                5. Recommended choice with detailed justification
                
                Consider both technical and business factors in recommendations.
                """
            ),
            
            'ensemble_analysis': PromptTemplate(
                input_variables=["individual_models", "ensemble_performance", "combination_strategy"],
                template="""
                Analyze this ensemble model configuration:
                
                Individual Models: {individual_models}
                Ensemble Performance: {ensemble_performance}
                Combination Strategy: {combination_strategy}
                
                Provide ensemble analysis covering:
                1. Individual model strengths and weaknesses
                2. How the ensemble leverages complementary capabilities
                3. Performance improvement over individual models
                4. Complexity vs performance trade-offs
                5. Recommendations for ensemble optimization
                
                Focus on practical insights for ensemble deployment.
                """
            )
        }
        
        return {
            'performance': performance_prompts,
            'business': business_prompts,
            'comparison': comparison_prompts
        }
    
    def analyze_model_performance(self, performance_data, analysis_type="comprehensive"):
        """
        Generate advanced performance analysis using specialized prompts.
        
        Args:
            performance_data: Dictionary containing performance metrics and context
            analysis_type: Type of analysis (accuracy, bias, uncertainty, comprehensive)
        
        Returns:
            Detailed analysis based on specified type
        """
        # TODO: Implement advanced performance analysis
        pass
    
    def generate_business_impact_analysis(self, model_data, business_context):
        """
        Generate comprehensive business impact analysis.
        
        Args:
            model_data: Model performance and characteristics
            business_context: Business use case and requirements
        
        Returns:
            Business-focused impact analysis
        """
        # TODO: Implement business impact analysis
        pass
    
    def compare_model_alternatives(self, model_comparison_data):
        """
        Generate detailed comparison between model alternatives.
        
        Args:
            model_comparison_data: Performance data for multiple models
        
        Returns:
            Comprehensive model comparison and recommendations
        """
        # TODO: Implement model comparison analysis
        pass
```

### Expected Deliverables
1. Comprehensive hybrid model explanation system
2. Advanced prompt engineering framework for model analysis
3. Multi-modal explanation outputs (technical, business, structured)
4. Fallback explanation system for LLM unavailability

### Success Metrics
- Generate explanations for 100% of trained models
- Achieve >90% stakeholder satisfaction with explanation quality
- Provide actionable insights in <30 seconds per model
- Maintain explanation consistency across different model types

---

## 📊 Exercise 2: Automated ML Pipeline Reporting with Natural Language Generation

### Challenge
Create an automated reporting system that generates comprehensive, business-ready reports for ML pipeline results using advanced NLG techniques.

### Scenario
Multi-model ML pipeline that needs to automatically generate reports for different stakeholders (executives, data scientists, business analysts, compliance officers).

### Tasks

#### Task 2.1: Intelligent Report Generation System
```python
class IntelligentMLReportGenerator:
    """
    Advanced ML pipeline report generation with natural language synthesis.
    """
    
    def __init__(self, llm, report_templates=None):
        self.llm = llm
        self.report_templates = report_templates or self._create_default_templates()
        self.report_history = []
    
    def _create_default_templates(self):
        """Create comprehensive report templates for different audiences."""
        # TODO: Implement sophisticated report templates
        
        executive_template = PromptTemplate(
            input_variables=["project_summary", "key_results", "business_impact", 
                           "recommendations", "timeline", "risks"],
            template="""
            Generate an executive summary report for this ML project:
            
            Project: {project_summary}
            Key Results: {key_results}
            Business Impact: {business_impact}
            Timeline: {timeline}
            
            Create a concise executive report covering:
            
            ## Executive Summary
            - Project objective and business value proposition
            - Key performance achievements and metrics
            - Expected business impact and ROI
            
            ## Strategic Recommendations
            {recommendations}
            
            ## Risk Assessment
            {risks}
            
            ## Next Steps
            - Immediate actions required
            - Resource needs and timeline
            - Success metrics and monitoring plan
            
            Keep the language executive-appropriate, focus on business value, 
            and provide clear action items. Limit to 2 pages maximum.
            """
        )
        
        technical_template = PromptTemplate(
            input_variables=["model_details", "performance_metrics", "technical_challenges", 
                           "architecture_decisions", "validation_results"],
            template="""
            Generate a technical report for the data science team:
            
            Model Details: {model_details}
            Performance Metrics: {performance_metrics}
            Technical Challenges: {technical_challenges}
            Architecture: {architecture_decisions}
            Validation: {validation_results}
            
            Create a comprehensive technical report covering:
            
            ## Model Architecture and Design
            - Algorithm selection rationale
            - Feature engineering approach
            - Hyperparameter optimization strategy
            
            ## Performance Analysis
            - Detailed metric analysis with statistical significance
            - Cross-validation results and stability assessment
            - Comparison with baseline and benchmark models
            
            ## Technical Implementation
            - Code architecture and design patterns
            - Scalability and performance optimizations
            - Infrastructure requirements and dependencies
            
            ## Validation and Testing
            - Validation methodology and results
            - Edge case handling and robustness testing
            - Model interpretability and explainability
            
            ## Future Work
            - Identified improvement opportunities
            - Technical debt and refactoring needs
            - Research directions and experimentation plans
            
            Use technical language appropriate for data scientists and ML engineers.
            """
        )
        
        business_analyst_template = PromptTemplate(
            input_variables=["business_metrics", "user_impact", "process_changes", 
                           "training_needs", "implementation_plan"],
            template="""
            Generate a business analyst report focusing on operational impact:
            
            Business Metrics: {business_metrics}
            User Impact: {user_impact}
            Process Changes: {process_changes}
            Training Needs: {training_needs}
            Implementation: {implementation_plan}
            
            Create a business-focused report covering:
            
            ## Business Process Impact
            - How the model changes current workflows
            - Efficiency gains and process improvements
            - User experience changes and adaptations needed
            
            ## Operational Metrics
            - KPI improvements and measurement methods
            - Quality metrics and monitoring requirements
            - Cost reduction and resource optimization
            
            ## Change Management
            - Training requirements for different user groups
            - Communication plan for stakeholders
            - Timeline for rollout and adoption
            
            ## Risk Mitigation
            - Operational risks and mitigation strategies
            - Contingency plans and fallback procedures
            - Compliance and regulatory considerations
            
            Focus on practical implementation and operational excellence.
            """
        )
        
        compliance_template = PromptTemplate(
            input_variables=["regulatory_requirements", "bias_analysis", "audit_trail", 
                           "data_governance", "risk_assessment"],
            template="""
            Generate a compliance and governance report:
            
            Regulatory Context: {regulatory_requirements}
            Bias Analysis: {bias_analysis}
            Audit Trail: {audit_trail}
            Data Governance: {data_governance}
            Risk Assessment: {risk_assessment}
            
            Create a compliance-focused report covering:
            
            ## Regulatory Compliance
            - Applicable regulations and requirements
            - Compliance verification and evidence
            - Gaps and remediation actions needed
            
            ## Fairness and Bias Assessment
            - Bias testing methodology and results
            - Fairness metrics across protected groups
            - Mitigation strategies for identified issues
            
            ## Data Governance
            - Data lineage and quality documentation
            - Privacy protection and consent management
            - Retention and deletion policies
            
            ## Audit and Monitoring
            - Model monitoring and drift detection
            - Incident response procedures
            - Regular review and validation schedules
            
            ## Risk Management
            - Risk identification and assessment
            - Mitigation strategies and controls
            - Contingency and rollback plans
            
            Ensure all content supports audit and regulatory review processes.
            """
        )
        
        return {
            'executive': executive_template,
            'technical': technical_template,
            'business_analyst': business_analyst_template,
            'compliance': compliance_template
        }
    
    def generate_comprehensive_report(self, pipeline_results, stakeholder_type="executive", 
                                    custom_context=None):
        """
        Generate comprehensive report for specified stakeholder type.
        
        Args:
            pipeline_results: Complete pipeline results and metrics
            stakeholder_type: Target audience (executive, technical, business_analyst, compliance)
            custom_context: Additional context for report customization
        
        Returns:
            Generated report tailored to stakeholder needs
        """
        print(f"📋 Generating {stakeholder_type} report...")
        
        # TODO: Process pipeline results for report generation
        processed_data = self._process_pipeline_results(pipeline_results, stakeholder_type)
        
        # TODO: Select and customize template
        template = self.report_templates[stakeholder_type]
        
        # TODO: Generate report using LLM
        try:
            # Create report generation chain
            chain = template | self.llm | StrOutputParser()
            
            # Generate the report
            report = chain.invoke(processed_data)
            
            # Post-process and format report
            formatted_report = self._format_report(report, stakeholder_type)
            
            # Store in history
            self.report_history.append({
                'stakeholder_type': stakeholder_type,
                'timestamp': datetime.now().isoformat(),
                'report': formatted_report,
                'context': custom_context
            })
            
            return formatted_report
            
        except Exception as e:
            print(f"   ⚠️ Report generation failed: {e}")
            return self._generate_fallback_report(processed_data, stakeholder_type)
    
    def _process_pipeline_results(self, pipeline_results, stakeholder_type):
        """Process pipeline results for specific stakeholder needs."""
        # TODO: Implement stakeholder-specific data processing
        
        # Extract key information based on stakeholder type
        if stakeholder_type == "executive":
            return {
                'project_summary': self._create_project_summary(pipeline_results),
                'key_results': self._extract_key_results(pipeline_results),
                'business_impact': self._calculate_business_impact(pipeline_results),
                'recommendations': self._generate_recommendations(pipeline_results),
                'timeline': self._extract_timeline_info(pipeline_results),
                'risks': self._assess_risks(pipeline_results)
            }
        elif stakeholder_type == "technical":
            return {
                'model_details': self._extract_model_details(pipeline_results),
                'performance_metrics': self._format_technical_metrics(pipeline_results),
                'technical_challenges': self._identify_technical_challenges(pipeline_results),
                'architecture_decisions': self._document_architecture(pipeline_results),
                'validation_results': self._extract_validation_results(pipeline_results)
            }
        elif stakeholder_type == "business_analyst":
            return {
                'business_metrics': self._extract_business_metrics(pipeline_results),
                'user_impact': self._assess_user_impact(pipeline_results),
                'process_changes': self._identify_process_changes(pipeline_results),
                'training_needs': self._assess_training_needs(pipeline_results),
                'implementation_plan': self._create_implementation_plan(pipeline_results)
            }
        elif stakeholder_type == "compliance":
            return {
                'regulatory_requirements': self._identify_regulatory_requirements(pipeline_results),
                'bias_analysis': self._extract_bias_analysis(pipeline_results),
                'audit_trail': self._create_audit_trail(pipeline_results),
                'data_governance': self._document_data_governance(pipeline_results),
                'risk_assessment': self._create_risk_assessment(pipeline_results)
            }
        else:
            raise ValueError(f"Unknown stakeholder type: {stakeholder_type}")
    
    def _create_project_summary(self, pipeline_results):
        """Create executive-level project summary."""
        # TODO: Implement project summary creation
        return "ML pipeline project summary with business context and objectives"
    
    def _extract_key_results(self, pipeline_results):
        """Extract key results for executive reporting."""
        # TODO: Implement key results extraction
        best_model = max(pipeline_results.get('model_results', {}), 
                        key=lambda x: pipeline_results['model_results'][x].get('accuracy', 0))
        best_accuracy = pipeline_results['model_results'][best_model]['accuracy']
        
        return f"Best model: {best_model} with {best_accuracy:.1%} accuracy"
    
    def _calculate_business_impact(self, pipeline_results):
        """Calculate and format business impact."""
        # TODO: Implement business impact calculation
        return "Projected business impact based on model performance and implementation scope"
    
    def _generate_recommendations(self, pipeline_results):
        """Generate executive recommendations."""
        # TODO: Implement recommendation generation
        recommendations = []
        
        # Performance-based recommendations
        best_accuracy = max(r.get('accuracy', 0) for r in pipeline_results.get('model_results', {}).values())
        if best_accuracy > 0.90:
            recommendations.append("Proceed with production deployment")
        elif best_accuracy > 0.80:
            recommendations.append("Consider additional optimization before deployment")
        else:
            recommendations.append("Significant model improvement needed")
        
        return "\\n".join(f"• {rec}" for rec in recommendations)
    
    def _format_report(self, report, stakeholder_type):
        """Format and enhance generated report."""
        # TODO: Implement report formatting
        formatted_report = f"""
# ML Pipeline Report - {stakeholder_type.title()} Summary

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{report}

---
*This report was automatically generated by the ML Pipeline Reporting System*
        """
        
        return formatted_report.strip()
    
    def _generate_fallback_report(self, processed_data, stakeholder_type):
        """Generate fallback report when LLM is unavailable."""
        # TODO: Implement comprehensive fallback reporting
        fallback_report = f"""
# ML Pipeline Report - {stakeholder_type.title()} Summary (Fallback Mode)

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
This report contains key findings from the ML pipeline execution. 
Advanced natural language generation is currently unavailable.

## Key Data Points
{self._format_data_points(processed_data)}

## Next Steps
- Review detailed metrics in technical documentation
- Consider enabling LLM integration for enhanced reporting
- Contact the data science team for detailed interpretation

---
*Fallback report - Enable LLM integration for comprehensive analysis*
        """
        
        return fallback_report
    
    def _format_data_points(self, data):
        """Format data points for fallback report."""
        formatted_points = []
        for key, value in data.items():
            formatted_points.append(f"• {key.replace('_', ' ').title()}: {value}")
        return "\\n".join(formatted_points)
    
    def generate_multi_stakeholder_report_package(self, pipeline_results):
        """
        Generate complete report package for all stakeholder types.
        
        Args:
            pipeline_results: Complete pipeline results
        
        Returns:
            Dictionary containing reports for all stakeholder types
        """
        print("📊 Generating multi-stakeholder report package...")
        
        report_package = {}
        stakeholder_types = ['executive', 'technical', 'business_analyst', 'compliance']
        
        for stakeholder_type in stakeholder_types:
            try:
                report = self.generate_comprehensive_report(pipeline_results, stakeholder_type)
                report_package[stakeholder_type] = {
                    'report': report,
                    'generated_at': datetime.now().isoformat(),
                    'status': 'success'
                }
                print(f"   ✅ {stakeholder_type} report generated")
            except Exception as e:
                print(f"   ❌ {stakeholder_type} report failed: {e}")
                report_package[stakeholder_type] = {
                    'report': None,
                    'error': str(e),
                    'status': 'failed'
                }
        
        return report_package

# Additional helper methods would be implemented for each stakeholder-specific data extraction
```

### Expected Deliverables
1. Multi-stakeholder report generation system
2. Stakeholder-specific report templates
3. Natural language generation for technical content
4. Automated report package creation

### Success Metrics
- Generate reports for 4 stakeholder types automatically
- Achieve >85% stakeholder satisfaction with report quality
- Complete report package generation in <2 minutes
- Maintain consistent reporting format across different pipeline results

---

## 🎯 Exercise 3: Interactive Model Explanation Dashboard

### Challenge
Build an interactive dashboard that combines ML model performance with real-time LLM-powered explanations and allows users to explore model behavior dynamically.

### Scenario
Production ML system where users need to understand individual predictions and overall model behavior through an interactive interface.

### Tasks

#### Task 3.1: Interactive Explanation Interface
```python
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class InteractiveModelDashboard:
    """
    Interactive dashboard for model exploration and explanation.
    """
    
    def __init__(self, model, explainer, data):
        self.model = model
        self.explainer = explainer
        self.data = data
        self.explanation_cache = {}
    
    def create_dashboard(self):
        """Create comprehensive interactive dashboard."""
        st.set_page_config(
            page_title="ML Model Explanation Dashboard",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 Interactive ML Model Explanation Dashboard")
        st.markdown("---")
        
        # Sidebar for controls
        self._create_sidebar()
        
        # Main dashboard layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._create_main_visualization_area()
        
        with col2:
            self._create_explanation_panel()
        
        # Bottom section for detailed analysis
        st.markdown("---")
        self._create_detailed_analysis_section()
    
    def _create_sidebar(self):
        """Create interactive sidebar controls."""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Model selection (if multiple models available)
        if hasattr(self, 'multiple_models'):
            selected_model = st.sidebar.selectbox(
                "Select Model:",
                options=list(self.multiple_models.keys())
            )
            self.current_model = self.multiple_models[selected_model]
        
        # Explanation type selection
        explanation_type = st.sidebar.radio(
            "Explanation Type:",
            ["Individual Prediction", "Feature Importance", "Model Performance", "Comparative Analysis"]
        )
        
        # Data filtering options
        st.sidebar.subheader("🔍 Data Filters")
        
        # Dynamic filtering based on data
        if isinstance(self.data, pd.DataFrame):
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns[:5]:  # Limit to top 5 for UI
                min_val = float(self.data[col].min())
                max_val = float(self.data[col].max())
                
                range_val = st.sidebar.slider(
                    f"{col}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"filter_{col}"
                )
        
        # Sample selection for individual prediction
        if explanation_type == "Individual Prediction":
            sample_idx = st.sidebar.number_input(
                "Sample Index:",
                min_value=0,
                max_value=len(self.data) - 1,
                value=0
            )
            st.session_state['selected_sample'] = sample_idx
        
        # Store selected explanation type
        st.session_state['explanation_type'] = explanation_type
    
    def _create_main_visualization_area(self):
        """Create main visualization area."""
        explanation_type = st.session_state.get('explanation_type', 'Model Performance')
        
        st.subheader("📊 Model Analysis")
        
        if explanation_type == "Individual Prediction":
            self._create_individual_prediction_viz()
        elif explanation_type == "Feature Importance":
            self._create_feature_importance_viz()
        elif explanation_type == "Model Performance":
            self._create_performance_viz()
        elif explanation_type == "Comparative Analysis":
            self._create_comparative_viz()
    
    def _create_individual_prediction_viz(self):
        """Create individual prediction visualization."""
        sample_idx = st.session_state.get('selected_sample', 0)
        
        # Get sample data
        sample = self.data.iloc[sample_idx:sample_idx+1]
        prediction = self.model.predict(sample)[0]
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(sample)[0]
        else:
            probabilities = None
        
        # Display prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", f"{prediction}")
        
        with col2:
            if probabilities is not None:
                confidence = max(probabilities)
                st.metric("Confidence", f"{confidence:.2%}")
        
        with col3:
            st.metric("Sample Index", sample_idx)
        
        # Feature values visualization
        feature_values = sample.iloc[0]
        
        fig = go.Figure(data=go.Bar(
            x=feature_values.values,
            y=feature_values.index,
            orientation='h',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Feature Values for Selected Sample",
            xaxis_title="Feature Value",
            yaxis_title="Features",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # SHAP-style explanation if available
        try:
            self._create_shap_explanation(sample, sample_idx)
        except Exception as e:
            st.warning(f"Advanced explanation unavailable: {e}")
    
    def _create_shap_explanation(self, sample, sample_idx):
        """Create SHAP-style explanation visualization."""
        # TODO: Implement SHAP explanation visualization
        st.subheader("🎯 Feature Contribution Analysis")
        
        # Simulate SHAP values (replace with actual SHAP implementation)
        feature_names = sample.columns
        shap_values = np.random.normal(0, 1, len(feature_names))
        
        # Create waterfall-style chart
        fig = go.Figure()
        
        cumulative = 0
        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
            fig.add_trace(go.Bar(
                x=[feature],
                y=[shap_val],
                name=f"{feature}: {shap_val:.3f}",
                marker_color='red' if shap_val < 0 else 'green'
            ))
        
        fig.update_layout(
            title="Feature Contributions to Prediction",
            xaxis_title="Features",
            yaxis_title="SHAP Value",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_feature_importance_viz(self):
        """Create feature importance visualization."""
        st.subheader("🔍 Feature Importance Analysis")
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = self.data.columns
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            # Create horizontal bar chart
            fig = px.bar(
                importance_df.tail(10),  # Top 10 features
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features",
                color='importance',
                color_continuous_scale='viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            st.subheader("📋 Complete Feature Importance Rankings")
            st.dataframe(
                importance_df.sort_values('importance', ascending=False),
                use_container_width=True
            )
        else:
            st.warning("Feature importance not available for this model type")
    
    def _create_performance_viz(self):
        """Create model performance visualization."""
        st.subheader("📈 Model Performance Overview")
        
        # Performance metrics
        y_true = self.data['target'] if 'target' in self.data.columns else None
        
        if y_true is not None:
            y_pred = self.model.predict(self.data.drop('target', axis=1))
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                st.metric("F1-Score", f"{f1:.3f}")
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                color_continuous_scale='Blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Target variable not available for performance calculation")
    
    def _create_comparative_viz(self):
        """Create comparative analysis visualization."""
        st.subheader("⚖️ Comparative Model Analysis")
        
        if hasattr(self, 'multiple_models'):
            # Compare multiple models
            comparison_data = []
            
            for model_name, model in self.multiple_models.items():
                # Calculate performance for each model
                if 'target' in self.data.columns:
                    y_true = self.data['target']
                    y_pred = model.predict(self.data.drop('target', axis=1))
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average='weighted')
                    
                    comparison_data.append({
                        'Model': model_name,
                        'Accuracy': accuracy,
                        'F1-Score': f1
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create comparison chart
            fig = px.bar(
                comparison_df,
                x='Model',
                y=['Accuracy', 'F1-Score'],
                barmode='group',
                title="Model Performance Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Multiple models not available for comparison")
    
    def _create_explanation_panel(self):
        """Create LLM-powered explanation panel."""
        st.subheader("🧠 AI-Powered Insights")
        
        explanation_type = st.session_state.get('explanation_type', 'Model Performance')
        
        # Generate explanation based on current view
        if st.button("Generate Explanation", type="primary"):
            with st.spinner("Generating AI explanation..."):
                explanation = self._generate_contextual_explanation(explanation_type)
                st.session_state['current_explanation'] = explanation
        
        # Display current explanation
        if 'current_explanation' in st.session_state:
            st.markdown("### 💡 Analysis")
            st.markdown(st.session_state['current_explanation'])
        
        # Quick insights section
        st.markdown("### ⚡ Quick Insights")
        self._display_quick_insights()
    
    def _generate_contextual_explanation(self, explanation_type):
        """Generate contextual explanation based on current view."""
        # TODO: Implement contextual explanation generation
        
        if explanation_type == "Individual Prediction":
            sample_idx = st.session_state.get('selected_sample', 0)
            return self._explain_individual_prediction(sample_idx)
        elif explanation_type == "Feature Importance":
            return self._explain_feature_importance()
        elif explanation_type == "Model Performance":
            return self._explain_model_performance()
        elif explanation_type == "Comparative Analysis":
            return self._explain_comparative_analysis()
        else:
            return "Explanation not available for this view."
    
    def _explain_individual_prediction(self, sample_idx):
        """Explain individual prediction."""
        # TODO: Implement individual prediction explanation
        sample = self.data.iloc[sample_idx]
        prediction = self.model.predict(sample.values.reshape(1, -1))[0]
        
        # Use cached explanation if available
        cache_key = f"individual_{sample_idx}"
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        # Generate new explanation
        if self.explainer and self.explainer.llm_available:
            explanation = f"""
            For sample #{sample_idx}, the model predicts: **{prediction}**
            
            This prediction is based on the specific feature values of this sample. 
            The most influential features for this prediction appear to be those 
            with the highest absolute values or those that deviate most from the average.
            
            **Key factors influencing this prediction:**
            - Feature values that are significantly above or below average
            - Interactions between multiple features
            - Historical patterns learned during training
            
            **Confidence level:** The model's confidence in this prediction can be 
            assessed by examining the prediction probabilities and feature contributions.
            """
        else:
            explanation = f"Prediction for sample #{sample_idx}: {prediction}. LLM explanation unavailable."
        
        self.explanation_cache[cache_key] = explanation
        return explanation
    
    def _display_quick_insights(self):
        """Display quick insights panel."""
        insights = [
            "🎯 Model shows consistent performance across validation sets",
            "📊 Top features account for 80% of predictive power",
            "⚠️ Monitor for potential bias in protected attributes",
            "🔄 Consider retraining if accuracy drops below 85%"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
    
    def _create_detailed_analysis_section(self):
        """Create detailed analysis section."""
        st.subheader("📋 Detailed Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Model Diagnostics", "Data Analysis", "Performance Trends", "Export Report"])
        
        with tab1:
            self._create_model_diagnostics()
        
        with tab2:
            self._create_data_analysis()
        
        with tab3:
            self._create_performance_trends()
        
        with tab4:
            self._create_export_options()
    
    def _create_model_diagnostics(self):
        """Create model diagnostics section."""
        st.write("### 🔧 Model Diagnostics")
        
        # Model information
        model_info = {
            "Model Type": type(self.model).__name__,
            "Number of Features": len(self.data.columns) - (1 if 'target' in self.data.columns else 0),
            "Training Samples": "Not available in this demo",
            "Model Parameters": len(str(self.model.get_params()))
        }
        
        for key, value in model_info.items():
            st.write(f"**{key}:** {value}")
    
    def _create_export_options(self):
        """Create export options."""
        st.write("### 📤 Export Options")
        
        if st.button("Generate Full Report"):
            report = self._generate_dashboard_report()
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"model_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    def _generate_dashboard_report(self):
        """Generate comprehensive dashboard report."""
        report = f"""
# ML Model Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Model Type: {type(self.model).__name__}
- Number of Features: {len(self.data.columns) - (1 if 'target' in self.data.columns else 0)}

## Current Analysis
- Selected View: {st.session_state.get('explanation_type', 'Not specified')}
- Generated Explanation: {st.session_state.get('current_explanation', 'None')}

## Summary
This report was generated from the Interactive Model Explanation Dashboard.
For detailed analysis, please refer to the dashboard interface.

---
Report generated by ML Model Dashboard
        """
        return report

# Streamlit app runner
def run_dashboard():
    """Run the interactive dashboard."""
    # TODO: Initialize with actual model and data
    # This would typically load a trained model and dataset
    
    st.write("Dashboard would run here with actual model and data")
    st.write("This is a template showing the dashboard structure")

if __name__ == "__main__":
    run_dashboard()
```

### Expected Deliverables
1. Interactive model explanation dashboard with multiple views
2. Real-time LLM integration for contextual explanations
3. Dynamic visualization system with user controls
4. Export functionality for reports and insights

### Success Metrics
- Support multiple explanation types (individual, feature importance, performance)
- Generate explanations in <5 seconds with LLM integration
- Achieve >90% user satisfaction with dashboard usability
- Enable export of comprehensive analysis reports

---

## 🏅 Evaluation Criteria

### Technical Implementation (30%)
- LangChain integration quality and error handling
- Prompt engineering sophistication and effectiveness
- Code quality, documentation, and testing
- Performance optimization and caching strategies

### Explanation Quality (25%)
- Accuracy and relevance of generated explanations
- Appropriateness for different stakeholder types
- Consistency across different models and scenarios
- Actionable insights and recommendations

### User Experience (25%)
- Interface design and usability
- Explanation clarity and accessibility
- Interactive features and responsiveness
- Multi-stakeholder support and customization

### Innovation and Creativity (20%)
- Novel approaches to hybrid evaluation
- Creative prompt engineering techniques
- Advanced visualization and reporting features
- Integration of multiple explanation modalities

---

## 📚 Advanced Resources

### LangChain Documentation
- Advanced Prompt Engineering: https://python.langchain.com/docs/modules/model_io/prompts/
- Output Parsers: https://python.langchain.com/docs/modules/model_io/output_parsers/
- Chains and Workflows: https://python.langchain.com/docs/modules/chains/

### Model Interpretation
- SHAP Documentation: https://shap.readthedocs.io/
- LIME for Explanations: https://lime-ml.readthedocs.io/
- Model Interpretability Guide: https://christophm.github.io/interpretable-ml-book/

### Natural Language Generation
- Advanced NLG Techniques
- Business Report Generation
- Multi-stakeholder Communication

---

## 🎯 Submission Guidelines

### File Structure
```
hybrid_evaluation_exercises/
├── exercise_1_model_interpretation/
│   ├── hybrid_model_explainer.py
│   ├── advanced_prompt_engine.py
│   ├── explanation_templates.py
│   └── model_interpretation_report.md
├── exercise_2_automated_reporting/
│   ├── intelligent_report_generator.py
│   ├── stakeholder_templates.py
│   ├── report_processing.py
│   └── automated_reporting_analysis.md
├── exercise_3_interactive_dashboard/
│   ├── interactive_dashboard.py
│   ├── dashboard_components.py
│   ├── visualization_utils.py
│   └── dashboard_user_guide.md
└── README.md
```

### Integration Requirements
- Demonstrate successful LangChain integration with error handling
- Provide fallback explanations when LLM is unavailable
- Include comprehensive prompt engineering examples
- Show multi-stakeholder explanation capabilities

### Evaluation Metrics
Include quantitative evaluation of:
- Explanation quality (human evaluation scores)
- System performance (response times, success rates)
- User satisfaction (usability testing results)
- Business value (stakeholder feedback)

---

**Ready to build the future of AI-powered ML explanation systems? Let's create some intelligent interfaces! 🧠🚀**