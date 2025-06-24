"""
Week 3: Hybrid ML + LLM Evaluation Flow

This flow combines traditional ML model evaluation with LLM-powered
interpretation and natural language reporting.

Usage:
    python hybrid_evaluation_flow.py run
    python hybrid_evaluation_flow.py run --use_llm True --llm_model llama3.2
"""

from metaflow import FlowSpec, step, Parameter, catch
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# LangChain imports with fallback
try:
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import Ollama
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class HybridEvaluationFlow(FlowSpec):
    """
    Hybrid evaluation flow combining traditional ML metrics with LLM interpretation.
    
    Features:
    - Traditional ML model training and evaluation
    - LLM-powered model interpretation and explanation
    - Natural language performance reports
    - Business-friendly recommendations
    - Hybrid quantitative + qualitative analysis
    """
    
    use_llm = Parameter('use_llm',
                       help='Use LLM for interpretation (requires Ollama)',
                       default=True)
    
    llm_model = Parameter('llm_model',
                         help='LLM model name for Ollama',
                         default='llama3.2')
    
    min_accuracy_threshold = Parameter('min_accuracy_threshold',
                                     help='Minimum acceptable accuracy',
                                     default=0.80)
    
    random_state = Parameter('random_state',
                            help='Random state for reproducibility',
                            default=42)
    
    @step
    def start(self):
        """
        Initialize the hybrid evaluation pipeline.
        """
        print(f"🚀 Starting Hybrid ML + LLM Evaluation Flow")
        print(f"   Use LLM: {self.use_llm}")
        print(f"   LLM Model: {self.llm_model}")
        print(f"   Min Accuracy: {self.min_accuracy_threshold}")
        print(f"   LangChain Available: {LANGCHAIN_AVAILABLE}")
        
        # Load wine dataset
        wine_data = load_wine()
        self.X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
        self.y = wine_data.target
        self.target_names = wine_data.target_names
        self.feature_names = wine_data.feature_names
        
        print(f"📊 Dataset: Wine Classification")
        print(f"   Samples: {self.X.shape[0]}")
        print(f"   Features: {self.X.shape[1]}")
        print(f"   Classes: {len(self.target_names)}")
        
        # Define models to evaluate
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        """
        Preprocess data for model training.
        """
        print("🔧 Preprocessing data...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, 
            random_state=self.random_state, stratify=self.y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"   📊 Training set: {self.X_train_scaled.shape}")
        print(f"   📊 Test set: {self.X_test_scaled.shape}")
        
        # Store preprocessing info
        self.preprocessing_info = {
            'train_size': self.X_train_scaled.shape[0],
            'test_size': self.X_test_scaled.shape[0],
            'feature_count': self.X_train_scaled.shape[1],
            'class_distribution': {
                'train': np.bincount(self.y_train).tolist(),
                'test': np.bincount(self.y_test).tolist()
            }
        }
        
        self.next(self.train_models)
    
    @step
    def train_models(self):
        """
        Train all models and collect traditional ML metrics.
        """
        print("🏋️ Training models and collecting metrics...")
        
        self.model_results = {}
        
        for name, model in self.models.items():
            print(f"\\n🔄 Training {name}...")
            
            start_time = datetime.now()
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, 'predict_proba') else None
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            
            # Classification report
            class_report = classification_report(
                self.y_test, y_pred, 
                target_names=self.target_names,
                output_dict=True
            )
            
            # Confusion matrix
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = dict(zip(self.feature_names, importances))
                feature_importance = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True))
            elif hasattr(model, 'coef_'):
                coefficients = np.abs(model.coef_)
                if coefficients.ndim > 1:
                    coefficients = coefficients.mean(axis=0)
                feature_importance = dict(zip(self.feature_names, coefficients))
                feature_importance = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True))
            
            # Store results
            self.model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'feature_importance': feature_importance,
                'training_time': training_time,
                'meets_threshold': accuracy >= self.min_accuracy_threshold
            }
            
            print(f"   ✅ Accuracy: {accuracy:.3f} | CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
            print(f"   {'✅' if accuracy >= self.min_accuracy_threshold else '❌'} Meets threshold: {accuracy >= self.min_accuracy_threshold}")
        
        # Find best model
        self.best_model_name = max(self.model_results.keys(), 
                                  key=lambda x: self.model_results[x]['accuracy'])
        self.best_model_results = self.model_results[self.best_model_name]
        
        print(f"\\n🏆 Best model: {self.best_model_name}")
        print(f"   📊 Accuracy: {self.best_model_results['accuracy']:.3f}")
        
        self.next(self.llm_interpretation)
    
    @catch(var='llm_error')
    @step
    def llm_interpretation(self):
        """
        Generate LLM-powered model interpretation and explanations.
        """
        print("🤖 Generating LLM interpretations...")
        
        self.llm_interpretations = {}
        self.llm_available = False
        
        if self.use_llm and LANGCHAIN_AVAILABLE:
            try:
                # Initialize LLM
                llm = Ollama(model=self.llm_model)
                self.llm_available = True
                print(f"   ✅ Connected to {self.llm_model}")
                
                # Create interpretation for each model
                for model_name, results in self.model_results.items():
                    print(f"   🔍 Interpreting {model_name}...")
                    
                    interpretation = self.generate_llm_interpretation(
                        llm, model_name, results
                    )
                    self.llm_interpretations[model_name] = interpretation
                
                # Generate overall comparison
                self.overall_comparison = self.generate_overall_comparison(llm)
                
                print(f"   ✅ LLM interpretations complete")
                
            except Exception as e:
                print(f"   ⚠️ LLM interpretation failed: {str(e)}")
                print(f"   📝 Falling back to rule-based interpretation...")
                self.llm_error = str(e)
                self.generate_fallback_interpretations()
        else:
            print("   📝 Using rule-based interpretation (LLM disabled or unavailable)")
            self.generate_fallback_interpretations()
        
        self.next(self.generate_hybrid_report)
    
    def generate_llm_interpretation(self, llm, model_name, results):
        """
        Generate LLM interpretation for a single model.
        """
        # Prepare model performance data
        accuracy = results['accuracy']
        cv_mean = results['cv_mean']
        cv_std = results['cv_std']
        meets_threshold = results['meets_threshold']
        
        # Prepare feature importance info
        feature_info = ""
        if results['feature_importance']:
            top_features = list(results['feature_importance'].items())[:3]
            feature_info = f"Top 3 features: {', '.join([f'{name} ({imp:.3f})' for name, imp in top_features])}"
        else:
            feature_info = "Feature importance not available for this model type."
        
        # Create interpretation prompt
        prompt_template = PromptTemplate(
            input_variables=["model_name", "accuracy", "cv_mean", "cv_std", 
                           "meets_threshold", "threshold", "feature_info"],
            template="""
            Analyze this machine learning model for wine classification:
            
            Model: {model_name}
            Test Accuracy: {accuracy:.3f}
            Cross-validation: {cv_mean:.3f} ± {cv_std:.3f}
            Meets threshold ({threshold}): {meets_threshold}
            {feature_info}
            
            Please provide a business-friendly analysis covering:
            1. Performance assessment (excellent/good/fair/poor)
            2. Reliability and consistency of the model
            3. Key strengths and potential concerns
            4. Practical recommendations for deployment
            
            Focus on actionable insights for business stakeholders.
            Keep the response concise and practical (max 4 paragraphs).
            """
        )
        
        # Create chain and generate interpretation
        chain = prompt_template | llm | StrOutputParser()
        
        interpretation = chain.invoke({
            "model_name": model_name,
            "accuracy": accuracy,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "meets_threshold": meets_threshold,
            "threshold": self.min_accuracy_threshold,
            "feature_info": feature_info
        })
        
        return interpretation.strip()
    
    def generate_overall_comparison(self, llm):
        """
        Generate LLM-powered overall model comparison.
        """
        # Prepare comparison data
        model_summary = []
        for name, results in self.model_results.items():
            model_summary.append(f"{name}: {results['accuracy']:.3f} accuracy, "
                               f"CV {results['cv_mean']:.3f}±{results['cv_std']:.3f}")
        
        models_text = "; ".join(model_summary)
        best_model = self.best_model_name
        
        prompt_template = PromptTemplate(
            input_variables=["models_summary", "best_model", "threshold"],
            template="""
            Compare these wine classification models:
            
            {models_summary}
            
            Best performing model: {best_model}
            Acceptance threshold: {threshold}
            
            Provide a strategic comparison focusing on:
            1. Overall model performance landscape
            2. Trade-offs between models (accuracy vs. interpretability vs. speed)
            3. Business recommendation for model selection
            4. Next steps for improvement or deployment
            
            Write for business stakeholders who need to make deployment decisions.
            Be concise and actionable (max 3 paragraphs).
            """
        )
        
        chain = prompt_template | llm | StrOutputParser()
        
        comparison = chain.invoke({
            "models_summary": models_text,
            "best_model": best_model,
            "threshold": self.min_accuracy_threshold
        })
        
        return comparison.strip()
    
    def generate_fallback_interpretations(self):
        """
        Generate rule-based interpretations when LLM is not available.
        """
        self.llm_interpretations = {}
        
        for model_name, results in self.model_results.items():
            accuracy = results['accuracy']
            cv_std = results['cv_std']
            meets_threshold = results['meets_threshold']
            
            # Performance assessment
            if accuracy > 0.95:
                performance = "Excellent"
            elif accuracy > 0.90:
                performance = "Very Good"
            elif accuracy > 0.80:
                performance = "Good"
            elif accuracy > 0.70:
                performance = "Fair"
            else:
                performance = "Poor"
            
            # Reliability assessment
            reliability = "High" if cv_std < 0.02 else "Moderate" if cv_std < 0.05 else "Low"
            
            # Generate interpretation
            interpretation = f"""
            Performance Assessment: {performance} (Test Accuracy: {accuracy:.3f})
            
            The {model_name} model shows {performance.lower()} performance with {reliability.lower()} reliability 
            (CV: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}). 
            {'This model meets' if meets_threshold else 'This model does not meet'} the minimum accuracy threshold of {self.min_accuracy_threshold}.
            
            {'Recommendation: Ready for production deployment with appropriate monitoring.' if accuracy > 0.90 and cv_std < 0.03 else 'Recommendation: Consider improvement through feature engineering or hyperparameter tuning before deployment.' if accuracy > 0.80 else 'Recommendation: Significant improvement needed before production consideration.'}
            
            {'Key features driving predictions are available for interpretation.' if results['feature_importance'] else 'Model lacks feature importance information for interpretability.'}
            """
            
            self.llm_interpretations[model_name] = interpretation.strip()
        
        # Overall comparison
        best_acc = self.best_model_results['accuracy']
        model_count = len(self.model_results)
        
        self.overall_comparison = f"""
        Model Comparison Summary: Evaluated {model_count} models with best performance of {best_acc:.3f} 
        achieved by {self.best_model_name}. 
        
        {'Multiple models meet the performance threshold, indicating robust predictive capability.' if sum(r['meets_threshold'] for r in self.model_results.values()) > 1 else 'Limited models meet the performance threshold, suggesting need for improvement.'}
        
        Business Recommendation: {'Deploy the best model with monitoring systems.' if best_acc > 0.90 else 'Improve model performance before production deployment.'}
        """
    
    @step
    def generate_hybrid_report(self):
        """
        Generate comprehensive hybrid report combining ML metrics and LLM insights.
        """
        print("📄 Generating hybrid evaluation report...")
        
        # Create comprehensive report structure
        self.hybrid_report = {
            'executive_summary': self.create_executive_summary(),
            'quantitative_analysis': self.create_quantitative_analysis(),
            'qualitative_insights': self.create_qualitative_insights(),
            'recommendations': self.create_recommendations(),
            'technical_details': self.create_technical_details()
        }
        
        # Generate formatted report text
        self.formatted_report = self.format_report()
        
        print("   ✅ Hybrid report generated")
        
        self.next(self.end)
    
    def create_executive_summary(self):
        """
        Create executive summary section.
        """
        best_acc = self.best_model_results['accuracy']
        models_meeting_threshold = sum(r['meets_threshold'] for r in self.model_results.values())
        
        return {
            'best_model': self.best_model_name,
            'best_accuracy': best_acc,
            'models_evaluated': len(self.model_results),
            'models_meeting_threshold': models_meeting_threshold,
            'performance_level': (
                'Excellent' if best_acc > 0.95 else
                'Very Good' if best_acc > 0.90 else
                'Good' if best_acc > 0.80 else
                'Needs Improvement'
            ),
            'recommendation': (
                'Ready for production' if best_acc > 0.90 else
                'Needs optimization' if best_acc > 0.80 else
                'Requires significant improvement'
            )
        }
    
    def create_quantitative_analysis(self):
        """
        Create quantitative analysis section.
        """
        model_performances = []
        for name, results in self.model_results.items():
            model_performances.append({
                'model': name,
                'accuracy': results['accuracy'],
                'cv_mean': results['cv_mean'],
                'cv_std': results['cv_std'],
                'training_time': results['training_time'],
                'meets_threshold': results['meets_threshold']
            })
        
        # Sort by accuracy
        model_performances.sort(key=lambda x: x['accuracy'], reverse=True)
        
        accuracies = [r['accuracy'] for r in self.model_results.values()]
        
        return {
            'model_rankings': model_performances,
            'statistical_summary': {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies)
            },
            'best_model_details': {
                'classification_report': self.best_model_results['classification_report'],
                'confusion_matrix': self.best_model_results['confusion_matrix'],
                'feature_importance': self.best_model_results['feature_importance']
            }
        }
    
    def create_qualitative_insights(self):
        """
        Create qualitative insights section.
        """
        return {
            'llm_available': self.llm_available,
            'model_interpretations': self.llm_interpretations,
            'overall_comparison': self.overall_comparison,
            'key_insights': [
                f"Best performing model: {self.best_model_name}",
                # f"Performance level: {self.hybrid_report['executive_summary']['performance_level'] if 'executive_summary' in self.hybrid_report else 'Unknown'}",
                f"Models meeting threshold: {sum(r['meets_threshold'] for r in self.model_results.values())}/{len(self.model_results)}"
            ]
        }
    
    def create_recommendations(self):
        """
        Create recommendations section.
        """
        recommendations = []
        
        best_acc = self.best_model_results['accuracy']
        cv_std = self.best_model_results['cv_std']
        
        # Performance-based recommendations
        if best_acc > 0.95 and cv_std < 0.02:
            recommendations.extend([
                "Deploy best model to production with confidence",
                "Implement real-time monitoring and alerting",
                "Set up A/B testing framework for continuous improvement"
            ])
        elif best_acc > 0.85:
            recommendations.extend([
                "Consider ensemble methods to boost performance",
                "Implement comprehensive testing before deployment",
                "Monitor performance closely in production"
            ])
        else:
            recommendations.extend([
                "Improve model through feature engineering",
                "Collect additional training data",
                "Consider advanced algorithms or neural networks"
            ])
        
        # Always include
        recommendations.extend([
            "Create model interpretation dashboard for stakeholders",
            "Establish model retraining pipeline",
            "Document model assumptions and limitations"
        ])
        
        return recommendations
    
    def create_technical_details(self):
        """
        Create technical details section.
        """
        return {
            'dataset_info': {
                'name': 'Wine Classification',
                'samples': self.X.shape[0],
                'features': self.X.shape[1],
                'classes': len(self.target_names)
            },
            'preprocessing_info': self.preprocessing_info,
            'model_configurations': {name: str(model) for name, model in self.models.items()},
            'evaluation_parameters': {
                'test_size': 0.2,
                'cv_folds': 5,
                'min_accuracy_threshold': self.min_accuracy_threshold,
                'random_state': self.random_state
            }
        }
    
    def format_report(self):
        """
        Format the report as readable text.
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            "🏆 HYBRID ML + LLM EVALUATION REPORT",
            "=" * 45,
            f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"🤖 LLM Integration: {'Enabled' if self.llm_available else 'Disabled'}",
            ""
        ])
        
        # Executive Summary
        exec_sum = self.hybrid_report['executive_summary']
        report_lines.extend([
            "📋 EXECUTIVE SUMMARY",
            "-" * 20,
            f"🏆 Best Model: {exec_sum['best_model']}",
            f"📊 Best Accuracy: {exec_sum['best_accuracy']:.3f}",
            f"⭐ Performance Level: {exec_sum['performance_level']}",
            f"✅ Models Meeting Threshold: {exec_sum['models_meeting_threshold']}/{exec_sum['models_evaluated']}",
            f"🚀 Recommendation: {exec_sum['recommendation']}",
            ""
        ])
        
        # Quantitative Analysis
        quant = self.hybrid_report['quantitative_analysis']
        report_lines.extend([
            "📊 QUANTITATIVE ANALYSIS",
            "-" * 25,
            "Model Performance Ranking:"
        ])
        
        for i, model in enumerate(quant['model_rankings'], 1):
            status = "✅" if model['meets_threshold'] else "❌"
            report_lines.append(
                f"   {i}. {model['model']}: {model['accuracy']:.3f} "
                f"(CV: {model['cv_mean']:.3f}±{model['cv_std']:.3f}) {status}"
            )
        
        report_lines.append("")
        
        # Qualitative Insights
        qual = self.hybrid_report['qualitative_insights']
        report_lines.extend([
            "🧠 QUALITATIVE INSIGHTS",
            "-" * 23,
            f"Analysis Method: {'LLM-Powered' if qual['llm_available'] else 'Rule-Based'}",
            ""
        ])
        
        if qual['overall_comparison']:
            report_lines.extend([
                "Overall Assessment:",
                qual['overall_comparison'],
                ""
            ])
        
        # Best Model Interpretation
        if self.best_model_name in qual['model_interpretations']:
            report_lines.extend([
                f"Best Model Analysis ({self.best_model_name}):",
                qual['model_interpretations'][self.best_model_name],
                ""
            ])
        
        # Recommendations
        recommendations = self.hybrid_report['recommendations']
        report_lines.extend([
            "🚀 RECOMMENDATIONS",
            "-" * 17
        ])
        
        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"   {i}. {rec}")
        
        report_lines.extend([
            "",
            "=" * 45,
            "📌 Report generated by Hybrid ML + LLM Evaluation Pipeline"
        ])
        
        return "\\n".join(report_lines)
    
    @step
    def end(self):
        """
        Finalize hybrid evaluation and display results.
        """
        print("🎉 Hybrid ML + LLM Evaluation Complete!")
        print("=" * 45)
        
        exec_sum = self.hybrid_report['executive_summary']
        
        print(f"🏆 Best Model: {exec_sum['best_model']}")
        print(f"📊 Best Accuracy: {exec_sum['best_accuracy']:.3f}")
        print(f"⭐ Performance Level: {exec_sum['performance_level']}")
        print(f"🤖 LLM Integration: {'✅ Enabled' if self.llm_available else '❌ Disabled'}")
        
        print(f"\\n📋 Quick Summary:")
        print(f"   • Models Evaluated: {len(self.model_results)}")
        print(f"   • Models Meeting Threshold: {sum(r['meets_threshold'] for r in self.model_results.values())}")
        print(f"   • Recommendation: {exec_sum['recommendation']}")
        
        if self.best_model_results['feature_importance']:
            print(f"\\n🔍 Top 3 Features:")
            top_features = list(self.best_model_results['feature_importance'].items())[:3]
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"   {i}. {feature}: {importance:.3f}")
        
        print(f"\\n📄 Full Report Available:")
        print(f"   • Quantitative metrics: Traditional ML evaluation")
        print(f"   • Qualitative insights: {'LLM-powered' if self.llm_available else 'Rule-based'} interpretation")
        print(f"   • Business recommendations: Actionable next steps")
        
        # Create final summary
        self.final_summary = {
            'status': 'success',
            'hybrid_report': self.hybrid_report,
            'formatted_report': self.formatted_report,
            'model_results': {name: {k: v for k, v in results.items() 
                                   if k not in ['model']}  # Exclude model objects
                            for name, results in self.model_results.items()},
            'llm_integration': {
                'available': self.llm_available,
                'model': self.llm_model if self.llm_available else None,
                'error': getattr(self, 'llm_error', None)
            },
            'parameters': {
                'use_llm': self.use_llm,
                'llm_model': self.llm_model,
                'min_accuracy_threshold': self.min_accuracy_threshold,
                'random_state': self.random_state
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print("\\n✨ Complete hybrid evaluation results saved!")
        print("💡 Access results with:")
        print("   from metaflow import Flow")
        print("   run = Flow('HybridEvaluationFlow').latest_run")
        print("   print(run.data.formatted_report)")


if __name__ == '__main__':
    HybridEvaluationFlow()
