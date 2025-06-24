"""
INRIVA AI Academy - Workshop Introduction Flow
==============================================

This is the basic Metaflow example from our workshop.
Run this to understand Metaflow fundamentals.

Usage:
    python workshop_intro_flow.py run
    python workshop_intro_flow.py show
"""

from metaflow import FlowSpec, step
import pandas as pd
import numpy as np

class WorkshopIntroFlow(FlowSpec):
    """
    Our first Metaflow workflow - demonstrates core concepts
    
    This flow shows:
    - Basic step structure
    - Data artifact management
    - Simple data processing
    - Automatic versioning
    """
    
    @step
    def start(self):
        """
        Initialize our workflow with sample data
        """
        print("🚀 Starting our first Metaflow workflow!")
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = {
            'values': np.random.normal(100, 15, 1000),
            'categories': np.random.choice(['A', 'B', 'C'], 1000),
            'timestamps': pd.date_range('2024-01-01', periods=1000)
        }
        
        print(f"✅ Generated {len(self.sample_data['values'])} data points")
        print("📊 Data preview:")
        print(f"   Values range: {self.sample_data['values'].min():.1f} - {self.sample_data['values'].max():.1f}")
        print(f"   Categories: {np.unique(self.sample_data['categories'])}")
        print(f"   Date range: {self.sample_data['timestamps'][0]} to {self.sample_data['timestamps'][-1]}")
        
        self.next(self.process_data)
    
    @step  
    def process_data(self):
        """
        Process our data and calculate statistics
        """
        print("🔧 Processing data...")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(self.sample_data)
        
        # Calculate comprehensive statistics
        self.statistics = {
            'mean': df['values'].mean(),
            'std': df['values'].std(),
            'median': df['values'].median(),
            'min': df['values'].min(),
            'max': df['values'].max(),
            'count_by_category': df['categories'].value_counts().to_dict(),
            'total_samples': len(df)
        }
        
        # Calculate some derived metrics
        self.statistics['coefficient_of_variation'] = self.statistics['std'] / self.statistics['mean']
        self.statistics['range'] = self.statistics['max'] - self.statistics['min']
        
        print(f"📊 Statistics calculated:")
        print(f"   Mean: {self.statistics['mean']:.2f}")
        print(f"   Std: {self.statistics['std']:.2f}")
        print(f"   CV: {self.statistics['coefficient_of_variation']:.3f}")
        print(f"   Category distribution: {self.statistics['count_by_category']}")
        
        # Store processed DataFrame as artifact
        self.processed_df = df
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Finalize workflow and create summary
        """
        print("🎉 Workflow completed successfully!")
        
        # Create final summary
        self.workflow_summary = {
            'total_samples_processed': self.statistics['total_samples'],
            'processing_steps': ['data_generation', 'statistical_analysis'],
            'key_metrics': {
                'mean_value': self.statistics['mean'],
                'data_variability': self.statistics['coefficient_of_variation']
            },
            'categories_found': len(self.statistics['count_by_category'])
        }
        
        print("📋 Workflow Summary:")
        print(f"   Total samples: {self.workflow_summary['total_samples_processed']}")
        print(f"   Mean value: {self.workflow_summary['key_metrics']['mean_value']:.2f}")
        print(f"   Categories: {self.workflow_summary['categories_found']}")
        print(f"   Processing steps: {len(self.workflow_summary['processing_steps'])}")
        
        print("\n✨ All artifacts saved automatically by Metaflow!")
        print("   Use 'python workshop_intro_flow.py show' to see run details")

if __name__ == '__main__':
    WorkshopIntroFlow()