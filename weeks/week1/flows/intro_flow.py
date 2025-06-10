"""
Week 1: Introduction to Metaflow
================================

A simple flow to demonstrate Metaflow basics.
"""

from metaflow import FlowSpec, step
import pandas as pd
import numpy as np

class IntroFlow(FlowSpec):
    """
    Introduction to Metaflow concepts
    """
    
    @step
    def start(self):
        """Initialize the workflow"""
        print("🚀 Welcome to Metaflow!")
        self.message = "Hello AI Academy!"
        self.next(self.end)
    
    @step
    def end(self):
        """Complete the workflow"""
        print(f"✅ {self.message}")
        print("🎉 Your first Metaflow workflow is complete!")

if __name__ == '__main__':
    IntroFlow()
