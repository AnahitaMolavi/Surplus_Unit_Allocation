#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:41:29 2024

@author: anahitamolavi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

def visualize_results(results_df: pd.DataFrame, opt_data: pd.DataFrame):
    """
    Visualize the optimization results.

    Parameters:
    - results_df: DataFrame containing optimal values.
    - opt_data: DataFrame containing parameters.

    Returns:
    - None
    """
    
    if results_df is not None and not results_df.empty:
        plt.hist(results_df['Surplus'], edgecolor='black', alpha=0.7)
        
        plt.title("Histogram of optimal surplus quantities")
        plt.xlabel("Optimal surplus quantity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
    else:
        logging.warning("No results to visualize.")
        

#Placeholder for performing sensitivity analysis        
def perform_sensitivity_analysis():
    
    return None
        
#Placeholder for performing sensitivity analysis        
def output_statistics(results_df: pd.DataFrame, opt_data: pd.DataFrame):

    
    return None
    