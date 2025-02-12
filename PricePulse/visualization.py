################################################################################
# Name: visualization.py
# Purpose: price elasticities visualization
# Date                          Version                Created By
# 5-Dec-2024                   1.0         Rajesh Kumar Jena(Initial Version)
################################################################################

import matplotlib.pyplot as plt
import seaborn as sns

def plot_elasticity_curve(prices, demands, lifecycle_stage):
    plt.figure()
    plt.plot(prices, demands, label=f'{lifecycle_stage}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Price (log)')
    plt.ylabel('Demand (log)')
    plt.title('Price Elasticity Curve')
    plt.legend()
    plt.savefig(f'elasticity_{lifecycle_stage}.png')
    
  