################################################################################
# Name: optimizer.py
# Purpose: quadratic optimization of revenue , profit and inventory turn
# Date                          Version                Created By
# 5-Dec-2024                   1.0         Rajesh Kumar Jena(Initial Version)
################################################################################

from scipy.optimize import minimize
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

class PriceOptimizer:
    def __init__(self, model, cost, inventory_constraint):
        self.model = model
        self.cost = cost  # Cost per SKU
        self.inventory = inventory_constraint  # Max inventory

    def profit_objective(self, price):
        demand = self.model.predict_demand(price)
        revenue = price * demand
        profit = revenue - self.cost * demand
        return -profit  # Minimize negative profit

    def optimize_price(self, initial_price):
        constraints = ({'type': 'ineq', 'fun': lambda x: self.inventory - self.model.predict_demand(x)})
        result = minimize(self.profit_objective, initial_price, constraints=constraints)
        return result.x


class PriceOptimizerExplainable(PriceOptimizer):
    def compare_scenario(self, optimal_price: float, scenario_price: float) -> dict:
        """
        Compare optimal price with a user-provided scenario price
        Returns comparison metrics and explanations
        """
        # Calculate key metrics
        comparison = {
            'optimal_price': optimal_price,
            'scenario_price': scenario_price,
            'price_difference_pct': self._calc_percentage_diff(optimal_price, scenario_price),
            'metrics': {
                'optimal': self._calculate_all_metrics(optimal_price),
                'scenario': self._calculate_all_metrics(scenario_price)
            },
            'tradeoffs': self._identify_tradeoffs(optimal_price, scenario_price),
            'constraint_warnings': self._check_constraint_violations(scenario_price)
        }
        
        return comparison

    def _calculate_all_metrics(self, price: float) -> dict:
        """Calculate complete set of business metrics for a given price"""
        demand = self.model.predict_demand(price)
        return {
            'demand': demand,
            'revenue': price * demand,
            'profit': (price - self.cost) * demand,
            'margin_pct': (price - self.cost) / price * 100,
            'inventory_turnover': demand / self.inventory_constraint,
            'price_vs_comp': self._get_price_position(price)
        }

    def _identify_tradeoffs(self, optimal: float, scenario: float) -> list:
        """Identify key tradeoffs between prices"""
        tradeoffs = []
        opt_metrics = self._calculate_all_metrics(optimal)
        scen_metrics = self._calculate_all_metrics(scenario)
        
        if scen_metrics['profit'] > opt_metrics['profit']:
            tradeoffs.append("Scenario price yields higher profit but may sacrifice market share")
        if scen_metrics['inventory_turnover'] > opt_metrics['inventory_turnover']:
            tradeoffs.append("Scenario price clears inventory faster but reduces margin")
        if abs(scen_metrics['price_vs_comp']) > abs(opt_metrics['price_vs_comp']):
            tradeoffs.append("Scenario price creates larger price gap vs competitors")
        
        return tradeoffs

    def _check_constraint_violations(self, price: float) -> list:
        """Check if scenario price violates business constraints"""
        warnings = []
        demand = self.model.predict_demand(price)
        
        if demand > self.inventory_constraint:
            warnings.append(f"Scenario demand ({demand:.0f}) exceeds inventory ({self.inventory_constraint})")
        if price < self.cost * 0.9:
            warnings.append(f"Scenario price is 10% below cost (${self.cost:.2f})")
            
        return warnings

    def scenario_report(self, comparison: dict, style: str = 'markdown') -> str:
        """
        Generate human-readable scenario comparison report
        Styles: markdown, html, text
        """
        report = []
        md = style == 'markdown'
        
        # Header
        report.append(f"# Scenario Analysis Report\n" if md else "SCENARIO ANALYSIS REPORT\n")
        
        # Price Comparison
        report.append(f"## Price Comparison\n" if md else "\nPRICE COMPARISON\n")
        report.append(f"- Optimal Price: ${comparison['optimal_price']:.2f}")
        report.append(f"- Scenario Price: ${comparison['scenario_price']:.2f} ({comparison['price_difference_pct']:+.1f}%)\n")
        
        # Metric Comparison Table
        report.append(f"## Key Metric Comparison\n" if md else "\nKEY METRIC COMPARISON\n")
        report.append(self._generate_metric_table(comparison['metrics'], style))
        
        # Tradeoffs
        report.append(f"## Tradeoff Analysis\n" if md else "\nTRADEOFF ANALYSIS\n")
        for tradeoff in comparison['tradeoffs']:
            report.append(f"- {tradeoff}")
            
        # Warnings
        if comparison['constraint_warnings']:
            report.append(f"## ⚠️ Scenario Warnings\n" if md else "\nSCENARIO WARNINGS\n")
            for warning in comparison['constraint_warnings']:
                report.append(f"- {warning}")
                
        return "\n".join(report)

    def _generate_metric_table(self, metrics: dict, style: str) -> str:
        """Generate formatted comparison table"""
        headers = ["Metric", "Optimal", "Scenario", "Δ"]
        rows = []
        
        for metric in ['profit', 'revenue', 'demand', 'margin_pct', 'inventory_turnover']:
            opt = metrics['optimal'][metric]
            scen = metrics['scenario'][metric]
            delta = self._calc_percentage_diff(opt, scen)
            
            rows.append([
                metric.capitalize(),
                f"${opt:,.2f}" if metric in ['profit','revenue'] else f"{opt:.1f}%",
                f"${scen:,.2f}" if metric in ['profit','revenue'] else f"{scen:.1f}%",
                f"{delta:+.1f}%"
            ])
            
        if style == 'markdown':
            table = ["| " + " | ".join(headers) + " |",
                     "|" + "|".join(["---"]*4) + "|"]
            for row in rows:
                table.append("| " + " | ".join(row) + " |")
            return "\n".join(table)
        else:
            return "\n".join(["\t".join(row) for row in [headers]+rows])

    def visualize_scenario(self, comparison: dict):
        """Visual comparison of optimal vs scenario prices"""
        fig, axs = plt.subplots(1, 2, figsize=(15,6))
        
        # Metric comparison radar chart
        metrics = ['profit', 'revenue', 'demand', 'margin_pct', 'inventory_turnover']
        optimal = [comparison['metrics']['optimal'][m] for m in metrics]
        scenario = [comparison['metrics']['scenario'][m] for m in metrics]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        axs[0].plot(angles, optimal, 'b-', label='Optimal')
        axs[0].plot(angles, scenario, 'r--', label='Scenario')
        axs[0].set_xticks(angles)
        axs[0].set_xticklabels(metrics)
        axs[0].set_title('Performance Comparison')
        
        # Price-demand curve with markers
        prices = np.linspace(0.8*min_price, 1.2*max_price, 100)
        demands = [self.model.predict_demand(p) for p in prices]
        axs[1].plot(prices, demands)
        axs[1].scatter(comparison['optimal_price'], comparison['metrics']['optimal']['demand'], 
                      c='blue', s=100, label='Optimal')
        axs[1].scatter(comparison['scenario_price'], comparison['metrics']['scenario']['demand'], 
                      c='red', s=100, label='Scenario')
        axs[1].set_xlabel('Price')
        axs[1].set_ylabel('Demand')
        axs[1].legend()
        
        plt.tight_layout()
        return fig




