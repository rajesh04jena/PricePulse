# PricePulse

PricePulse is a Python library for retail price optimization using Bayesian regression for price elasticity estimation and quadratic optimization for optimal pricing. The model incorporates hierarchical Bayesian methods, non-linear elasticity, lifecycle-aware pricing, marketing impact, and inventory constraints to maximize revenue, profit and inventory turn while maintaining stock availability.

## Features

- **Hierarchical Bayesian Model:** Utilizes partial pooling across product hierarchy to share statistical strength.
- **Non-Linear Price Elasticity:** Incorporates quadratic terms in log-price to capture varying elasticity.
- **Lifecycle Stages:** Accounts for New Product Introduction (NPI), Mature, and End-of-Life (EOL) stages with distinct pricing strategies.
- **Marketing Impact:** Uses an exponential decay prior on lagged marketing spend coefficients.
- **Inventory Constraints:** Prevents stockouts by integrating inventory constraints in the optimization process.

## Model Explainability

The library also aims to enhance the price optimizer with explainability features using feature attribution, constraint analysis, and natural language explanations. 

Sample Model Explainability Output :

```python
print(explainer.explain(optimal_price=149.99, style='detailed'))

```

```sh

Key factors in this pricing decision:
1. Price was reduced because demand is highly responsive to price changes (-1.82 elasticity)
2. Product is in mature stage where we prioritize profit maximization
3. Inventory levels healthy (current: 650, max: 1000)
4. Recent marketing spend is contributing 12.4% demand lift

Tradeoff analysis:
At $149.99: 
- Expected profit change: +18.72%
- Inventory turnover: 0.7x
- Price vs competition: -4.25% difference
```


Explanation Features:
- **Causal Factors Considered:**
  - Price elasticity magnitude and direction
  - Lifecycle stage priorities
  - Inventory position analysis
  - Marketing impact quantification
  - Tradeoff visualization between pricing and stock constraints
- **Multiple Explanation Styles:**
  - Simple bullet points for executives
  - Detailed technical breakdown for analysts
  - Visual tradeoff diagrams
- **Contextual Comparisons:**
  - Price position vs competitors
  - Inventory health status
  - Marketing contribution window


## Usage

will be launched soon
