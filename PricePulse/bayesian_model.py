import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import statsmodels.api as sm
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
np.random.seed(42)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from graphviz import Digraph

@dataclass
class ElasticityEstimate:
    """Container for elasticity estimates from linear regression"""
    mean: float
    std_error: float
    pvalue: float
    nobs: int

class BayesianElasticityModel:
    
    def __init__(
    self, 
    data: pd.DataFrame, 
    hierarchy: List[str], 
    lifecycle_stages: List[str],
    date_col: str = "date",
    marketing_lag_cols: Optional[List[str]] = None,
    price_col: str = "price",
    comp_price_col: str = "comp_price",
    demand_col: str = "demand",
    include_quadratic: bool = True,
    prior_strength: float = 1.0,
    detect_seasonality: bool = True,
    seasonality_period: Optional[int] = None,
    detect_trend: bool = True,
    trend_degree: int = 1) :
        """
        Bayesian hierarchical model for price elasticity estimation,
        with linear regression priors for stability and time series components.
        
        Parameters:
        -----------
        data : pandas DataFrame
            Data containing prices, demand, and hierarchy information
        hierarchy : list of str
            Hierarchical levels in order (e.g., ["category", "brand", "sku"])
        lifecycle_stages : list of str
            Product lifecycle stages (e.g., ["npi", "mature", "eol"])
        date_col : str
            Column name for date/time information
        marketing_lag_cols : list of str, optional
            Columns containing marketing lag variables
        price_col : str
            Column name for price
        comp_price_col : str
            Column name for competitor price
        demand_col : str
            Column name for demand
        include_quadratic : bool
            Whether to include quadratic price term
        prior_strength : float
            Strength of linear regression priors
        detect_seasonality : bool
            Whether to automatically detect and model seasonality
        seasonality_period : int, optional
            Known seasonality period (e.g., 7 for weekly, 12 for monthly)
        detect_trend : bool
            Whether to detect and model trend components
        trend_degree : int
            Polynomial degree for trend modeling (1=linear, 2=quadratic)
        """
        self.data = data.copy()
        self.hierarchy = hierarchy
        self.lifecycle_stages = lifecycle_stages
        self.date_col = date_col
        self.price_col = price_col
        self.comp_price_col = comp_price_col
        self.demand_col = demand_col
        self.include_quadratic = include_quadratic
        self.model = None
        self.trace = None
        self.lr_elasticities = None
        self.prior_strength = prior_strength
        
        # Time series parameters
        self.detect_seasonality = detect_seasonality
        self.seasonality_period = seasonality_period
        self.detect_trend = detect_trend
        self.trend_degree = trend_degree
        self.seasonal_components = None
        self.trend_components = None
        
        # Get marketing lag columns
        if marketing_lag_cols is None:
            self.marketing_lag_cols = [col for col in self.data.columns if "marketing_lag" in col]
        else:
            self.marketing_lag_cols = marketing_lag_cols
            
        # Validate data
        self._validate_data()
        
        # Prepare data transformations
        self._prepare_data()
    def _validate_data(self):
        """Validate input data and time series requirements"""
        # Check required columns
        required_cols = [self.price_col, self.comp_price_col, self.demand_col, self.date_col]
        for level in self.hierarchy:
            required_cols.append(f"{level}_idx")
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check lifecycle stage column
        if "lifecycle_stage_idx" not in self.data.columns:
            if "lifecycle_stage" in self.data.columns:
                # Convert categorical to numeric index
                lifecycle_map = {stage: i for i, stage in enumerate(self.lifecycle_stages)}
                self.data["lifecycle_stage_idx"] = self.data["lifecycle_stage"].map(lifecycle_map)
            else:
                raise ValueError("Missing lifecycle_stage or lifecycle_stage_idx column")
        
        # Validate date column
        try:
            self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
        except Exception as e:
            raise ValueError(f"Could not convert {self.date_col} to datetime: {e}")
        
        # Sort data by date
        self.data = self.data.sort_values(by=[self.date_col])
        
        # Check for sufficient data for time series analysis
        if len(self.data) < 30:  # Minimum observations for reliable time series analysis
            print("Warning: Less than 30 observations may lead to unreliable seasonality and trend detection")


    def _prepare_data(self):
        """Prepare data for modeling including time series components"""
        # Log transform price and demand (adding small constant to handle zeros)
        epsilon = 1e-5  # Small constant to add before log transform
        
        # Log transform price
        self.data["log_price"] = np.log(self.data[self.price_col])
        
        # Log transform competitor price (if available and not zero)
        if self.comp_price_col in self.data.columns:
            self.data["log_comp_price"] = np.log(
                np.maximum(self.data[self.comp_price_col], epsilon)
            )
        
        # Log transform demand with special handling for zeros
        self.data["log_demand"] = np.log(
            np.maximum(self.data[self.demand_col], epsilon)
        )
        
        # Create indicators for zero demand (potential censoring)
        self.data["zero_demand"] = (self.data[self.demand_col] < epsilon).astype(int)
        
        # Ensure all index columns are integers
        for level in self.hierarchy:
            idx_col = f"{level}_idx"
            if idx_col in self.data.columns:
                self.data[idx_col] = self.data[idx_col].astype(int)
            
        # Scale marketing variables if they exist
        for col in self.marketing_lag_cols:
            if col in self.data.columns:
                self.data[col] = (self.data[col] - self.data[col].mean()) / self.data[col].std()
        
        # Extract time features
        self._extract_time_features()
        
        # Detect and extract seasonality if enabled
        if self.detect_seasonality:
            self._detect_extract_seasonality()
        
        # Detect and extract trend if enabled
        if self.detect_trend:
            self._detect_extract_trend()
    
    def sample(self, draws=1000, tune=500, chains=2, cores=1):
            """
            Sample from the posterior distribution of the Bayesian model.
    
            Parameters:
            -----------
            draws : int
                Number of posterior samples to draw.
            tune : int
                Number of tuning steps for each chain.
            chains : int
                Number of MCMC chains.
            cores : int
                Number of CPU cores to use for sampling.
    
            Returns:
            --------
            trace : pymc.backends.base.MultiTrace
                The trace object containing posterior samples.
            """
            if self.model is None:
                raise ValueError("The model has not been built. Call 'build_model' first.")
            
            with self.model:
                self.trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=cores,
                    return_inferencedata=True
                )
            
            return self.trace
                
    def _extract_time_features(self):
        """Extract date-based features for time series analysis"""
        # Extract basic time components
        self.data['dayofweek'] = self.data[self.date_col].dt.dayofweek
        self.data['month'] = self.data[self.date_col].dt.month
        self.data['quarter'] = self.data[self.date_col].dt.quarter
        self.data['year'] = self.data[self.date_col].dt.year
        
        # Create sequential time index
        self.data['time_idx'] = range(len(self.data))
    
    def _detect_extract_seasonality(self):
        """Detect and extract seasonality components from the time series"""
        # If seasonality period is provided, use it, otherwise try to detect it
        if self.seasonality_period is None:
            # Use autocorrelation to detect seasonality
            from statsmodels.tsa.stattools import acf
            
            # Use demand to detect seasonality
            demand_series = self.data[self.demand_col].values
            acf_values = acf(demand_series, nlags=min(len(demand_series)//2, 365))
            
            # Find peaks in ACF (excluding lag 0)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(acf_values[1:], height=0.1)
            
            # Add 1 to account for excluding lag 0
            peaks = peaks + 1
            
            if len(peaks) > 0:
                # Use the first significant peak as the seasonality period
                self.seasonality_period = peaks[0]
                print(f"Detected seasonality period: {self.seasonality_period}")
            else:
                # Default to weekly seasonality if nothing detected
                self.seasonality_period = 7
                print("No clear seasonality detected, defaulting to weekly (7 days)")
        
        # Create seasonal dummies or Fourier terms
        if self.seasonality_period <= 12:  # For shorter periods, use dummies
            # Create dummy variables for each season
            for i in range(self.seasonality_period):
                season_idx = self.data['time_idx'] % self.seasonality_period
                self.data[f'season_{i}'] = (season_idx == i).astype(int)
            
            self.seasonal_components = [f'season_{i}' for i in range(self.seasonality_period - 1)]
        else:  # For longer periods, use Fourier terms
            # Create Fourier terms for seasonality
            n_fourier_terms = min(4, self.seasonality_period // 2)  # Use up to 4 pairs
            for i in range(1, n_fourier_terms + 1):
                self.data[f'sin_{i}'] = np.sin(2 * np.pi * i * self.data['time_idx'] / self.seasonality_period)
                self.data[f'cos_{i}'] = np.cos(2 * np.pi * i * self.data['time_idx'] / self.seasonality_period)
            
            self.seasonal_components = []
            for i in range(1, n_fourier_terms + 1):
                self.seasonal_components.extend([f'sin_{i}', f'cos_{i}'])
    
    def _detect_extract_trend(self):
        """Detect and extract trend components from the time series"""
        # Create polynomial trend features
        self.trend_components = []
        for degree in range(1, self.trend_degree + 1):
            trend_col = f'trend_{degree}'
            # Normalize time index to prevent numerical issues
            normalized_time = (self.data['time_idx'] - self.data['time_idx'].mean()) / self.data['time_idx'].std()
            self.data[trend_col] = normalized_time ** degree
            self.trend_components.append(trend_col)
        
        # Check if trend is significant
        if self.trend_components:
            X = sm.add_constant(self.data[self.trend_components])
            y = self.data[self.demand_col]
            model = sm.OLS(y, X).fit()
            
            # Check if any trend component is significant (p < 0.05)
            trend_significant = any(model.pvalues[1:] < 0.05)
            
            if not trend_significant:
                print("Warning: Trend components not statistically significant")
        
    def fit_linear_regression_priors(self):
        """
        Fit linear regression models with seasonality and trend to obtain 
        initial elasticity estimates for use as priors in the Bayesian model.
        
        Returns:
        --------
        dict
            Dictionary of elasticity estimates for each level and item
        """
        print("Fitting linear regression models with time series components to establish priors...")
        
        # Store all elasticity estimates
        all_elasticities = {}
        
        # For each level in the hierarchy
        for level in self.hierarchy:
            level_elasticities = {}
            
            # Get unique items and stages
            unique_items = self.data[f"{level}_idx"].unique()
            
            # For each item at this level
            for item_id in unique_items:
                item_elasticities = {}
                
                # Filter data for this item
                item_data = self.data[self.data[f"{level}_idx"] == item_id]
                
                # For each lifecycle stage
                for stage_idx, stage in enumerate(self.lifecycle_stages):
                    # Filter data for this stage
                    stage_data = item_data[item_data["lifecycle_stage_idx"] == stage_idx]
                    
                    # Skip if not enough data
                    if len(stage_data) < 5:  # Minimum observations for regression
                        item_elasticities[stage] = ElasticityEstimate(
                            mean=-1.0,  # Default elasticity
                            std_error=1.0,  # Large uncertainty
                            pvalue=1.0,
                            nobs=len(stage_data)
                        )
                        continue
                    
                    # Prepare features - always include price
                    features = ["log_price"]
                    
                    # Add competitor price if available
                    if "log_comp_price" in stage_data.columns:
                        features.append("log_comp_price")
                    
                    # Add marketing variables
                    features.extend([col for col in self.marketing_lag_cols if col in stage_data.columns])
                    
                    # Add seasonality components if available
                    if self.seasonal_components:
                        features.extend([col for col in self.seasonal_components if col in stage_data.columns])
                    
                    # Add trend components if available
                    if self.trend_components:
                        features.extend([col for col in self.trend_components if col in stage_data.columns])
                    
                    # Create X (add constant for intercept)
                    X = sm.add_constant(stage_data[features])
                    y = stage_data["log_demand"]
                    
                    # Fit linear regression
                    try:
                        model = sm.OLS(y, X).fit()
                        
                        # Extract price elasticity and standard error
                        price_elasticity = model.params["log_price"]
                        price_elasticity = min(price_elasticity, -0.01)
                        std_error = model.bse["log_price"]
                        std_error = max(std_error, 0.5)
                        pvalue = model.pvalues["log_price"]
                        
                        # Store results
                        item_elasticities[stage] = ElasticityEstimate(
                            mean=price_elasticity,
                            std_error=std_error,
                            pvalue=pvalue,
                            nobs=len(stage_data)
                        )
                        
                    except Exception as e:
                        print(f"Warning: Regression failed for {level} {item_id}, stage {stage}: {e}")
                        # Use default values
                        item_elasticities[stage] = ElasticityEstimate(
                            mean=-1.0,  # Default elasticity
                            std_error=1.0,  # Large uncertainty
                            pvalue=1.0,
                            nobs=len(stage_data)
                        )
                
                # Store elasticities for this item
                level_elasticities[item_id] = item_elasticities
            
            # Store elasticities for this level
            all_elasticities[level] = level_elasticities
        
        self.lr_elasticities = all_elasticities
        return all_elasticities
        
    def build_model(self):
        """Build the hierarchical Bayesian model with seasonality and trend components"""
        # First, fit linear regression to get priors if not already done
        if self.lr_elasticities is None:
            self.fit_linear_regression_priors()
        
        with pm.Model() as self.model:
            # Create shared variables for indices
            hierarchy_indices = {}
            for level in self.hierarchy:
                hierarchy_indices[level] = pm.ConstantData(
                    f"{level}_idx", self.data[f"{level}_idx"].values
                )
            
            # Lifecycle stage indices
            lifecycle_idx = pm.ConstantData(
                "lifecycle_idx", self.data["lifecycle_stage_idx"].values
            )
            
            # Number of unique values at each level
            n_units = {
                level: len(np.unique(self.data[f"{level}_idx"]))
                for level in self.hierarchy
            }
            
            # Number of lifecycle stages
            n_stages = len(self.lifecycle_stages)
            
            # ----- Hierarchical Price Elasticity Model with LR Priors -----
            
            # Global hyperpriors for price elasticity
            #mu_beta_price = pm.Normal("mu_beta_price", mu=-1.0, sigma=0.5)
            #sigma_beta_price = pm.HalfNormal("sigma_beta_price", sigma=0.5)

            mu_beta_price = pm.TruncatedNormal("mu_beta_price", 
                                 mu=-2.0,  # More negative prior
                                 sigma=0.3, 
                                 upper=-0.1,  # Force meaningful negativity
                                 lower=-5.0)
            
            sigma_beta_price = pm.HalfNormal("sigma_beta_price", sigma=0.2)

            # Initialize beta_price dictionaries for each level
            beta_price = {}
            
            # Create hierarchy of price elasticities with LR priors
            # Start with highest level - using global hyperpriors
            highest_level = self.hierarchy[0]
            
            # Create prior means and standard errors for this level
            prior_means = np.zeros((n_units[highest_level], n_stages))
            prior_stds = np.ones((n_units[highest_level], n_stages))
            
            # Fill in from linear regression where available
            level_elasticities = self.lr_elasticities[highest_level]
            for item_id, item_elasticities in level_elasticities.items():
                for stage_idx, stage in enumerate(self.lifecycle_stages):
                    if stage in item_elasticities:
                        prior_means[item_id, stage_idx] = item_elasticities[stage].mean
                        # Scale std error by prior strength parameter
                        prior_stds[item_id, stage_idx] = max(
                            item_elasticities[stage].std_error * self.prior_strength,
                            0.1  # Minimum standard deviation to allow learning
                        )
            
            # Create parameters with informed priors
            # Now using the mu_beta_price and sigma_beta_price as hyperpriors
            # beta_price[highest_level] = pm.Normal(
            #     f"beta_price_{highest_level}",
            #     mu=pm.math.switch(
            #         pt.lt(prior_stds, 0.5),  # If we have reliable priors
            #         prior_means,  # Use the LR priors
            #         mu_beta_price  # Otherwise use the global hyperprior
            #     ),
            #     sigma=pm.math.switch(
            #         pt.lt(prior_stds, 0.5),  # If we have reliable priors
            #         prior_stds,  # Use the LR standard errors
            #         sigma_beta_price  # Otherwise use the global variance
            #     ),
            #     shape=(n_units[highest_level], n_stages)
            # )
            
            # Highest level (categories)
            beta_price[highest_level] = pm.TruncatedNormal(
                f"beta_price_{highest_level}",
                mu=pm.math.switch(
                    pt.lt(prior_stds, 0.5),
                    prior_means,
                    mu_beta_price
                ),
                sigma=pm.math.switch(
                    pt.lt(prior_stds, 0.5),
                    prior_stds,
                    sigma_beta_price
                ),               
                lower=-5.0,  # Prevent extreme negative values
                upper=-0.01, 
                shape=(n_units[highest_level], n_stages))
            
            # Continue with lower levels, using higher level as prior
            for i in range(1, len(self.hierarchy)):
                parent_level = self.hierarchy[i-1]
                current_level = self.hierarchy[i]
                
                # Get parent indices for current level items
                parent_mapping = {}
                for _, row in self.data[[f"{parent_level}_idx", f"{current_level}_idx"]].drop_duplicates().iterrows():
                    parent_mapping[row[f"{current_level}_idx"]] = row[f"{parent_level}_idx"]
                
                # Create mapping array from child to parent
                parent_idx_array = np.zeros(n_units[current_level], dtype=int)
                for child_id, parent_id in parent_mapping.items():
                    parent_idx_array[child_id] = parent_id
                
                # Make shared data for the parent indices
                parent_idx = pm.ConstantData(
                    f"{current_level}_parent_idx", 
                    parent_idx_array
                )
                
                # Get linear regression priors for this level
                level_elasticities = self.lr_elasticities[current_level]
                
                prior_means = np.zeros((n_units[current_level], n_stages))
                prior_stds = np.ones((n_units[current_level], n_stages))
                
                # Fill in from linear regression where available
                for item_id, item_elasticities in level_elasticities.items():
                    if item_id >= n_units[current_level]:
                        continue  # Skip if ID is out of range
                        
                    for stage_idx, stage in enumerate(self.lifecycle_stages):
                        if stage in item_elasticities:
                            prior_means[item_id, stage_idx] = item_elasticities[stage].mean
                            prior_stds[item_id, stage_idx] = max(
                                item_elasticities[stage].std_error * self.prior_strength,
                                0.1  # Minimum standard deviation
                            )
                
                # Now use parent indices to correctly link hierarchical levels
                # Create item parameters that pull from both parent values and LR priors
                # beta_price[current_level] = pm.Normal(
                #     f"beta_price_{current_level}",
                #     # Use a weighted combination of parent value and LR prior
                #     mu=pm.math.switch(
                #         pt.lt(prior_stds, 0.5),  # If we have reliable priors
                #         # Weighted combination of parent and LR
                #         0.5 * prior_means + 0.5 * beta_price[parent_level][parent_idx_array],
                #         # Otherwise rely more on parent
                #         beta_price[parent_level][parent_idx_array]
                #     ),
                #     # Use a smaller sigma when we have parent information
                #     sigma=pm.math.switch(
                #         pt.lt(prior_stds, 0.5),  # If we have reliable priors
                #         prior_stds,  # Use LR standard errors
                #         sigma_beta_price / 2  # Otherwise use tighter variance around parent
                #     ),
                #     shape=(n_units[current_level], n_stages)
                # )
            
                beta_price[current_level] = pm.TruncatedNormal(
                f"beta_price_{current_level}",
                mu=pm.math.switch(
                    pt.lt(prior_stds, 0.5),
                    0.5 * prior_means + 0.5 * beta_price[parent_level][parent_idx_array],
                    beta_price[parent_level][parent_idx_array]
                ),
                sigma=pm.math.switch(
                    pt.lt(prior_stds, 0.5),
                    prior_stds,
                    sigma_beta_price / 2
                ),
                lower=-5.0,  # Prevent extreme negative values
                upper=-0.01,             
                shape=(n_units[current_level], n_stages))
                
            # ----- Seasonality and Trend Components -----
            
            # Seasonality coefficients
            beta_seasonal = {}
            if self.seasonal_components:
                # Create coefficients for each seasonal component
                for s_comp in self.seasonal_components:
                    beta_seasonal[s_comp] = pm.Normal(
                        f"beta_{s_comp}",
                        mu=0,
                        sigma=0.5,
                        shape=(n_stages,)  # Allow different effects by lifecycle stage
                    )
            
            # Trend coefficients
            beta_trend = {}
            if self.trend_components:
                # Create coefficients for each trend component
                for t_comp in self.trend_components:
                    beta_trend[t_comp] = pm.Normal(
                        f"beta_{t_comp}",
                        mu=0,
                        sigma=0.5,
                        shape=(n_stages,)  # Allow different effects by lifecycle stage
                    )
            
            # ----- Additional Model Components -----
            
            # Intercept terms (varying by SKU and lifecycle)
            lowest_level = self.hierarchy[-1]
            intercept = pm.Normal(
                "intercept", 
                mu=0, 
                sigma=1, 
                shape=(n_units[lowest_level], n_stages)
            )
            
            # Quadratic price term (if enabled)
            if self.include_quadratic:
                beta_price_quad = pm.Normal(
                    "beta_price_quad", 
                    mu=0, 
                    sigma=0.2,
                    shape=n_stages
                )
            
            # Competitor price elasticity
            if "log_comp_price" in self.data.columns:
                beta_comp_price = pm.Normal(
                    "beta_comp_price",
                    mu=0.5,  # Prior belief: competitor price increases our demand
                    sigma=0.3,
                    lower=0,
                    shape=n_stages
                )
            
            # Marketing effectiveness
            beta_marketing = {}
            for mkt_col in self.marketing_lag_cols:
                if mkt_col in self.data.columns:
                    beta_marketing[mkt_col] = pm.Normal(
                        f"beta_{mkt_col}",
                        mu=0.2,  # Prior belief: marketing increases demand
                        sigma=0.2,
                        shape=n_stages
                    )
            
            # Model error term (observation noise)
            sigma = pm.HalfNormal("sigma", sigma=0.5)
            
            # ----- Model Equation -----
            
            # Get the appropriate indices for the model
            sku_idx = hierarchy_indices[lowest_level]
            
            # Base linear predictor for log demand
            mu = (
                intercept[sku_idx, lifecycle_idx] +
                beta_price[lowest_level][sku_idx, lifecycle_idx] * self.data["log_price"]
            )
            
            # Add quadratic price term if enabled
            if self.include_quadratic:
                mu += beta_price_quad[lifecycle_idx] * (self.data["log_price"] ** 2)
            
            # Add competitor price effect if available
            if "log_comp_price" in self.data.columns:
                mu += beta_comp_price[lifecycle_idx] * self.data["log_comp_price"]
            
            # Add marketing effects if available
            for mkt_col in beta_marketing:
                mu += beta_marketing[mkt_col][lifecycle_idx] * self.data[mkt_col]
            
            # Add seasonality components if available
            if self.seasonal_components:
                for s_comp in self.seasonal_components:
                    mu += beta_seasonal[s_comp][lifecycle_idx] * self.data[s_comp]
            
            # Add trend components if available
            if self.trend_components:
                for t_comp in self.trend_components:
                    mu += beta_trend[t_comp][lifecycle_idx] * self.data[t_comp]
            
            # Likelihood function for observed demand
            pm.Normal("likelihood", mu=mu, sigma=sigma, observed=self.data["log_demand"])

    def fit(self, draws=1000, tune=1000, chains=4, target_accept=0.9, return_inferencedata=True):
        """Fit the Bayesian hierarchical model"""
        if self.model is None:
            self.build_model()
        
        with self.model:
            print("Sampling from posterior distribution...")
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=return_inferencedata
            )
        
        print("Sampling complete!")
        return self.trace
    
    def get_elasticity_estimates(self, hdi_prob=0.95):
        """
        Extract elasticity estimates from the posterior distribution
        
        Parameters:
        -----------
        hdi_prob : float
            Probability mass for highest density interval
        
        Returns:
        --------
        dict
            Dictionary of elasticity posterior summaries for each level, item, and lifecycle stage
        """
        if self.trace is None:
            raise ValueError("Model has not been fit yet. Call fit() first.")
        
        results = {}
        
        # For each level in the hierarchy
        for level in self.hierarchy:
            level_results = {}
            
            # Extract posterior samples for this level
            var_name = f"beta_price_{level}"
            
            # Get summary statistics
            summary = az.summary(
                self.trace, 
                var_names=[var_name],
                hdi_prob=hdi_prob
            )
            
            # Number of unique values at this level
            n_units = len(np.unique(self.data[f"{level}_idx"]))
            
            # For each item at this level
            for item_id in range(n_units):
                item_results = {}
                
                # For each lifecycle stage
                for stage_idx, stage in enumerate(self.lifecycle_stages):
                    var_idx = f"{var_name}[{item_id},{stage_idx}]"
                    
                    if var_idx in summary.index:
                        item_results[stage] = {
                            "mean": summary.loc[var_idx, "mean"],
                            "median": summary.loc[var_idx, "median"],
                            "hdi_lower": summary.loc[var_idx, f"hdi_{int(hdi_prob*100)}%_lower"],
                            "hdi_upper": summary.loc[var_idx, f"hdi_{int(hdi_prob*100)}%_upper"],
                            "std": summary.loc[var_idx, "sd"],
                            "effective_sample": summary.loc[var_idx, "ess_bulk"]
                        }
                
                # Store results for this item
                level_results[item_id] = item_results
            
            # Store results for this level
            results[level] = level_results
        
        return results
    
    def analyze_seasonality_trend(self):
        """Analyze the posterior distributions of seasonality and trend components"""
        if self.trace is None:
            raise ValueError("Model has not been fit yet. Call fit() first.")
        
        results = {}
        
        # Analyze seasonality components
        if self.seasonal_components:
            seasonal_results = {}
            
            for s_comp in self.seasonal_components:
                var_name = f"beta_{s_comp}"
                
                # Get summary statistics
                summary = az.summary(self.trace, var_names=[var_name])
                
                # Store results for each lifecycle stage
                stage_results = {}
                for stage_idx, stage in enumerate(self.lifecycle_stages):
                    var_idx = f"{var_name}[{stage_idx}]"
                    
                    if var_idx in summary.index:
                        stage_results[stage] = {
                            "mean": summary.loc[var_idx, "mean"],
                            "std": summary.loc[var_idx, "sd"],
                            "hdi_lower": summary.loc[var_idx, "hdi_3%"],
                            "hdi_upper": summary.loc[var_idx, "hdi_97%"]
                        }
                
                seasonal_results[s_comp] = stage_results
            
            results["seasonality"] = seasonal_results
        
        # Analyze trend components
        if self.trend_components:
            trend_results = {}
            
            for t_comp in self.trend_components:
                var_name = f"beta_{t_comp}"
                
                # Get summary statistics
                summary = az.summary(self.trace, var_names=[var_name])
                
                # Store results for each lifecycle stage
                stage_results = {}
                for stage_idx, stage in enumerate(self.lifecycle_stages):
                    var_idx = f"{var_name}[{stage_idx}]"
                    
                    if var_idx in summary.index:
                        stage_results[stage] = {
                            "mean": summary.loc[var_idx, "mean"],
                            "std": summary.loc[var_idx, "sd"],
                            "hdi_lower": summary.loc[var_idx, "hdi_3%"],
                            "hdi_upper": summary.loc[var_idx, "hdi_97%"]
                        }
                
                trend_results[t_comp] = stage_results
            
            results["trend"] = trend_results
        
        return results
    
    def predict(self, new_data=None, return_summary=True):
        """
        Generate predictions using the fitted model
        
        Parameters:
        -----------
        new_data : pandas DataFrame, optional
            New data for prediction. If None, use the training data
        return_summary : bool
            Whether to return summary statistics or posterior samples
        
        Returns:
        --------
        pandas DataFrame or numpy array
            Predicted demand values (log scale)
        """
        if self.trace is None:
            raise ValueError("Model has not been fit yet. Call fit() first.")
        
        # Use training data if new data not provided
        if new_data is None:
            data = self.data.copy()
        else:
            data = new_data.copy()
            
            # Prepare new data (log transforms, seasonality, trend)
            # Log transform price
            data["log_price"] = np.log(data[self.price_col])
            
            # Log transform competitor price if available
            if self.comp_price_col in data.columns:
                data["log_comp_price"] = np.log(
                    np.maximum(data[self.comp_price_col], 1e-5)
                )
            
            # Extract time features if date column exists
            if self.date_col in data.columns:
                data[self.date_col] = pd.to_datetime(data[self.date_col])
                data['dayofweek'] = data[self.date_col].dt.dayofweek
                data['month'] = data[self.date_col].dt.month
                data['quarter'] = data[self.date_col].dt.quarter
                data['year'] = data[self.date_col].dt.year
                
                # Create sequential time index continuing from training data
                max_time_idx = self.data['time_idx'].max()
                data['time_idx'] = range(max_time_idx + 1, max_time_idx + 1 + len(data))
                
                # Create seasonality features
                if self.seasonal_components:
                    if max([int(s.split('_')[1]) for s in self.seasonal_components if s.startswith('season_')], default=0) > 0:
                        # Seasonal dummies
                        for i in range(self.seasonality_period):
                            season_idx = data['time_idx'] % self.seasonality_period
                            data[f'season_{i}'] = (season_idx == i).astype(int)
                    else:
                        # Fourier terms
                        n_fourier_terms = max([int(s.split('_')[1]) for s in self.seasonal_components if s.startswith('sin_')], default=0)
                        for i in range(1, n_fourier_terms + 1):
                            data[f'sin_{i}'] = np.sin(2 * np.pi * i * data['time_idx'] / self.seasonality_period)
                            data[f'cos_{i}'] = np.cos(2 * np.pi * i * data['time_idx'] / self.seasonality_period)
                
                # Create trend features
                if self.trend_components:
                    for degree in range(1, self.trend_degree + 1):
                        # Normalize using the same parameters as training data for consistency
                        mean_time = self.data['time_idx'].mean()
                        std_time = self.data['time_idx'].std()
                        normalized_time = (data['time_idx'] - mean_time) / std_time
                        data[f'trend_{degree}'] = normalized_time ** degree
        
        # Generate predictions using the posterior samples
        with self.model:
            pm.set_data({"log_price": data["log_price"]})
            if "log_comp_price" in data.columns:
                pm.set_data({"log_comp_price": data["log_comp_price"]})
            
            # Set data for seasonal components
            if self.seasonal_components:
                for s_comp in self.seasonal_components:
                    if s_comp in data.columns:
                        pm.set_data({s_comp: data[s_comp]})
            
            # Set data for trend components
            if self.trend_components:
                for t_comp in self.trend_components:
                    if t_comp in data.columns:
                        pm.set_data({t_comp: data[t_comp]})
            
            # Set data for marketing variables
            for mkt_col in self.marketing_lag_cols:
                if mkt_col in data.columns:
                    pm.set_data({mkt_col: data[mkt_col]})
            
            # Need to set indices for hierarchical model
            for level in self.hierarchy:
                pm.set_data({f"{level}_idx": data[f"{level}_idx"].values})
            
            # Set lifecycle stage indices
            pm.set_data({"lifecycle_idx": data["lifecycle_stage_idx"].values})
            
            # Generate posterior predictive samples
            posterior_pred = pm.sample_posterior_predictive(self.trace)
        
        # Extract predictions
        if return_summary:
            # Calculate summary statistics
            pred_mean = posterior_pred.posterior_predictive.likelihood.mean(dim=["chain", "draw"]).values
            pred_std = posterior_pred.posterior_predictive.likelihood.std(dim=["chain", "draw"]).values
            pred_lower = np.percentile(posterior_pred.posterior_predictive.likelihood.values, 2.5, axis=(0, 1))
            pred_upper = np.percentile(posterior_pred.posterior_predictive.likelihood.values, 97.5, axis=(0, 1))
            
            # Transform back to original scale (exp of log values)
            predictions = pd.DataFrame({
                "pred_log_demand_mean": pred_mean,
                "pred_log_demand_std": pred_std,
                "pred_log_demand_lower": pred_lower,
                "pred_log_demand_upper": pred_upper,
                "pred_demand_mean": np.exp(pred_mean),
                "pred_demand_lower": np.exp(pred_lower),
                "pred_demand_upper": np.exp(pred_upper)
            })
            
            return predictions
        else:
            # Return all posterior samples
            return posterior_pred.posterior_predictive.likelihood.values
    
    def compare_elasticities_over_time(self, time_column='date', n_periods=3, period_type='auto'):
        """
        Compare elasticity estimates across different time periods.
        
        Parameters:
        -----------
        time_column : str
            Column name containing the date/time information
        n_periods : int
            Number of time periods to split the data into
        period_type : str
            Type of periods to use: 'auto', 'month', 'quarter', 'year'
            
        Returns:
        --------
        dict
            Dictionary of elasticity comparisons across time periods
        """
        if time_column not in self.data.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")
            
        # Ensure the time column is datetime type
        self.data[time_column] = pd.to_datetime(self.data[time_column])
        
        # Determine period boundaries
        if period_type == 'auto':
            # Automatically determine appropriate period type based on data range
            date_range = self.data[time_column].max() - self.data[time_column].min()
            days = date_range.days
            
            if days <= 90:  # Less than 3 months, use weeks
                period_type = 'week'
            elif days <= 730:  # Less than 2 years, use months
                period_type = 'month'
            elif days <= 1825:  # Less than 5 years, use quarters
                period_type = 'quarter'
            else:  # More than 5 years, use years
                period_type = 'year'
        
        # Create period labels based on period_type
        if period_type == 'week':
            self.data['period'] = self.data[time_column].dt.isocalendar().week
            period_label = 'Week'
        elif period_type == 'month':
            self.data['period'] = self.data[time_column].dt.to_period('M')
            period_label = 'Month'
        elif period_type == 'quarter':
            self.data['period'] = self.data[time_column].dt.to_period('Q')
            period_label = 'Quarter'
        elif period_type == 'year':
            self.data['period'] = self.data[time_column].dt.year
            period_label = 'Year'
        else:
            raise ValueError(f"Invalid period_type: {period_type}")
        
        # Get unique periods and sort them
        unique_periods = sorted(self.data['period'].unique())
        
        # If there are too many periods, group them
        if len(unique_periods) > n_periods:
            # Create roughly equal-sized period groups
            period_groups = np.array_split(unique_periods, n_periods)
            period_map = {}
            for i, group in enumerate(period_groups):
                for period in group:
                    period_map[period] = i
            
            # Map periods to groups
            self.data['period_group'] = self.data['period'].map(period_map)
            
            # Create descriptive labels for each group
            period_labels = []
            for group in period_groups:
                if len(group) == 1:
                    period_labels.append(str(group[0]))
                else:
                    period_labels.append(f"{group[0]} - {group[-1]}")
        else:
            # Use individual periods
            self.data['period_group'] = self.data['period']
            period_labels = [str(p) for p in unique_periods]
        
        # Store original data and model state
        original_data = self.data.copy()
        original_model = self.model
        original_trace = self.trace
        original_lr_elasticities = self.lr_elasticities
        
        # Compare elasticities across periods
        comparisons = self._compare_period_elasticities(period_labels)
        
        # Restore original data and model
        self.data = original_data
        self.model = original_model
        self.trace = original_trace
        self.lr_elasticities = original_lr_elasticities
        
        return comparisons
    
    def _compare_period_elasticities(self, period_labels):
        """
        Helper method to compare elasticities across different time periods.
        
        Parameters:
        -----------
        period_labels : list
            Labels for each time period
            
        Returns:
        --------
        dict
            Comparison of elasticities across periods
        """
        unique_period_groups = sorted(self.data['period_group'].unique())
        period_elasticities = {}
        
        # For each period, fit model and extract elasticities
        for period_idx, period_group in enumerate(unique_period_groups):
            print(f"Analyzing period {period_labels[period_idx]}...")
            
            # Filter data for this period
            period_data = self.data[self.data['period_group'] == period_group].copy()
            
            # Create a new model instance for this period
            period_model = BayesianElasticityModel(
                data=period_data,
                hierarchy=self.hierarchy,
                lifecycle_stages=self.lifecycle_stages,
                marketing_lag_cols=self.marketing_lag_cols,
                price_col=self.price_col,
                comp_price_col=self.comp_price_col,
                demand_col=self.demand_col,
                include_quadratic=self.include_quadratic,
                prior_strength=self.prior_strength
            )
            
            # Fit model for this period
            try:
                # Use fewer samples for period models to speed up computation
                period_model.fit(draws=500, tune=500, chains=2)
                
                # Extract elasticity estimates
                elasticities = period_model.get_elasticity_estimates()
                period_elasticities[period_labels[period_idx]] = elasticities
            except Exception as e:
                print(f"Warning: Failed to fit model for period {period_labels[period_idx]}: {e}")
                # Use linear regression estimates as fallback
                period_model.fit_linear_regression_priors()
                period_elasticities[period_labels[period_idx]] = {
                    level: {
                        item_id: {
                            stage: {
                                "mean": item_stages[stage].mean if stage in item_stages else None,
                                "std": item_stages[stage].std_error if stage in item_stages else None,
                            }
                            for stage in self.lifecycle_stages
                        }
                        for item_id, item_stages in level_elasticities.items()
                    }
                    for level, level_elasticities in period_model.lr_elasticities.items()
                }
        
        # Create comparison of elasticities across periods
        comparisons = {}
        for level in self.hierarchy:
            level_comparisons = {}
            
            # Get unique items at this level
            unique_items = self.data[f"{level}"].unique()
            
            for item_id in unique_items:
                item_comparisons = {}
                
                # For each lifecycle stage
                for stage in self.lifecycle_stages:
                    stage_comparisons = {
                        period: period_elasticities[period][level][item_id][stage]["mean"]
                        if (level in period_elasticities[period] and 
                            item_id in period_elasticities[period][level] and 
                            stage in period_elasticities[period][level][item_id])
                        else None
                        for period in period_labels
                    }
                    
                    # Skip if all values are None
                    if all(v is None for v in stage_comparisons.values()):
                        continue
                    
                    # Calculate stability metrics
                    non_null_values = [v for v in stage_comparisons.values() if v is not None]
                    if len(non_null_values) >= 2:
                        stability_metrics = {
                            "range": max(non_null_values) - min(non_null_values),
                            "std_dev": np.std(non_null_values),
                            "coef_variation": np.std(non_null_values) / abs(np.mean(non_null_values)) 
                                if np.mean(non_null_values) != 0 else np.nan
                        }
                        stage_comparisons["stability"] = stability_metrics
                    
                    item_comparisons[stage] = stage_comparisons
                
                # Skip if no stages have comparisons
                if not item_comparisons:
                    continue
                    
                level_comparisons[item_id] = item_comparisons
            
            comparisons[level] = level_comparisons
        
        return comparisons
    
