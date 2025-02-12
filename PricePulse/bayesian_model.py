################################################################################
# Name: bayesian_model.py
# Purpose: Generate price elastitcities and cross price  elastitcities
# Date                          Version                Created By
# 5-Dec-2024                   1.0         Rajesh Kumar Jena(Initial Version)
################################################################################

import pymc as pm
import aesara.tensor as at
import numpy as np

class BayesianElasticityModel:
    def __init__(self, data, hierarchy, lifecycle_stages):
        self.data = data
        self.hierarchy = hierarchy
        self.lifecycle_stages = lifecycle_stages  # ['npi', 'mature', 'eol']
        self.model = None

    def build_model(self):
        with pm.Model() as self.model:
            # Hierarchical indices
            hierarchy_indices = {
                level: pm.Data(level, self.data[f"{level}_idx"].values)
                for level in self.hierarchy
            }

            # Priors for hierarchical levels
            mu_beta_p1 = pm.Normal("mu_beta_p1", mu=-1, sigma=1)
            sigma_beta_p1 = pm.HalfNormal("sigma_beta_p1", 1)

            # Varying intercepts and slopes across hierarchy
            beta_p1 = {}
            for level in self.hierarchy:
                beta_p1[level] = pm.Normal(
                    f"beta_p1_{level}",
                    mu=mu_beta_p1,
                    sigma=sigma_beta_p1,
                    shape=(
                        len(np.unique(self.data[f"{level}_idx"])),
                        len(self.lifecycle_stages),
                    ),
                )

            # Quadratic term for price elasticity
            beta_p2 = pm.Normal(
                "beta_p2", mu=-0.1, sigma=0.1, shape=len(self.lifecycle_stages)
            )

            # Competition elasticity
            beta_comp = pm.Normal("beta_comp", mu=0.5, sigma=0.2)

            # Marketing lag coefficients (exponential decay prior)
            marketing_lags = [col for col in self.data if "marketing_lag" in col]
            alpha = pm.Exponential("alpha", 1)
            beta_marketing = pm.Deterministic(
                "beta_marketing", alpha * at.exp(-0.5 * np.arange(len(marketing_lags)))
            )

            # Likelihood
            log_demand = (
                beta_p1["sku"][hierarchy_indices["sku"]]  # SKU-level intercepts
                + beta_p2[self.data["lifecycle_stage"]] * (self.data["log_price"] ** 2)
                + beta_comp * self.data["log_comp_price"]
                + at.dot(self.data[marketing_lags], beta_marketing)
                # ... Add other terms (attributes, inventory)
            )
            pm.Normal(
                "log_demand_obs",
                mu=log_demand,
                sigma=pm.Exponential("sigma", 1),
                observed=self.data["log_demand"],
            )

    def fit(self, samples=1000, tune=1000):
        with self.model:
            self.trace = pm.sample(samples, tune=tune, target_accept=0.9)
        return self.trace
