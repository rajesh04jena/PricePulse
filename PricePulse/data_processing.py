################################################################################
# Name: data_processing.py
# Purpose: Data preprocessing
# Date                          Version                Created By
# 5-Dec-2024                   1.0         Rajesh Kumar Jena(Initial Version)
################################################################################

import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, data, hierarchy_levels, lifecycle_column="lifecycle_stage"):
        self.data = data
        self.hierarchy = (
            hierarchy_levels  # ['group', 'category', 'class', 'brand', 'sku']
        )
        self.lifecycle_col = lifecycle_column

    def add_log_transforms(self, columns):
        for col in columns:
            self.data[f"log_{col}"] = np.log(self.data[col] + 1e-6)
        return self.data

    def add_lagged_features(self, column, lags):
        for lag in lags:
            self.data[f"{column}_lag_{lag}"] = self.data.groupby("sku")[column].shift(
                lag
            )
        return self.data

    def encode_hierarchy(self):
        # Create indices for hierarchical levels
        for level in self.hierarchy:
            self.data[f"{level}_idx"], _ = pd.factorize(self.data[level])
        return self.data

    def prepare_lifecycle_dummies(self):
        self.data = pd.get_dummies(
            self.data, columns=[self.lifecycle_col], prefix="", prefix_sep=""
        )
        return self.data
