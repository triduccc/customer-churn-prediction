# This file contains class for cleaning, feature engineering, clustering for customer segmentation.

import datetime as dt
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from scipy.stats import boxcox
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    def __init__(self, data_path):
        # Initialize the DataCleaner with a data path
        self.data_path = data_path
        self.df = None
        self.df_uk = None
        self.rfm_data = None

    def load_data(self):
        # Load and display basic information about the dataset
        dtype = dict(
            InvoiceNo=np.object_,
            StockCode=np.object_,
            Description=np.object_,
            Quantity=np.int64,
            UnitPrice=np.float64,
            CustomerID=np.object_,
            Country=np.object_,
        )

        self.df = pd.read_csv(
            self.data_path,
            encoding="ISO-8859-1",
            parse_dates=["InvoiceDate"],
            dtype=dtype,
        )

        # Transform the ID into 6 digit format
        self.df["CustomerID"] = (
            self.df["CustomerID"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(6)
        )

        print(f"Data's size: {self.df.shape}")
        print(f"Number of record: {len(self.df):,}")

        return self.df
    
    def clean_data(self):
        # Removing invalid records and focus on UK customers

        # Add UnitPrice
        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]

        # Remove the cancelled invoice (starts with C)
        self.df = self.df[~self.df["InvoiceNo"].astype(str).str.startswith("C")]

        # UK customers only
        self.df_uk = self.df[self.df["Country"] == "United Kingdom"].copy()

        # Remove the record with no CustomerID
        self.df_uk = self.df_uk.dropna(subset=["CustomerID"])

        # Remove invalid records
        self.df_uk = self.df_uk[
            (self.df_uk["Quantity"] > 0) & (self.df_uk["UnitPrice"] > 0)
        ]

        return self.df_uk

    def create_time_features(self):
        self.df_uk["DayOfWeek"] = self.df_uk["InvoiceDate"].dt.dayofweek
        self.df_uk["HourOfDay"] = self.df_uk["InvoiceDate"].dt.hour

    def calculate_rfm(self):
        # Calculate Recency, Frequency, Monetary metrics
        snapshot_date = self.df_uk["InvoiceDate"].max() + pd.Timedelta(days=1)

        self.rfm_data = self.df_uk.groupby("CustomerID").agg(
            {
                "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # Recency
                "InvoiceNo": lambda x: len(x.unique()),  # Frequency
                "TotalPrice": lambda x: x.sum(),  # Monetary
            }
        )

        self.rfm_data.columns = ["Recency", "Frequency", "Monetary"]
        return self.rfm_data
    
    def save_cleaned_data(self, output_dir="../data/processed"):
        # Save data to a specific directory
        os.makedirs(output_dir, exist_ok=True)
        self.df_uk.to_csv(f"{output_dir}/cleaned_uk_data.csv", index=False)
        print(f"Saved the cleaned data: {output_dir}/cleaned_uk_data.csv")
        
class FeatureEngineer:
    def __init__(self, data_path):

    def load_data(self):

    def create_customer_features(self):

    def transform_features(self):

    def scale_features(self):

    def plot_features_boxplots(self, transformed=False, save_path=None):

    def plot_features_histograms(self, transformed=False, save_path=None):

    def save_features(self, output_dir="../data/processed"):

class Clustering:
    def __init__(self, scaled_features_path, original_features_path):

    def load_data(self):

    def apply_pca(self, n_components=None):

    def plot_pca_variance(self):

    def find_optimal_clusters(self, k_range=range(2, 11)):

    def plot_optimal_clusters(self):

    def apply_kmeans(self, k_values=[3, 4]):

    def plot_clusters_pca(self, k_values=[3, 4]):

    def plot_clusters_pca_3d(self, k_values=[3, 4]):

    def create_radar_chart(self, k, cluster_names=None):

    def create_individual_radar_plots(self, k, cluster_names=None):

    def save_clusters(self, output_dir="../data/processed"):

class DataVisualizer:
    def __init__(self):

    def plot_revenue_over_time(self, df):

    def plot_time_patterns(self, df):

    def plot_product_analysis(self, df):

    def plot_customer_distribution(self, df):

    def plot_rfm_analysis(self, rfm_data):
