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

        # Add TotalPrice
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
    # Class for creating new features from the transaction data

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.customer_features = None
        self.customer_features_transformed = None
        self.customer_features_scaled = None

        # Define the features
        self.feature_customer = [
            "Sum_Quantity",
            "Mean_UnitPrice",
            "Mean_TotalPrice",
            "Sum_TotalPrice",
            "Count_Invoice",
            "Count_Stock",
            "Mean_InvoiceCountPerStock",
            "Mean_StockCountPerInvoice",
            "Mean_UnitPriceMeanPerInvoice",
            "Mean_QuantitySumPerInvoice",
            "Mean_TotalPriceMeanPerInvoice",
            "Mean_TotalPriceSumPerInvoice",
            "Mean_UnitPriceMeanPerStock",
            "Mean_QuantitySumPerStock",
            "Mean_TotalPriceMeanPerStock",
            "Mean_TotalPriceSumPerStock",
        ]

        self.feature_customer2 = ["CustomerID"] + self.feature_customer

    def load_data(self):
        # Load cleaned data for feature engineering
        self.df = pd.read_csv(self.data_path)
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])

        print(f"Size of data: {self.df.shape}")
        return self.df

    def create_customer_features(self):
        # customer level aggregated features
        num_customers = self.df["CustomerID"].nunique() 
        self.customer_features = pd.DataFrame(
            data=np.zeros((num_customers, len(self.feature_customer2)), dtype=float),
            columns=self.feature_customer2,
        )

        self.customer_features["CustomerID"] = self.customer_features["CustomerID"].astype("object")
        print("Calculating features for customer...")


        for i, (customer_id, value) in enumerate(self.df.groupby("CustomerID")):
            self.customer_features.iat[i, 0] = customer_id

            # 1. Quantity sum
            self.customer_features.iat[i, 1] = value.Quantity.sum()

            # 2. UnitPrice mean
            self.customer_features.iat[i, 2] = value.UnitPrice.mean()

            # 3. TotalPrice mean
            self.customer_features.iat[i, 3] = value.TotalPrice.mean()

            # 4. TotalPrice sum
            self.customer_features.iat[i, 4] = value.TotalPrice.sum()

            # 5. Invoice count
            self.customer_features.iat[i, 5] = value.InvoiceNo.nunique()

            # 6. Stock count
            self.customer_features.iat[i, 6] = value.StockCode.nunique()

            # 7-16. Other metrics
            self.customer_features.iat[i, 7] = value.groupby("StockCode").size().mean()
            self.customer_features.iat[i, 8] = value.groupby("InvoiceNo").size().mean()
            self.customer_features.iat[i, 9] = (
                value.groupby("InvoiceNo")["UnitPrice"].mean().mean()
            )
            self.customer_features.iat[i, 10] = (
                value.groupby("InvoiceNo")["Quantity"].sum().mean()
            )
            self.customer_features.iat[i, 11] = (
                value.groupby("InvoiceNo")["TotalPrice"].mean().mean()
            )
            self.customer_features.iat[i, 12] = (
                value.groupby("InvoiceNo")["TotalPrice"].sum().mean()
            )
            self.customer_features.iat[i, 13] = (
                value.groupby("StockCode")["UnitPrice"].mean().mean()
            )
            self.customer_features.iat[i, 14] = (
                value.groupby("StockCode")["Quantity"].sum().mean()
            )
            self.customer_features.iat[i, 15] = (
                value.groupby("StockCode")["TotalPrice"].mean().mean()
            )
            self.customer_features.iat[i, 16] = (
                value.groupby("StockCode")["TotalPrice"].sum().mean()
            )

            if (i + 1) % 500 == 0:
                print(f"Handled {i + 1}/{num_customers} customers...")

            print("Finished calculating features")
            return self.customer_features

    def transform_features(self):
        # Apply Box-Cox transformation to normalize feature distribution

        # Customer ID as index
        customer_features_indexed = self.customer_features.set_index("CustomerID") # ts now is a new df

        # Box-Cox
        feature_values = customer_features_indexed.values + 1  # Cộng 1 cho Box-Cox
        self.customer_features_transformed = customer_features_indexed.copy()
        
        print("Applying Box-Cox transformation.")
        for i, feature in enumerate(self.feature_customer):
            transformed, lambda_param = boxcox(feature_values[:, i])
            self.customer_features_transformed.iloc[:, i] = transformed

        print("Box-Cox transformation has been done.")
        return self.customer_features_transformed

    def scale_features(self):
        # Apply standardization to features.
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.customer_features_transformed)

        self.customer_features_scaled = pd.DataFrame(
            features_scaled,
            columns=self.feature_customer,
            index=self.customer_features_transformed.index,
        )

        print("Feature scaling has been done.")
        return self.customer_features_scaled

    def plot_features_boxplots(self, transformed=False, save_path=None):
        if transformed and self.customer_features_transformed is not None:
            data = self.customer_features_transformed
        else:
            if self.customer_features is not None:
                data = self.customer_features.set_index("CustomerID")
                title = "Box Plots before Box-Cox Transformation"
            else:
                print("No features yet, please run create_customer_features().")
                return

        with sns.plotting_context(context="notebook"):
            plt.figure(figsize=(15, 15))

            for i, feature in enumerate(self.feature_customer):
                plt.subplot(4, 4, i + 1)
                plt.boxplot(data.iloc[:, i] if transformed else data[feature])
                plt.title(feature, fontsize=10)
                plt.xticks([])

            plt.tight_layout()
            # plt.suptitle(title, fontsize=16, y=1.1)

            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                print(f"Saved plot: {save_path}")
            plt.show()

    def plot_features_histograms(self, transformed=False, save_path=None):
        if transformed and self.customer_features_transformed is not None:
            data = self.customer_features_transformed
            title = "Histograms after Box-Cox Transformation"
        else:
            if self.customer_features is not None:
                data = self.customer_features.set_index("CustomerID")
                title = "Histograms before Box-Cox Transformation"
            else:
                print("No features yet, please run create_customer_features().")
                return

        with sns.plotting_context(context="notebook"):
            plt.figure(figsize=(15, 15))

            for i, feature in enumerate(self.feature_customer):
                plt.subplot(4, 4, i + 1)
                plt.hist(
                    data.iloc[:, i] if transformed else data[feature],
                    bins=30,
                    alpha=0.9,
                )
                plt.title(feature, fontsize=10)
                plt.ylabel("Tần suất", fontsize=8)

            plt.tight_layout()
            # plt.suptitle(title, fontsize=16, y=0.98)

            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                print(f"Plot saved: {save_path}")

            plt.show()

    def save_features(self, output_dir="../data/processed"):
        os.makedirs(output_dir, exist_ok=True)

        # origignal features
        customer_features_indexed = self.customer_features.set_index("CustomerID")
        customer_features_indexed.to_csv(f"{output_dir}/customer_features.csv")

        # transformed features
        self.customer_features_transformed.to_csv(
            f"{output_dir}/customer_features_transformed.csv"
        )

        # Lưu scaled features
        self.customer_features_scaled.to_csv(
            f"{output_dir}/customer_features_scaled.csv"
        )

        print(f"All features saved in: {output_dir}")
        
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
