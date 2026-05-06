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

    def load_data(self):
    
    def clean_data(self):

    def create_time_features(self):

    def calculate_rfm(self):
    
    def save_cleaned_data(self, output_dir="../data/processed"):

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
