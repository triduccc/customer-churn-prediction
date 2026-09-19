# Online Retail Customer Segmentation

Customer segmentation analysis for the UCI Online Retail transaction dataset. The project cleans UK purchases, builds customer-level behavioral features, and uses PCA and K-Means clustering to identify customer groups.

## Setup

```bash
pip install -r requirements.txt
```

Open the notebooks in Jupyter or VS Code and run them from the project root. The raw input file is expected at `data/online_retail.csv`.

## Project workflow

Run the notebooks in order:

1. `notebooks/cleaning_and_eda.ipynb` - load the raw transactions, remove cancellations and invalid records, keep UK customers, and explore the data.
2. `notebooks/feature_engineering.ipynb` - aggregate transactions by customer, apply Box-Cox transformations, and standardize the features.
3. `notebooks/modelling.ipynb` - apply PCA, compare cluster counts with the elbow and silhouette methods, and visualize K-Means segments.

## Generated data

The pipeline writes processed files to `data/processed/`:

- `cleaned_uk_data.csv`
- `customer_features.csv`
- `customer_features_transformed.csv`
- `customer_features_scaled.csv`

## Code structure

```text
├── data/
│   └── online_retail.csv       # Raw transaction data
├── notebooks/
│   ├── cleaning_and_eda.ipynb  # Cleaning and exploratory analysis
│   ├── feature_engineering.ipynb # Customer-level feature creation
│   └── modelling.ipynb         # PCA, K-Means, evaluation, and visualizations
├── src/
│   └── clustering.py           # Reusable analysis and visualization classes
└── requirements.txt             # Pinned Python dependencies
```

The main implementation is organized in `src/clustering.py`.


## Results

The project developed a pipeline that transformed raw transaction data into customer-level behavioral features and applied K-Means clustering to identify distinct customer groups. The resulting clusters were characterized based on purchasing behavior and transaction patterns, providing meaningful customer segments that can support further analysis and more targeted business strategies.



