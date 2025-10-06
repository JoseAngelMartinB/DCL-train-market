import os
import numpy as np
import pandas as pd

def load_data(path_data, y_col="price"):
    """Load data and separate features from target"""
    data = pd.read_csv(path_data, sep=",")
    y = data[y_col]
    X = data.drop(columns=[y_col], axis=1, inplace=False)
    return X, y

def sine_cosine_transform(df, col, period, drop=False):
    """Apply sine/cosine transformation to cyclical features"""
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    if drop:
        df.drop(columns=[col], inplace=True)
    return df

def process_categorical_features(df, cols_to_encode):
    """Apply one-hot encoding to categorical features"""
    return pd.get_dummies(df, columns=cols_to_encode, drop_first=False, dtype=int)

def remove_unused_columns(df, columns_to_drop):
    """Remove specified columns from dataframe"""
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=existing_columns_to_drop, inplace=True)
    return df

def filter_train_types(X, y, train_types=["OUIGO", "IRYO"]):
    """Filter data for specific train types"""
    if "train_type" in X.columns:
        mask = X["train_type"].isin(train_types)
        X_filtered = X[mask].copy()
        y_filtered = y[mask].copy()
        print(f"Data shape after filtering train types {train_types}: {X_filtered.shape}")
        return X_filtered, y_filtered
    else:
        print("Warning: 'train_type' column not found. Returning original data.")
        return X.copy(), y.copy()

def preprocess_price_data(input_path, output_path):
    """
    Complete preprocessing pipeline for price modeling data
    """
    print("Loading raw data...")
    X, y = load_data(input_path, y_col="price")
    
    print(f"Original data shape: {X.shape}")
    print(f"Original columns: {list(X.columns)}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")
    
    # Copy for processing
    X_processed = X.copy()
    
    # --- Apply sine/cosine transformation to temporal features ---
    print("Applying temporal transformations...")
    X_processed = sine_cosine_transform(X_processed, "month", 12, drop=True)
    X_processed = sine_cosine_transform(X_processed, "day_of_week", 7, drop=True)
    
    # --- Remove unused columns ---
    print("Removing unused columns...")
    columns_to_drop = [
        'capacity', 'train_model', 'year', 'departure_time', 'passengers'
    ] + [
        f"duration_competitor_{i}" for i in range(-2, 3) if i != 0
    ] + [
        f"train_type_competitor_{i}" for i in range(-2, 3) if i != 0
    ]
    
    X_processed = remove_unused_columns(X_processed, columns_to_drop)
    print(f"Columns after dropping unused ones: {list(X_processed.columns)}")
    
    # --- Filter for specific train types (OUIGO and IRYO) ---
    print("Filtering train types...")
    X_processed, y = filter_train_types(X_processed, y, train_types=["OUIGO", "IRYO"])
    
    # --- Apply one-hot encoding to categorical features ---
    print("Applying one-hot encoding...")
    cols_to_encode = ["train_type"] if "train_type" in X_processed.columns else []
    
    if cols_to_encode:
        X_processed = process_categorical_features(X_processed, cols_to_encode)
    
    print(f"Final processed data shape: {X_processed.shape}")
    print(f"Final features: {list(X_processed.columns)}")
    print(f"Target range after filtering: {y.min():.2f} - {y.max():.2f}")
    
    # --- Save processed data ---
    print(f"Saving processed data to {output_path}...")
    df_clean = X_processed.copy()
    df_clean["price"] = y
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_clean.to_csv(output_path, index=False)
    print(f"Processed data saved successfully!")
    
    return X_processed, y

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Input and output paths
    input_path = os.path.join(BASE_DIR, "../DataGenerationROBIN/data/MAD-BCN/aggregated/MAD-BCN_2025_RENFE.csv")
    output_path = os.path.join(BASE_DIR, "preprocesed_data/price_RENFE_MAD-BCN_2025.csv")
    
    # Run preprocessing
    X_processed, y = preprocess_price_data(input_path, output_path)
    
    print("\n=== Preprocessing Summary ===")
    print(f"Final dataset shape: {X_processed.shape}")
    print(f"Number of features: {X_processed.shape[1]}")
    print(f"Target variable statistics:")
    print(f"  Mean: {y.mean():.2f}")
    print(f"  Std: {y.std():.2f}")
    print(f"  Min: {y.min():.2f}")
    print(f"  Max: {y.max():.2f}")
    print(f"  25th percentile: {y.quantile(0.25):.2f}")
    print(f"  50th percentile (median): {y.quantile(0.50):.2f}")
    print(f"  75th percentile: {y.quantile(0.75):.2f}")
    
    print("\nPreprocessing complete!")
