import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay

from utils import *

# --- Config ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "preprocesed_data/demand_MAD-BCN_2025.csv")
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "saved_models/demand_tree_model.pkl")
FIG_PATH = os.path.join(BASE_DIR, "figures/demand_tree_")
TARGET_COL = "passengers"
UNUSED_COLS = ['service_id', 'capacity']
TEST_SIZE = 0.2
RANDOM_STATE = 2025
SHOWPLOTS = False  # Set to False to disable plots
PRICESENS = False  # Set to False to disable sensitivity analysis

# Grid search parameters for Decision Tree
PARAM_GRID = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [5, 10, 15, 20],
    'min_samples_leaf': [2, 5, 7, 10],
    'max_features': ['sqrt', 'log2', None]
}

print("Loading and preprocessing data...")

# --- Load and preprocess data ---
data = pd.read_csv(DATA_PATH)
y = data[TARGET_COL]
columns_to_drop = [TARGET_COL] + UNUSED_COLS
missing_columns = [col for col in columns_to_drop if col not in data.columns]
if missing_columns:
    warnings.warn(
        f"Columns not found and skipped during drop: {missing_columns}",
        UserWarning,
    )
X = data.drop(columns=columns_to_drop, axis=1, inplace=False, errors="ignore")
feature_mins = X.min().values
feature_maxs = X.max().values

print(f"Data shape: {data.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target range: {y.min():.2f} - {y.max():.2f}")

# --- Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# --- Standardize features ---
feat_scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()

# Scale ALL features (including categorical and trigonometric features)
scaled_feat = list(X.columns)  # Scale all features
print(f"Features to be scaled: {scaled_feat}")
if scaled_feat:
    X_train_scaled[scaled_feat] = feat_scaler.fit_transform(X_train[scaled_feat])
    X_val_scaled[scaled_feat] = feat_scaler.transform(X_val[scaled_feat])

# Convert to DataFrame to maintain column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)

# Save feature mean and std
feature_means = feat_scaler.mean_ if scaled_feat else np.zeros(len(X.columns))
feature_stds = feat_scaler.scale_ if scaled_feat else np.ones(len(X.columns))

print("Features standardized.")

# --- Standardize target ---
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1)).flatten()

# Save target mean and std
target_mean = target_scaler.mean_[0]
target_std = target_scaler.scale_[0]

print("Target standardized.")

# --- Model Training with Grid Search ---
print("Starting grid search for optimal hyperparameters...")

# Create base model
base_model = DecisionTreeRegressor(random_state=RANDOM_STATE)

# Perform grid search
grid_search = GridSearchCV(
    base_model, 
    PARAM_GRID, 
    cv=3, 
    scoring='neg_mean_squared_error',  # 'r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'
    n_jobs=-1,
    verbose=1
)

# Fit grid search on scaled data
grid_search.fit(X_train_scaled, y_train_scaled)

# Get best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# --- Make predictions ---
y_train_pred = best_model.predict(X_train_scaled)
y_val_pred = best_model.predict(X_val_scaled)

# Calculate metrics on scaled data
train_r2 = r2_score(y_train_scaled, y_train_pred)
val_r2 = r2_score(y_val_scaled, y_val_pred)
train_mae = np.mean(np.abs(y_train_pred - y_train_scaled))
val_mae = np.mean(np.abs(y_val_pred - y_val_scaled))
train_mse = mean_squared_error(y_train_scaled, y_train_pred)
val_mse = mean_squared_error(y_val_scaled, y_val_pred)

print(f"Training R2: {train_r2:.4f}")
print(f"Validation R2: {val_r2:.4f}")
print(f"Training MAE (scaled): {train_mae:.4f}")
print(f"Validation MAE (scaled): {val_mae:.4f}")
print(f"Training MSE (scaled): {train_mse:.4f}")
print(f"Validation MSE (scaled): {val_mse:.4f}")

# --- Save model ---
model_data = {
    'model': best_model,
    'feature_scaler': feat_scaler,
    'target_scaler': target_scaler,
    'scaled_features': scaled_feat,
    'feature_means': feature_means,
    'feature_stds': feature_stds,
    'feature_mins': feature_mins,
    'feature_maxs': feature_maxs,
    'target_mean': target_mean,
    'target_std': target_std,
    'best_params': grid_search.best_params_
}

joblib.dump(model_data, SAVED_MODEL_PATH)
print(f"Model saved to {SAVED_MODEL_PATH}")

# --- Evaluation on original scale ---
y_val_pred_inv = target_scaler.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
y_val_true_inv = y_val.values

print("\n=== Final Results (Original Scale) ===")
print(f"Final R2: {r2_score(y_val_true_inv, y_val_pred_inv):.4f}")
print(f"Final RMSE: {np.sqrt(mean_squared_error(y_val_true_inv, y_val_pred_inv)):.4f}")
print(f"Final MSE: {mean_squared_error(y_val_true_inv, y_val_pred_inv):.4f}")
print(f"Mean absolute error: {np.mean(np.abs(y_val_pred_inv - y_val_true_inv)):.4f}")
print(f"Mean true: {np.mean(y_val_true_inv):.2f}, Mean pred: {np.mean(y_val_pred_inv):.2f}")
print(f"Number of negative predictions: {np.sum(y_val_pred_inv + 1e-6 < 0)}")

# --- Plot Results ---
if SHOWPLOTS:
    
    # --- Plot predictions vs true values ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_val_true_inv, y_val_pred_inv, alpha=0.3)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Decision Tree: Predicted vs True Values")
    plt.plot([y_val_true_inv.min(), y_val_true_inv.max()], [y_val_true_inv.min(), y_val_true_inv.max()], 'r--')
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "predicted_vs_true.pdf")
    plt.close()  # Close figure instead of show

    # --- Plot histogram of prediction errors ---
    plt.figure(figsize=(8, 6))
    errors = y_val_pred_inv - y_val_true_inv
    plt.hist(errors, bins=200, color="blue", alpha=0.7)
    plt.xlim(-300, 300)
    plt.xlabel("Prediction Error (Predicted - True)")
    plt.ylabel("Frequency")
    plt.title("Decision Tree: Histogram of Prediction Errors")
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "prediction_errors_histogram.pdf")
    plt.close()

    # --- Plot histogram of true vs predicted values ---
    plt.figure(figsize=(10, 6))
    plt.hist(y_val_true_inv, bins=200, histtype='step', linewidth=2, color='blue', label='True Values', density=True)
    plt.hist(y_val_pred_inv, bins=200, histtype='step', linewidth=2, color='orange', label='Predicted Values', density=True)
    plt.title("Decision Tree: Histogram of True vs Predicted Values")
    plt.xlabel("Number of Passengers")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "true_vs_predicted_histogram.pdf")
    plt.close()

    # --- Plot Empirical CDF of True vs Predicted Values ---
    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(y_val_true_inv), np.linspace(0, 1, len(y_val_true_inv), endpoint=False), label="True Values", linewidth=2)
    plt.plot(np.sort(y_val_pred_inv), np.linspace(0, 1, len(y_val_pred_inv), endpoint=False), label="Predicted Values", linewidth=2)
    plt.title("Decision Tree: Empirical CDF of True vs Predicted Values")
    plt.xlabel("Number of Passengers")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "empirical_cdf_true_vs_predicted.pdf")
    plt.close()

    # --- Count number of instances at a distance < d of the real value ---
    def count_within_distance(y_true, y_pred, distance):
        n_instances = np.sum(np.abs(y_pred - y_true) < distance)
        percentage = np.round((n_instances / len(y_true)) * 100, 2)
        return n_instances, percentage

    print(f"\nAccuracy Analysis:")
    print(f"Percentage of instances with prediction within 10 of the real value: {count_within_distance(y_val_true_inv, y_val_pred_inv, 10)[1]}%")
    print(f"Percentage of instances with prediction within 25 of the real value: {count_within_distance(y_val_true_inv, y_val_pred_inv, 25)[1]}%")
    print(f"Percentage of instances with prediction within 50 of the real value: {count_within_distance(y_val_true_inv, y_val_pred_inv, 50)[1]}%")

    # --- Feature Importance from Tree ---
    print("\n=== Tree Feature Importance ===")
    feature_importance = best_model.feature_importances_
    feature_names = X_train_scaled.columns
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    
    print("Top 10 most important features:")
    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"{feature_names[idx]:30s}: {feature_importance[idx]:.4f}")
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.bar(np.array(feature_names)[sorted_idx[:15]], feature_importance[sorted_idx[:15]])
    plt.xticks(rotation=90)
    plt.ylabel("Feature Importance")
    plt.title("Decision Tree: Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(FIG_PATH + "tree_feature_importance.pdf")
    plt.close()

    # --- Permutation Feature Importance ---
    print("\n=== Permutation Feature Importance ===")
    perm_importance = permutation_importance(best_model, X_val_scaled, y_val_scaled, n_repeats=10, random_state=RANDOM_STATE)
    
    sorted_idx_perm = perm_importance.importances_mean.argsort()[::-1]
    
    print("Top 10 features by permutation importance:")
    for i in range(min(10, len(sorted_idx_perm))):
        idx = sorted_idx_perm[i]
        print(f"{feature_names[idx]:30s}: {perm_importance.importances_mean[idx]:.4f} ± {perm_importance.importances_std[idx]:.4f}")
    
    # Plot permutation feature importances
    plt.figure(figsize=(12, 6))
    plt.bar(np.array(feature_names)[sorted_idx_perm[:15]], 
            perm_importance.importances_mean[sorted_idx_perm[:15]],
            yerr=perm_importance.importances_std[sorted_idx_perm[:15]])
    plt.xticks(rotation=90)
    plt.ylabel("Drop in R² when permuted")
    plt.title("Decision Tree: Top 15 Permutation Feature Importances")
    plt.tight_layout()
    plt.savefig(FIG_PATH + "permutation_feature_importance.pdf")
    plt.close()

    # --- Tree Visualization (for smaller trees) ---
    max_depth = grid_search.best_params_.get('max_depth')
    if max_depth is not None and max_depth <= 5:
        plt.figure(figsize=(20, 10))
        plot_tree(best_model, 
                 feature_names=feature_names, 
                 max_depth=3,  # Limit depth for readability
                 filled=True, 
                 fontsize=8)
        plt.title("Decision Tree Structure (Limited to depth 3 for readability)")
        plt.tight_layout()
        plt.savefig(FIG_PATH + "tree_structure.pdf")
        plt.close()
    else:
        if max_depth is None:
            print(f"Tree has unlimited depth (max_depth=None) - too deep for visualization.")
        else:
            print(f"Tree too deep (max_depth={max_depth}) for visualization.")

# --- Partial Dependence Plot for Price ---
if PRICESENS:
    print("\n=== Partial Dependence Analysis ===")
    
    # Check if 'price' column exists
    if 'price' in X_val_scaled.columns:
        price_idx = list(X_val_scaled.columns).index('price')
        
        # Create price range (in original scale)
        price_min_orig = feature_mins[price_idx]
        price_max_orig = min(90, feature_maxs[price_idx])  # Cap at 90 as in original
        prices_orig = np.linspace(price_min_orig, price_max_orig, 100)
        
        train_types = ['AVE', 'IRYO', 'OUIGO']
        colors = ['tab:blue', 'tab:green', 'tab:red']
        
        plt.figure(figsize=(10, 6))
        
        for train_type, color in zip(train_types, colors):
            col_name = f'train_type_{train_type}'
            if col_name not in X_val_scaled.columns:
                print(f"Column {col_name} not found in X_val_scaled. Skipping.")
                continue
                
            # Find samples with this train type
            train_type_idx = list(X_val_scaled.columns).index(col_name)
            standardized_1 = (1 - feature_means[train_type_idx]) / feature_stds[train_type_idx]
            mask = np.isclose(X_val_scaled[col_name], standardized_1)
            eligible_baselines = X_val_scaled[mask]
            
            if len(eligible_baselines) == 0:
                print(f"No baselines found for {col_name}=1. Skipping.")
                continue
                
            # Use up to 100 baseline samples
            n_baselines = min(100, len(eligible_baselines))
            baseline_indices = np.random.choice(eligible_baselines.shape[0], n_baselines, replace=False)
            baselines = eligible_baselines.iloc[baseline_indices]
            
            all_responses = []
            for _, baseline_row in baselines.iterrows():
                responses = []
                for price_orig in prices_orig:
                    # Scale the price
                    price_scaled = (price_orig - feature_means[price_idx]) / feature_stds[price_idx]
                    
                    # Create sample with modified price
                    sample = baseline_row.copy()
                    sample['price'] = price_scaled
                    
                    # Make prediction - keep as DataFrame to preserve feature names
                    sample_df = pd.DataFrame([sample], columns=X_val_scaled.columns)
                    pred_scaled = best_model.predict(sample_df)[0]
                    pred_orig = target_scaler.inverse_transform([[pred_scaled]])[0, 0]
                    responses.append(pred_orig)
                    
                all_responses.append(responses)
                
            all_responses = np.array(all_responses)
            mean_response = all_responses.mean(axis=0)
            percentile_05 = np.percentile(all_responses, 5, axis=0)
            percentile_95 = np.percentile(all_responses, 95, axis=0)
            
            plt.plot(prices_orig, mean_response, label=f'{train_type}', color=color, linewidth=2)
            plt.fill_between(prices_orig, 
                           percentile_05, 
                           percentile_95,
                           color=color, alpha=0.15)
        
        plt.xlabel('Price (€)')
        plt.ylabel('Predicted Passengers')
        plt.ylim(0, 1000)
        plt.title('Decision Tree: Partial Dependence of Passengers on Price\n(baselines with train_type=1)')
        plt.legend(title='Train Type')
        plt.grid()
        plt.tight_layout()
        plt.savefig(FIG_PATH + "partial_dependence_price.pdf")
        plt.close()
        
    else:
        print("Price column not found for partial dependence analysis.")

print(f"\nAll figures saved with prefix: {FIG_PATH}")
print("Analysis complete!")
