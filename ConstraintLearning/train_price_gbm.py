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
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
from sklearn.ensemble import GradientBoostingRegressor

from utils import smape_score

# --- Config ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "preprocesed_data/price_RENFE_MAD-BCN_2025.csv")
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "saved_models/price_gbm_model.pkl")
FIG_PATH = os.path.join(BASE_DIR, "figures/price_gbm_")
TARGET_COL = "price"
UNUSED_COLS = ['service_id']
TEST_SIZE = 0.2
RANDOM_STATE = 2025
SHOWPLOTS = True  # Set to True to enable plots
PRICESENS = True  # Set to True to enable sensitivity analysis

# Create necessary directories
os.makedirs(os.path.join(BASE_DIR, "saved_models"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "figures"), exist_ok=True)

print("Loading preprocessed data...")

# --- Load preprocessed data ---
data = pd.read_csv(DATA_PATH)
print(f"Loaded data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Separate features and target
y = data[TARGET_COL]
columns_to_drop = [TARGET_COL] + UNUSED_COLS
missing_columns = [col for col in columns_to_drop if col not in data.columns]
if missing_columns:
    warnings.warn(
        f"Columns not found and skipped during drop: {missing_columns}",
        UserWarning,
    )
X = data.drop(columns=columns_to_drop, axis=1, inplace=False, errors="ignore")

# Store original feature ranges for constraints
feature_mins = X.min().values
feature_maxs = X.max().values

print(f"Dataset shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target range: {y.min():.2f} - {y.max():.2f}")

# --- Split the data ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
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

# Save target statistics
target_mean = target_scaler.mean_[0]
target_std = target_scaler.scale_[0]

print("Target standardized.")

# --- STEP 1: Analysis of n_estimators impact to find optimal efficiency ---
print("\n=== n_estimators Impact Analysis (with hyperparameter optimization) ===")

# Define hyperparameter grid WITHOUT n_estimators
PARAM_GRID_NO_NEST = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

n_estimators_to_test = [5, 10, 20, 50, 75, 100]

# Store results for plotting
nest_results = {
    'n_estimators': [],
    'train_r2': [],
    'val_r2': [],
    'train_mse': [],
    'val_mse': [],
    'train_mae': [],
    'val_mae': [],
    'best_params': []
}

print("Testing different n_estimators values with optimized hyperparameters:")
print("n_estimators | Train R2 | Val R2   | Train MSE | Val MSE  | Best Params")
print("-" * 100)

for n_est in n_estimators_to_test:
    print(f"Optimizing hyperparameters for n_estimators={n_est}...")
    
    # Create base model with fixed n_estimators
    base_model_fixed = GradientBoostingRegressor(
        n_estimators=n_est,
        random_state=RANDOM_STATE
    )
    
    # Perform grid search for other hyperparameters
    grid_search_nest = GridSearchCV(
        base_model_fixed, 
        PARAM_GRID_NO_NEST, 
        cv=3, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0  # Reduce verbosity
    )
    
    # Fit grid search
    grid_search_nest.fit(X_train_scaled, y_train_scaled)
    
    # Get best model for this n_estimators
    best_model_nest = grid_search_nest.best_estimator_
    
    # Predictions
    y_train_pred_test = best_model_nest.predict(X_train_scaled)
    y_val_pred_test = best_model_nest.predict(X_val_scaled)
    
    # Metrics
    train_r2_test = r2_score(y_train_scaled, y_train_pred_test)
    val_r2_test = r2_score(y_val_scaled, y_val_pred_test)
    train_mse_test = mean_squared_error(y_train_scaled, y_train_pred_test)
    val_mse_test = mean_squared_error(y_val_scaled, y_val_pred_test)
    train_mae_test = np.mean(np.abs(y_train_pred_test - y_train_scaled))
    val_mae_test = np.mean(np.abs(y_val_pred_test - y_val_scaled))
    
    # Store results
    nest_results['n_estimators'].append(n_est)
    nest_results['train_r2'].append(train_r2_test)
    nest_results['val_r2'].append(val_r2_test)
    nest_results['train_mse'].append(train_mse_test)
    nest_results['val_mse'].append(val_mse_test)
    nest_results['train_mae'].append(train_mae_test)
    nest_results['val_mae'].append(val_mae_test)
    nest_results['best_params'].append(grid_search_nest.best_params_)
    
    # Format best params for display
    params_str = str(grid_search_nest.best_params_)[:40] + "..." if len(str(grid_search_nest.best_params_)) > 40 else str(grid_search_nest.best_params_)
    
    print(f"{n_est:11d} | {train_r2_test:8.4f} | {val_r2_test:8.4f} | {train_mse_test:9.4f} | {val_mse_test:8.4f} | {params_str}")

# Find the minimum n_estimators that achieves 97.5% of best validation MSE
best_val_mse = min(nest_results['val_mse'])  # Best MSE is the minimum
threshold_mse = 1.025 * best_val_mse  # 97.5% of best MSE means within 2.5% of best

optimal_n_estimators = None
for i, (n_est, val_mse) in enumerate(zip(nest_results['n_estimators'], nest_results['val_mse'])):
    if val_mse <= threshold_mse:
        optimal_n_estimators = n_est
        break

print(f"\nBest validation MSE: {best_val_mse:.4f}")
print(f"97.5% threshold MSE: {threshold_mse:.4f}")
if optimal_n_estimators:
    print(f"Minimum n_estimators achieving 97.5% of best MSE: {optimal_n_estimators}")
else:
    print("No n_estimators achieved 97.5% of best MSE in tested range")

# Find the point of diminishing returns (where MSE improvement is < 0.1% per additional tree)
improvement_rates = []
for i in range(1, len(nest_results['val_mse'])):
    prev_mse = nest_results['val_mse'][i-1]
    curr_mse = nest_results['val_mse'][i]
    prev_n = nest_results['n_estimators'][i-1]
    curr_n = nest_results['n_estimators'][i]
    
    # For MSE, improvement is a decrease, so we calculate the relative improvement
    if prev_mse > 0:
        improvement_per_tree = (prev_mse - curr_mse) / prev_mse / (curr_n - prev_n) if curr_n > prev_n else 0
    else:
        improvement_per_tree = 0
    improvement_rates.append(improvement_per_tree)

if improvement_rates:
    min_improvement_threshold = 0.001 / 100  # 0.1% MSE improvement per tree
    diminishing_returns_idx = None
    for i, rate in enumerate(improvement_rates):
        if rate < min_improvement_threshold:
            diminishing_returns_idx = i + 1  # +1 because improvement_rates is offset by 1
            break
    
    if diminishing_returns_idx:
        diminishing_returns_n = nest_results['n_estimators'][diminishing_returns_idx]
        print(f"Diminishing returns start at n_estimators: {diminishing_returns_n} (MSE improvement < 0.1% per tree)")

# --- Calculate efficiency scores to find the most efficient n_estimators ---
# Normalize metrics to [0,1] for comparison
# For MSE, normalize inversely (lower MSE is better)
val_mse_norm = 1 - (np.array(nest_results['val_mse']) - min(nest_results['val_mse'])) / (max(nest_results['val_mse']) - min(nest_results['val_mse']))
complexity_norm = 1 - np.array(nest_results['n_estimators']) / max(nest_results['n_estimators'])  # Inverted so higher is better

# Calculate efficiency score (weighted average of performance and simplicity)
efficiency_scores = 0.75 * val_mse_norm + 0.25 * complexity_norm  # 75% performance, 25% simplicity

# Find and mark the most efficient point
best_efficiency_idx = np.argmax(efficiency_scores)
best_efficiency_n = nest_results['n_estimators'][best_efficiency_idx]

print(f"\nEfficiency Analysis:")
print(f"Most efficient n_estimators (75% MSE performance + 25% simplicity): {best_efficiency_n}")
print(f"Efficiency score: {efficiency_scores[best_efficiency_idx]:.4f}")

# --- STEP 2: Train final model with optimal n_estimators ---
print(f"\n=== Training Final Model with n_estimators={best_efficiency_n} ===")

# Get the best hyperparameters for the most efficient n_estimators
final_best_params = nest_results['best_params'][best_efficiency_idx].copy()
final_best_params['n_estimators'] = best_efficiency_n

print(f"Final model parameters: {final_best_params}")

# Train final model
best_model = GradientBoostingRegressor(
    **final_best_params,
    random_state=RANDOM_STATE
)

best_model.fit(X_train_scaled, y_train_scaled)

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
    'feature_means': feature_means,
    'feature_stds': feature_stds,
    'feature_mins': feature_mins,
    'feature_maxs': feature_maxs,
    'target_mean': target_mean,
    'target_std': target_std,
    'best_params': final_best_params,
    'scaled_features': scaled_feat,
    'efficiency_analysis': {
        'nest_results': nest_results,
        'best_efficiency_n': best_efficiency_n,
        'efficiency_scores': efficiency_scores.tolist()
    }
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
print(f"sMAPE: {smape_score(y_val_pred_inv, y_val_true_inv):.4f}")
print(f"Mean true: {np.mean(y_val_true_inv):.2f}, Mean pred: {np.mean(y_val_pred_inv):.2f}")
print(f"Number of negative predictions: {np.sum(y_val_pred_inv + 1e-6 < 0)}")

# --- STEP 3: Generate all plots and analysis ---
if SHOWPLOTS:
    
    # --- Plot n_estimators impact on R2 ---
    plt.figure(figsize=(12, 5))
    
    # R2 plot
    plt.subplot(1, 2, 1)
    plt.plot(nest_results['n_estimators'], nest_results['train_r2'], 'o-', label='Train R²', color='blue', alpha=0.7)
    plt.plot(nest_results['n_estimators'], nest_results['val_r2'], 'o-', label='Validation R²', color='orange', alpha=0.7)
    if optimal_n_estimators:
        plt.axvline(x=optimal_n_estimators, color='red', linestyle='--', alpha=0.7, label=f'97.5% threshold (n={optimal_n_estimators})')
    if 'diminishing_returns_n' in locals() and diminishing_returns_n:
        plt.axvline(x=diminishing_returns_n, color='green', linestyle='--', alpha=0.7, label=f'Diminishing returns (n={diminishing_returns_n})')
    plt.xlabel('Number of Estimators')
    plt.ylabel('R² Score')
    plt.title('R² vs Number of Estimators')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # MSE plot
    plt.subplot(1, 2, 2)
    plt.plot(nest_results['n_estimators'], nest_results['train_mse'], 'o-', label='Train MSE', color='blue', alpha=0.7)
    plt.plot(nest_results['n_estimators'], nest_results['val_mse'], 'o-', label='Validation MSE', color='orange', alpha=0.7)
    if optimal_n_estimators:
        plt.axvline(x=optimal_n_estimators, color='red', linestyle='--', alpha=0.7, label=f'97.5% threshold (n={optimal_n_estimators})')
    if 'diminishing_returns_n' in locals() and diminishing_returns_n:
        plt.axvline(x=diminishing_returns_n, color='green', linestyle='--', alpha=0.7, label=f'Diminishing returns (n={diminishing_returns_n})')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs Number of Estimators')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(FIG_PATH + "n_estimators_analysis.pdf")
    plt.close()
    
    # --- Plot improvement per tree ---
    if improvement_rates:
        plt.figure(figsize=(10, 6))
        plt.plot(nest_results['n_estimators'][1:], np.array(improvement_rates) * 10000, 'o-', color='purple', alpha=0.7)
        plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='0.1% per tree threshold')
        plt.xlabel('Number of Estimators')
        plt.ylabel('MSE Improvement per Tree (×10⁴)')
        plt.title('Marginal MSE Improvement per Additional Tree')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(FIG_PATH + "marginal_improvement_per_tree.pdf")
        plt.close()
    
    # --- Plot performance vs complexity trade-off ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(nest_results['n_estimators'], val_mse_norm, 'o-', label='Normalized Val MSE (inverted)', color='blue', alpha=0.7)
    plt.plot(nest_results['n_estimators'], complexity_norm, 'o-', label='Normalized Simplicity (1-n/max_n)', color='red', alpha=0.7)
    plt.plot(nest_results['n_estimators'], efficiency_scores, 'o-', label='Efficiency Score (0.75×MSE + 0.25×Simplicity)', color='green', alpha=0.7, linewidth=2)
    
    # Mark the most efficient point
    plt.axvline(x=best_efficiency_n, color='green', linestyle='--', alpha=0.7, label=f'Most efficient (n={best_efficiency_n})')
    
    plt.xlabel('Number of Estimators')
    plt.ylabel('Normalized Score')
    plt.title('Performance vs Complexity Trade-off Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(FIG_PATH + "performance_complexity_tradeoff.pdf")
    plt.close()
    
    # --- Plot predictions vs true values ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_val_true_inv, y_val_pred_inv, alpha=0.3)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Price GBM: Predicted vs True Values")
    plt.plot([y_val_true_inv.min(), y_val_true_inv.max()], [y_val_true_inv.min(), y_val_true_inv.max()], 'r--')
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "predicted_vs_true.pdf")
    plt.close()

    # --- Plot histogram of prediction errors ---
    plt.figure(figsize=(8, 6))
    errors = y_val_pred_inv - y_val_true_inv
    plt.hist(errors, bins=200, color="blue", alpha=0.7)
    plt.xlim(-50, 50)
    plt.xlabel("Prediction Error (Predicted - True)")
    plt.ylabel("Frequency")
    plt.title("Price GBM: Histogram of Prediction Errors")
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "prediction_errors_histogram.pdf")
    plt.close()

    # --- Plot histogram of true vs predicted values ---
    plt.figure(figsize=(10, 6))
    plt.hist(y_val_true_inv, bins=200, histtype='step', linewidth=2, color='blue', label='True Values', density=True)
    plt.hist(y_val_pred_inv, bins=200, histtype='step', linewidth=2, color='orange', label='Predicted Values', density=True)
    plt.title("Price GBM: Histogram of True vs Predicted Values")
    plt.xlabel("Price (€)")
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
    plt.title("Price GBM: Empirical CDF of True vs Predicted Values")
    plt.xlabel("Price (€)")
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
    print(f"Percentage of instances with prediction within 5€ of the real value: {count_within_distance(y_val_true_inv, y_val_pred_inv, 5)[1]}%")
    print(f"Percentage of instances with prediction within 10€ of the real value: {count_within_distance(y_val_true_inv, y_val_pred_inv, 10)[1]}%")
    print(f"Percentage of instances with prediction within 20€ of the real value: {count_within_distance(y_val_true_inv, y_val_pred_inv, 20)[1]}%")

    # --- Feature Importance from GBM ---
    print("\n=== GBM Feature Importance ===")
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
    plt.title("Price GBM: Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(FIG_PATH + "feature_importance.pdf")
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
    plt.title("Price GBM: Top 15 Permutation Feature Importances")
    plt.tight_layout()
    plt.savefig(FIG_PATH + "permutation_feature_importance.pdf")
    plt.close()

    # --- GBM specific analysis ---
    print(f"\n=== GBM Model Info ===")
    print(f"Number of estimators: {best_model.n_estimators}")
    print(f"Max depth: {best_model.max_depth}")
    print(f"Learning rate: {best_model.learning_rate}")
    print(f"Subsample: {best_model.subsample}")
    print(f"Max features: {best_model.max_features}")

# --- Partial Dependence Analysis ---
if PRICESENS:
    print("\n=== Partial Dependence Analysis ===")
    
    # For price modeling, we might want to see how competitor prices affect our price
    competitor_price_cols = [col for col in X_val_scaled.columns if 'price_competitor' in col]
    
    if competitor_price_cols:
        # Analyze partial dependence on competitor prices
        for comp_price_col in competitor_price_cols[:2]:  # Limit to first 2 for visualization
            if comp_price_col in X_val_scaled.columns:
                comp_idx = list(X_val_scaled.columns).index(comp_price_col)
                
                # Find the original scale range for this competitor price
                if comp_price_col in scaled_feat:
                    scaler_idx = scaled_feat.index(comp_price_col)
                    comp_min_orig = (X_val_scaled[comp_price_col].min() * feature_stds[scaler_idx] + feature_means[scaler_idx])
                    comp_max_orig = (X_val_scaled[comp_price_col].max() * feature_stds[scaler_idx] + feature_means[scaler_idx])
                    comp_max_orig = min(90, comp_max_orig)  # Cap at reasonable price
                    comp_prices_orig = np.linspace(comp_min_orig, comp_max_orig, 50)
                    
                    # Scale the original prices for prediction
                    comp_prices_scaled = (comp_prices_orig - feature_means[scaler_idx]) / feature_stds[scaler_idx]
                else:
                    comp_prices_scaled = np.linspace(X_val_scaled[comp_price_col].min(), X_val_scaled[comp_price_col].max(), 50)
                    comp_prices_orig = comp_prices_scaled
                
                # Use a few representative samples as baseline
                n_baselines = min(50, len(X_val_scaled))
                baseline_indices = np.random.choice(X_val_scaled.shape[0], n_baselines, replace=False)
                baselines = X_val_scaled.iloc[baseline_indices]
                
                all_responses = []
                for _, baseline_row in baselines.iterrows():
                    # Create variations for this competitor price
                    samples_matrix = np.tile(baseline_row.values, (len(comp_prices_scaled), 1))
                    samples_matrix[:, comp_idx] = comp_prices_scaled
                    
                    # Create DataFrame for prediction
                    samples_df = pd.DataFrame(samples_matrix, columns=X_val_scaled.columns)
                    
                    # Make predictions
                    pred_scaled_batch = best_model.predict(samples_df)
                    pred_orig_batch = target_scaler.inverse_transform(pred_scaled_batch.reshape(-1, 1)).flatten()
                    
                    all_responses.append(pred_orig_batch)
                
                all_responses = np.array(all_responses)
                mean_response = all_responses.mean(axis=0)
                percentile_05 = np.percentile(all_responses, 5, axis=0)
                percentile_95 = np.percentile(all_responses, 95, axis=0)
                
                plt.figure(figsize=(10, 6))
                plt.plot(comp_prices_orig, mean_response, label=f'Mean Response', color='blue', linewidth=2)
                plt.fill_between(comp_prices_orig, percentile_05, percentile_95, color='blue', alpha=0.15, label='5th-95th percentile')
                
                plt.xlabel(f'{comp_price_col.replace("_", " ").title()} (€)')
                plt.ylabel('Predicted Price (€)')
                plt.title(f'Price GBM: Partial Dependence on {comp_price_col.replace("_", " ").title()}')
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.savefig(FIG_PATH + f"partial_dependence_{comp_price_col}.pdf")
                plt.close()
    
    # Also create general partial dependence plot for duration if available
    if 'duration' in X_val_scaled.columns:
        duration_idx = list(X_val_scaled.columns).index('duration')
        
        if 'duration' in scaled_feat:
            scaler_idx = scaled_feat.index('duration')
            duration_min_orig = (X_val_scaled['duration'].min() * feature_stds[scaler_idx] + feature_means[scaler_idx])
            duration_max_orig = (X_val_scaled['duration'].max() * feature_stds[scaler_idx] + feature_means[scaler_idx])
            durations_orig = np.linspace(duration_min_orig, duration_max_orig, 50)
            durations_scaled = (durations_orig - feature_means[scaler_idx]) / feature_stds[scaler_idx]
        else:
            durations_scaled = np.linspace(X_val_scaled['duration'].min(), X_val_scaled['duration'].max(), 50)
            durations_orig = durations_scaled
        
        # Use representative samples for baseline
        n_baselines = min(50, len(X_val_scaled))
        baseline_indices = np.random.choice(X_val_scaled.shape[0], n_baselines, replace=False)
        baselines = X_val_scaled.iloc[baseline_indices]
        
        all_responses = []
        for _, baseline_row in baselines.iterrows():
            samples_matrix = np.tile(baseline_row.values, (len(durations_scaled), 1))
            samples_matrix[:, duration_idx] = durations_scaled
            
            samples_df = pd.DataFrame(samples_matrix, columns=X_val_scaled.columns)
            pred_scaled_batch = best_model.predict(samples_df)
            pred_orig_batch = target_scaler.inverse_transform(pred_scaled_batch.reshape(-1, 1)).flatten()
            
            all_responses.append(pred_orig_batch)
        
        all_responses = np.array(all_responses)
        mean_response = all_responses.mean(axis=0)
        percentile_05 = np.percentile(all_responses, 5, axis=0)
        percentile_95 = np.percentile(all_responses, 95, axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(durations_orig, mean_response, label='Mean Response', color='green', linewidth=2)
        plt.fill_between(durations_orig, percentile_05, percentile_95, color='green', alpha=0.15, label='5th-95th percentile')
        
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Predicted Price (€)')
        plt.title('Price GBM: Partial Dependence on Duration')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(FIG_PATH + "partial_dependence_duration.pdf")
        plt.close()

print(f"\nAll figures saved with prefix: {FIG_PATH}")
print("Price GBM training complete!")
