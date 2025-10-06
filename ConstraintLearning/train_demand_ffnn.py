import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "preprocesed_data/demand_MAD-BCN_2025.csv")
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "saved_models/demand_ffnn_model.pt")
FIG_PATH = os.path.join(BASE_DIR, "figures/demand_ffnn_")
TARGET_COL = "passengers"
UNUSED_COLS = ['service_id', 'capacity']
BATCH_SIZE = 256
N_EPOCHS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 50  # Early stopping patience
HIDDEN_LAYERS = [64, 8]
DROPOUT = 0.1
TEST_SIZE = 0.2
RANDOM_STATE = 2025
SHOWPLOTS = True  # Set to False to disable plots
PRICESENS = True  # Set to False to disable sensitivity analysis

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

# --- Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
)


## --- Standardize features ---
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
y_train = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_val = target_scaler.transform(y_val.values.reshape(-1, 1))

# Save target mean and std
target_mean = target_scaler.mean_[0]
target_std = target_scaler.scale_[0]


# --- Torch tensors ---
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


# --- Model ---
input_size = X_train.shape[1]
output_size = 1
model = FeedForwardNN(input_size, output_size, HIDDEN_LAYERS, DROPOUT)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.L1Loss()  # nn.MSELoss() #nn.L1Loss()

print(summary(model, input_size=(input_size,)))

# Add ReduceLROnPlateau scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.9,
    patience=10,
    verbose=True
)

# --- Track losses and R2 during training ---
train_losses = []
val_losses = []
val_r2s = []
val_smapes = []

best_loss = float("inf")
epochs_no_improve = 0  # Counter for early stopping
best_epoch = -1

# --- Training loop ---
for epoch in range(N_EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            y_true.append(batch_y.numpy())
            y_pred.append(outputs.numpy())
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()
    r2 = r2_score(y_true, y_pred)
    val_r2s.append(r2)
    smape = smape_score(y_pred, y_true)
    val_smapes.append(smape)
    print(
        f"Epoch {epoch + 1}/{N_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val R2: {r2:.4f} - Val sMAPE: {smape:.2f} - LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # Step the scheduler
    scheduler.step(val_loss)

    # --- Save best model & Early Stopping ---
    if val_loss < best_loss - 1e-8:  # Allow for very small numerical differences
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_means": feature_means,
                "feature_stds": feature_stds,
                "feature_mins": feature_mins,
                "feature_maxs": feature_maxs,
                "target_mean": target_mean,
                "target_std": target_std,
            },
            SAVED_MODEL_PATH,
        )
        best_loss = val_loss
        best_epoch = epoch
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {epoch + 1} epochs – "
                f"no improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

# --- Load best model before evaluation ---
checkpoint = torch.load(SAVED_MODEL_PATH, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
feature_means = checkpoint["feature_means"]
feature_stds = checkpoint["feature_stds"]
feature_mins = checkpoint["feature_mins"]
feature_maxs = checkpoint["feature_maxs"]
target_mean = checkpoint["target_mean"]
target_std = checkpoint["target_std"]
print(f"Loaded best model from epoch {best_epoch + 1} with val loss {best_loss:.4f}")

# --- Evaluation ---
y_pred_inv = y_pred * target_scaler.scale_[0] + target_scaler.mean_[0]
y_true_inv = y_true * target_scaler.scale_[0] + target_scaler.mean_[0]

print("Final R2:", r2_score(y_true_inv, y_pred_inv))
print("Final RMSE:", np.sqrt(mean_squared_error(y_true_inv, y_pred_inv)))
print("Final sMAPE:", smape_score(y_pred_inv, y_true_inv))
print("Mean abs error:", np.mean(np.abs(y_pred_inv - y_true_inv)))
print("Mean true:", np.mean(y_true_inv), "Mean pred:", np.mean(y_pred_inv))
print("Number of negative predictions:", np.sum(y_pred_inv < 0))


# --- Plot Loss and R2 curves ---
if SHOWPLOTS:
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "loss_curves.pdf")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(val_r2s, label="Validation R2", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.title("Validation R2 Over Epochs")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "val_r2_curve.pdf")
    plt.show()

    # --- Plot predictions vs true values ---
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_inv, y_pred_inv, alpha=0.3)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs True Values")
    plt.plot(
        [y_true_inv.min(), y_true_inv.max()],
        [y_true_inv.min(), y_true_inv.max()],
        "r--",
    )
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "predicted_vs_true.pdf")
    plt.show()

    # --- Plot histogram of prediction errors ---
    plt.figure(figsize=(8, 6))
    plt.hist(y_pred_inv - y_true_inv, bins=200, color="blue", alpha=0.7)
    plt.xlim(-300, 300)
    plt.xlabel("Prediction Error (Predicted - True)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Prediction Errors")
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "prediction_errors_histogram.pdf")
    plt.show()

    # --- Plot histogram of true vs predicted values ---
    plt.figure(figsize=(10, 6))
    plt.hist(
        y_true_inv,
        bins=200,
        histtype="step",
        linewidth=2,
        color="blue",
        label="True Values",
        density=True,
    )
    plt.hist(
        y_pred_inv,
        bins=200,
        histtype="step",
        linewidth=2,
        color="orange",
        label="Predicted Values",
        density=True,
    )
    plt.title("Histogram of True vs Predicted Values")
    plt.xlabel("Number of Passengers")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "true_vs_predicted_histogram.pdf")
    plt.show()

    # --- Plot Empirical CDF of True vs Predicted Values ---
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.sort(y_true_inv),
        np.linspace(0, 1, len(y_true_inv), endpoint=False),
        label="True Values",
        linewidth=2,
    )
    plt.plot(
        np.sort(y_pred_inv),
        np.linspace(0, 1, len(y_pred_inv), endpoint=False),
        label="Predicted Values",
        linewidth=2,
    )
    plt.title(
        "Empirical Cumulative Distribution Function (CDF) of True vs Predicted Values"
    )
    plt.xlabel("Number of Passengers")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "empirical_cdf_true_vs_predicted.pdf")
    plt.show()

    # --- Count number of instances at a distance < d of the real value ---
    def count_within_distance(y_true, y_pred, distance):
        n_instances = np.sum(np.abs(y_pred - y_true) < distance)
        percentage = np.round((n_instances / len(y_true)) * 100, 2)
        return n_instances, percentage

    print(
        f"Percentage of instances with prediction within 10 of the real value: {count_within_distance(y_true_inv, y_pred_inv, 10)[1]}%"
    )
    print(
        f"Percentage of instances with prediction within 25 of the real value: {count_within_distance(y_true_inv, y_pred_inv, 25)[1]}%"
    )
    print(
        f"Percentage of instances with prediction within 50 of the real value: {count_within_distance(y_true_inv, y_pred_inv, 50)[1]}%"
    )

    # --- Permutation Feature Importance ---
    print("\nPermutation Feature Importance (by drop in R²):")
    base_r2 = r2_score(y_true, y_pred)
    importances = []
    for col in X_val.columns:
        X_val_permuted = X_val.copy()
        X_val_permuted[col] = np.random.permutation(X_val_permuted[col].values)
        X_val_tensor_perm = torch.tensor(X_val_permuted.values, dtype=torch.float32)
        with torch.no_grad():
            y_pred_perm = model(X_val_tensor_perm).numpy().flatten()
        r2_perm = r2_score(y_true, y_pred_perm)
        importances.append(base_r2 - r2_perm)
        print(f"{col:30s}: {base_r2 - r2_perm:.4f}")

    # Plot feature importances
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(np.array(X_val.columns)[sorted_idx], np.array(importances)[sorted_idx])
    plt.xticks(rotation=90)
    plt.ylabel("Drop in R² when permuted")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(FIG_PATH + "permutation_feature_importance.pdf")
    plt.show()

# --- Partial Dependence Plot for Price ---
if PRICESENS:
    n_baselines = 100  # Number of random baseline samples
    price_idx = list(X_val.columns).index("price")
    prices = np.linspace(
        X_val["price"].min(),
        (90 - feature_means[price_idx]) / feature_stds[price_idx],
        100,
    )

    train_types = ["AVE", "IRYO", "OUIGO"]
    colors = ["tab:blue", "tab:green", "tab:red"]
    plt.figure(figsize=(9, 6))

    for train_type, color in zip(train_types, colors):
        col_name = f"train_type_{train_type}"
        if col_name not in X_val.columns:
            print(f"Column {col_name} not found in X_val. Skipping.")
            continue
        train_type_idx = list(X_val.columns).index(col_name)
        standardized_1 = (1 - feature_means[train_type_idx]) / feature_stds[
            train_type_idx
        ]
        mask = np.isclose(X_val[col_name], standardized_1)
        eligible_baselines = X_val[mask]

        if len(eligible_baselines) < n_baselines:
            print(
                f"Warning: Only {len(eligible_baselines)} baselines found with {col_name}=1. Using all available."
            )
            n_baselines_eff = len(eligible_baselines)
        else:
            n_baselines_eff = n_baselines

        if n_baselines_eff == 0:
            print(f"No baselines found for {col_name}=1. Skipping.")
            continue

        baseline_indices = np.random.choice(
            eligible_baselines.shape[0], n_baselines_eff, replace=False
        )
        baselines = eligible_baselines.iloc[baseline_indices]

        all_responses = []
        for _, baseline_row in baselines.iterrows():
            baseline = baseline_row.copy()
            responses = []
            for p in prices:
                sample = baseline.copy()
                sample["price"] = p
                sample_tensor = torch.tensor(
                    sample.values.reshape(1, -1), dtype=torch.float32
                )
                with torch.no_grad():
                    pred = model(sample_tensor).numpy()[0, 0]
                pred = pred * target_std + target_mean
                responses.append(pred)
            all_responses.append(responses)
        all_responses = np.array(all_responses)
        mean_response = all_responses.mean(axis=0)
        std_response = all_responses.std(axis=0)

        plt.plot(
            prices * feature_stds[price_idx] + feature_means[price_idx],
            mean_response,
            label=f"{train_type}",
            color=color,
        )
        plt.fill_between(
            prices * feature_stds[price_idx] + feature_means[price_idx],
            mean_response - std_response,
            mean_response + std_response,
            color=color,
            alpha=0.15,
        )

    plt.xlabel("Price")
    plt.ylabel("Predicted Passengers")
    plt.title(
        "Partial Dependence of Passengers on Price\n(baselines with train_type=1)"
    )
    plt.legend(title="Train Type")
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIG_PATH + "partial_dependence_price.pdf")
    plt.show()
