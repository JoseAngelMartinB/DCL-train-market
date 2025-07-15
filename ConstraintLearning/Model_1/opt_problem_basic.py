import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gurobipy as gp

# --- Add project root to sys.path for imports ---
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from ConstraintLearning.utils import *
import pickle

# --- Load trained model and scalers ---
SAVED_MODEL_PATH = "../saved_models/demand_ffnn_model.pt"
checkpoint = torch.load(SAVED_MODEL_PATH, map_location='cpu', weights_only=False)
feature_means = checkpoint['feature_means']
feature_stds = checkpoint['feature_stds']
feature_mins = checkpoint['feature_mins']
feature_maxs = checkpoint['feature_maxs']
target_mean = checkpoint['target_mean']
target_std = checkpoint['target_std']


# Rebuild model architecture (match train script)
input_size = len(feature_means)
output_size = 1
HIDDEN_LAYERS = [256]  # Must match training
DROPOUT = 0.1
model = FeedForwardNN(input_size, output_size, HIDDEN_LAYERS, DROPOUT)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# --- Ask user for context day
# # Load both DataFrames
orig_df = pd.read_csv('../../DataGenerationROBIN/data/MAD-BCN/aggregated/MAD-BCN_2025.csv')
cleaned_df = pd.read_csv('../preprocesed_data/cleaned_MAD-BCN_2025.csv')

# Extract date from service_id in the original DataFrame
def extract_date(service_id):
    # Example: '00003_01-01-2025-06.27'
    parts = service_id.split('_')
    date_part = parts[1].split('-')
    # date_part: ['01', '01', '2025', '06.27']
    return f"{date_part[2]}-{date_part[1]}-{date_part[0]}"  # YYYY-MM-DD

orig_df['date'] = orig_df['service_id'].apply(extract_date)

# Add the date column to the cleaned DataFrame (same order)
cleaned_df['date'] = orig_df['date']
day = '2025-01-22'
total_passengers = cleaned_df[cleaned_df['date'] == day]['passengers'].sum()
print(f"Total expected passengers for {day}: {total_passengers}")
day_context_matrix = cleaned_df[cleaned_df['date'] == day].drop(columns=['passengers', 'date'], errors='ignore').to_numpy()
cleaned_df.drop(columns=['passengers', 'date'], errors='ignore', inplace=True)

feature_names = list(cleaned_df.columns)
n_trains_context = day_context_matrix.shape[0]
PRICE_COMP_M2_IDX = feature_names.index('price_competitor_-2')
PRICE_COMP_M1_IDX = feature_names.index('price_competitor_-1')
PRICE_COMP_P1_IDX = feature_names.index('price_competitor_1')
PRICE_COMP_P2_IDX = feature_names.index('price_competitor_2')
price_idx = feature_names.index('price')

# --- Extract 'capacity' from original data ---
capacity_values = orig_df['capacity'][orig_df['date'] == day].to_numpy()
print("Capacity values from original data:", capacity_values)


# --- Set up Gurobi model for all trains in the day ---
opt_m = gp.Model("PricingOptimization")
opt_m.setParam('OutputFlag', 1)

layers = getattr(model, 'layers')

# --- Create all price variables first ---
price_vars = []
for train_idx in range(n_trains_context):
    context = day_context_matrix[train_idx]
    train_type_AVE = context[feature_names.index('train_type_AVE')]
    train_type_AVLO = context[feature_names.index('train_type_AVLO')]
    
    if train_type_AVE == 1 or train_type_AVLO == 1:
        price_var = opt_m.addVar(lb=context[price_idx]*0.75, #feature_mins[price_idx], 
                                ub=context[price_idx]*1.25, #feature_maxs[price_idx], 
                                name=f"price_var_{train_idx}")
    else:
        price_var = opt_m.addVar(lb=context[price_idx], 
                                ub=context[price_idx], 
                                name=f"price_fixed_{train_idx}")
    price_vars.append(price_var)

opt_m.update()

# --- Now start the main loop for neural network embedding ---
output_vars = []
s_aux_vars = []
RealDemands = []

for train_idx in range(n_trains_context):
    context = day_context_matrix[train_idx]
    capacity_value = capacity_values[train_idx]
    train_type_AVE = context[feature_names.index('train_type_AVE')]
    train_type_AVLO = context[feature_names.index('train_type_AVLO')]
    
    print(f"Train {train_idx}: train_type_AVE={train_type_AVE}, train_type_AVLO={train_type_AVLO}, price={context[price_idx]}, capacity={capacity_value}")

    # --- Build NN as MILP for this train ---
    x = {}
    z = {}
    x[0] = {}
    z[0] = {}

    x_maxs = {}
    x_mins = {}
    x_maxs[0] = {}
    x_mins[0] = {}

    l_0 = layers[0]
    for i in range(l_0.in_features):
        if i == price_idx:
            if train_type_AVE == 1 or train_type_AVLO == 1:
                x[0][i] = (price_var - feature_means[i]) / feature_stds[i]
                x_maxs[0][i] = (feature_maxs[i] - feature_means[i]) / feature_stds[i]
                x_mins[0][i] = (feature_mins[i] - feature_means[i]) / feature_stds[i]
            else:
                x[0][i] = (context[i] - feature_means[i]) / feature_stds[i]
                x_maxs[0][i] = (context[i] - feature_means[i]) / feature_stds[i]
                x_mins[0][i] = (context[i] - feature_means[i]) / feature_stds[i]
        elif i in [PRICE_COMP_M2_IDX, PRICE_COMP_M1_IDX, PRICE_COMP_P1_IDX, PRICE_COMP_P2_IDX]:
            # Handle edge cases for competitor prices
            if i == PRICE_COMP_M2_IDX:
                if train_idx < 2:  # First two trains
                    x[0][i] = (context[i] - feature_means[i]) / feature_stds[i]
                else:
                    prev_price = price_vars[train_idx - 2]
                    x[0][i] = (prev_price - feature_means[i]) / feature_stds[i]
            elif i == PRICE_COMP_M1_IDX:
                if train_idx < 1:  # First train
                    x[0][i] = (context[i] - feature_means[i]) / feature_stds[i]
                else:
                    prev_price = price_vars[train_idx - 1]
                    x[0][i] = (prev_price - feature_means[i]) / feature_stds[i]
            elif i == PRICE_COMP_P1_IDX:
                if train_idx >= n_trains_context - 1:  # Last train
                    x[0][i] = (context[i] - feature_means[i]) / feature_stds[i]
                else:
                    next_price = price_vars[train_idx + 1]
                    x[0][i] = (next_price - feature_means[i]) / feature_stds[i]
            else:  # PRICE_COMP_P2_IDX
                if train_idx >= n_trains_context - 2:  # Last two trains
                    x[0][i] = (context[i] - feature_means[i]) / feature_stds[i]
                else:
                    next_price = price_vars[train_idx + 2]
                    x[0][i] = (next_price - feature_means[i]) / feature_stds[i]
            # Set bounds for all competitor price features
            x_maxs[0][i] = (feature_maxs[i] - feature_means[i]) / feature_stds[i]
            x_mins[0][i] = (feature_mins[i] - feature_means[i]) / feature_stds[i]
        else:
            x[0][i] = (context[i] - feature_means[i]) / feature_stds[i]
            x_maxs[0][i] = (context[i] - feature_means[i]) / feature_stds[i]
            x_mins[0][i] = (context[i] - feature_means[i]) / feature_stds[i]
    opt_m.update()

    for ind, layer in enumerate(layers):
        l = layer
        x[ind+1] = {}
        z[ind+1] = {}
        x_maxs[ind+1] = {}
        x_mins[ind+1] = {}

        for i in range(l.out_features):
            m = l.weight.detach().numpy()[i]
            b = l.bias.detach().numpy()[i]

            if ind < len(layers) - 1:
                ub = sum(x_maxs[ind][j] * max(0, m[j]) + x_mins[ind][j] * min(0, m[j]) for j in range(l.in_features)) + b
                lb = sum(x_mins[ind][j] * max(0, m[j]) + x_maxs[ind][j] * min(0, m[j]) for j in range(l.in_features)) + b
                x_maxs[ind+1][i] = ub
                x_mins[ind+1][i] = lb

                x[ind+1][i] = opt_m.addVar(0, max(0, ub), name=f'x_{ind+1}_{i}_train{train_idx}')
                z[ind+1][i] = opt_m.addVar(0, 1, vtype=gp.GRB.BINARY, name=f'z_{ind+1}_{i}_train{train_idx}')
                opt_m.addConstr(x[ind+1][i] >= sum(x[ind][j] * m[j] for j in range(l.in_features)) + b)
                opt_m.addConstr(x[ind+1][i] <= sum(x[ind][j] * m[j] for j in range(l.in_features)) + b - lb * (1 - z[ind+1][i]))
                opt_m.addConstr(x[ind+1][i] <= ub * z[ind+1][i])
            else:
                x[ind+1][i] = opt_m.addVar(lb=-gp.GRB.INFINITY, name=f'x_{ind+1}_{i}_train{train_idx}')
                opt_m.addConstr(x[ind+1][i] == (sum(x[ind][j] * m[j] for j in range(l.in_features)) + b) * target_std + target_mean)
                output_var = x[ind+1][i]
        opt_m.update()

    # --- RealDemand for this train ---
    cap = float(capacity_value)
    M = cap * 1000

    # First, handle max(0, output_var)
    s_aux = opt_m.addVar(lb=0, name=f"output_var_nonneg_{train_idx}")
    bin1_aux = opt_m.addVar(vtype=gp.GRB.BINARY, name=f"bin_output_nonneg1_{train_idx}")
    bin2_aux = opt_m.addVar(vtype=gp.GRB.BINARY, name=f"bin_output_nonneg2_{train_idx}")
    opt_m.addConstr(s_aux >= 0, name=f"output_var_nonneg_ge_zero_{train_idx}")
    opt_m.addConstr(s_aux >= output_var, name=f"output_var_nonneg_ge_output_{train_idx}")
    opt_m.addConstr(s_aux <= output_var + M * (1 - bin1_aux), name="output_var_nonneg_le_output_plus_M")
    opt_m.addConstr(s_aux <= M * bin1_aux, name=f"output_var_nonneg_le_M_{train_idx}")
    opt_m.addConstr(output_var <= M * bin1_aux, name=f"output_var_le_M_{train_idx}")
    opt_m.addConstr(output_var >= -M * (1 - bin1_aux), name=f"output_var_ge_minus_M_{train_idx}")
    # RealDemand logic with strict enforcement
    RealDemand = opt_m.addVar(lb=0, ub=cap, name=f"RealDemand_{train_idx}")
    opt_m.addConstr(RealDemand <= s_aux, name=f"RealDemand_ge_output_{train_idx}")
    opt_m.addConstr(RealDemand >= s_aux - M * (1 - bin2_aux), name=f"RealDemand_le_output_{train_idx}")
    opt_m.addConstr(RealDemand >= cap - M * bin2_aux, name=f"RealDemand_le_cap_minus_M_{train_idx}")

    output_vars.append(output_var)
    s_aux_vars.append(s_aux)
    RealDemands.append(RealDemand)
    opt_m.update()

#opt_m.addConstr(gp.quicksum(s_aux_vars) >= 0.75 * total_passengers, name="min_total_demand")
#opt_m.addConstr(gp.quicksum(s_aux_vars) <= 1.25 * total_passengers, name="max_total_demand")
opt_m.update()

# --- Objective: maximize sum of all RealDemand * price_var for AVE/AVLO only ---
total_revenue = gp.LinExpr()
for i in range(n_trains_context):
    train_type_AVE = day_context_matrix[i][feature_names.index('train_type_AVE')]
    train_type_AVLO = day_context_matrix[i][feature_names.index('train_type_AVLO')]
    if train_type_AVE == 1 or train_type_AVLO == 1:
        total_revenue += price_vars[i] * RealDemands[i]
opt_m.setObjective(total_revenue, gp.GRB.MAXIMIZE)
opt_m.update()
opt_m.setParam('MIPGap', 0.01)  # Set a small MIP gap for faster convergence
opt_m.setParam('MIPFocus', 3)  # Focus on improving the best bound
opt_m.optimize()

# --- Print solution ---
if opt_m.status == gp.GRB.OPTIMAL:
    print("Optimal solution found:")
    for train_idx in range(n_trains_context):
        print(f"Train {train_idx}:")
        print(f"  Optimal price: {price_vars[train_idx].X}")
        print(f"  Predicted demand: {output_vars[train_idx].X}")
        print(f"  RealDemand (min(capacity, output)): {RealDemands[train_idx].X}")
elif opt_m.status == gp.GRB.INTERRUPTED:
    print("Optimization was interrupted.")
    for train_idx in range(n_trains_context):
        print(f"Train {train_idx}:")
        print(f"  Optimal price: {price_vars[train_idx].X}")
        print(f"  Predicted demand: {output_vars[train_idx].X}")
        print(f"  RealDemand (min(capacity, output)): {RealDemands[train_idx].X}")
else:
    print("No optimal solution found.")
