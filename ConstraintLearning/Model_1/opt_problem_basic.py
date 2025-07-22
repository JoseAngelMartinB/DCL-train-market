import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gurobipy as gp

# Add project root to sys.path for imports
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from ConstraintLearning.utils import *
import pickle

# Get project root (same pattern as your sys.path addition)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


# --- Config ---
SAVED_MODEL_PATH = os.path.join(project_root, "ConstraintLearning/saved_models/demand_ffnn_model.pt")
RENFE_PRICES_INTERVAL = [10, 160]
DAY = '2025-03-12' # Weekday low demand day
#DAY = '2025-03-22' # Weekend low demand day
#DAY = '2025-08-13' # Weekday low demand day
#DAY = '2025-08-23' # Weekend low demand day

# Check if file exists
if not os.path.exists(SAVED_MODEL_PATH):
    print(f"Model file not found at: {SAVED_MODEL_PATH}")
    print("Available files in the directory:")
    model_dir = os.path.dirname(SAVED_MODEL_PATH)
    if os.path.exists(model_dir):
        print(os.listdir(model_dir))
    else:
        print(f"Directory {model_dir} does not exist")
    sys.exit(1)

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
HIDDEN_LAYERS = [50,50]  # Must match training
DROPOUT = 0.1
model = FeedForwardNN(input_size, output_size, HIDDEN_LAYERS, DROPOUT)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# --- Ask user for context day
# # Load both DataFrames
orig_csv_path = os.path.join(project_root, 'DataGenerationROBIN/data/MAD-BCN/aggregated/MAD-BCN_2025.csv')
cleaned_csv_path = os.path.join(project_root, 'ConstraintLearning/preprocesed_data/cleaned_MAD-BCN_2025.csv')

# Check if files exist
if not os.path.exists(orig_csv_path):
    print(f"Original CSV file not found at: {orig_csv_path}")
    print(f"Project root: {project_root}")
    sys.exit(1)

if not os.path.exists(cleaned_csv_path):
    print(f"Cleaned CSV file not found at: {cleaned_csv_path}")
    print(f"Project root: {project_root}")
    sys.exit(1)

orig_df = pd.read_csv(orig_csv_path)
cleaned_df = pd.read_csv(cleaned_csv_path)

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
day = DAY
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
        price_var = opt_m.addVar(lb=max(context[price_idx] - 5, RENFE_PRICES_INTERVAL[0]), #*0.75, #feature_mins[price_idx], 
                                ub=min(context[price_idx] + 5, RENFE_PRICES_INTERVAL[1]),  #*1.25, #feature_maxs[price_idx], 
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

opt_m.addConstr(gp.quicksum(s_aux_vars) >= 0.75 * total_passengers, name="min_total_demand")
opt_m.addConstr(gp.quicksum(s_aux_vars) <= 1.25 * total_passengers, name="max_total_demand")
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


# --- Save results to CSV ---
if opt_m.status in [gp.GRB.OPTIMAL, gp.GRB.INTERRUPTED]:
    # Get the objective value
    if opt_m.status == gp.GRB.OPTIMAL:
        objective_value = opt_m.objVal
    else:
        objective_value = opt_m.objBound  # Use best bound if interrupted
    
    # Get the service_ids for the selected day
    day_service_ids = orig_df[orig_df['date'] == day]['service_id'].tolist()
    
    # Prepare data for CSV
    results_data = []
    for train_idx in range(n_trains_context):
        # Get service_id as the train identifier
        service_id = day_service_ids[train_idx]
        
        # Get original price
        original_price = day_context_matrix[train_idx][price_idx]
        
        # Get train type info
        train_type_AVE = day_context_matrix[train_idx][feature_names.index('train_type_AVE')]
        train_type_AVLO = day_context_matrix[train_idx][feature_names.index('train_type_AVLO')]
        
        # Determine train type string
        if train_type_AVE == 1:
            train_type = "AVE"
        elif train_type_AVLO == 1:
            train_type = "AVLO"
        else:
            # Check for IRYO and OUIGO
            try:
                train_type_IRYO = day_context_matrix[train_idx][feature_names.index('train_type_IRYO')]
                if train_type_IRYO == 1:
                    train_type = "IRYO"
                else:
                    train_type_OUIGO = day_context_matrix[train_idx][feature_names.index('train_type_OUIGO')]
                    if train_type_OUIGO == 1:
                        train_type = "OUIGO"
                    else:
                        train_type = "Other"
            except ValueError:
                # If IRYO/OUIGO columns don't exist, use "Other"
                train_type = "Other"

        
        results_data.append({
            'train_idx': service_id,  # Use service_id as train_idx
            'train_type': train_type,
            'original_price': original_price,
            'optimized_price': price_vars[train_idx].X,
            'difference': price_vars[train_idx].X - original_price,
            'predicted_demand': output_vars[train_idx].X,
            'real_demand': RealDemands[train_idx].X,
            'capacity': capacity_values[train_idx]
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    
    # Create filename with date and objective value
    objective_str = f"{objective_value:.2f}".replace('.', '_')
    filename = f"results_{day}_obj_{objective_str}.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    # Save to CSV
    results_df.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")
    print(f"Total revenue (objective): {objective_value:.2f}")
else:
    print("Cannot save results - optimization failed.")
