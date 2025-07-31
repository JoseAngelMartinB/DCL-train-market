import os
import psutil
import threading
import time
import sys
import numpy as np
import pandas as pd
import joblib
import gurobipy as gp

# Add project root to sys.path for imports
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from ConstraintLearning.utils import *

process = psutil.Process(os.getpid())
peak_ram_mb = 0

def monitor_ram(process, interval=0.5):
    global peak_ram_mb
    peak_ram_mb = 0
    while monitoring_flag:
        mem = process.memory_info().rss / (1024 * 1024)
        if mem > peak_ram_mb:
            peak_ram_mb = mem
        time.sleep(interval)

# Get project root (same pattern as your sys.path addition)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# --- Config ---
ML_MODEL_NAME = 'rf'
SAVED_MODEL_PATH = os.path.join(project_root, f"ConstraintLearning/saved_models/demand_{ML_MODEL_NAME}_model.pkl")
RESULTS_PATH = os.path.join(project_root, f'ConstraintLearning/Model_1/results_{ML_MODEL_NAME}/')
CLEAR_PREVIOUS_RESULTS = True  # Set to True to clear previous results
OPTIM_RESULTS_PATH = os.path.join(project_root, 'ConstraintLearning/Model_1/opt_results.csv')
RENFE_PRICES_INTERVAL = [10, 160]
TIME_LIMIT = 3 * 3600  # Time limit for optimization
DISPLAY_LIMIT = 5  # Limit for displaying train prices on console

# Define different scenarios to test (start with just one scenario for testing)
DELTA_VALUES = [5,10]  # Start with just one delta
DAYS = [
  #  '2025-03-12',  # Weekday low demand day
    '2025-03-22',  # Weekend low demand day
 #   '2025-08-13',  # Weekday high demand day
#    '2025-08-23'   # Weekend high demand day
]

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

# Load previous optimization results if available
if os.path.exists(OPTIM_RESULTS_PATH):
    optim_results_df = pd.read_csv(OPTIM_RESULTS_PATH)
    print(f"Loaded previous optimization results from: {OPTIM_RESULTS_PATH}")
else:
    optim_results_df = pd.DataFrame(columns=[
        'filename', 'ml_model', 'day', 'delta', 'total_revenue', 'gap', 'n_vars',
        'n_bin_vars', 'n_cont_vars', 'n_constrs', 'peak_ram_usage_MB',
        'status', 'time_seconds', 'executed_on', 'device'
    ])

# Create results directory if it doesn't exist
os.makedirs(RESULTS_PATH, exist_ok=True)
if CLEAR_PREVIOUS_RESULTS and os.path.exists(RESULTS_PATH):
    print(f"Clearing previous results in: {RESULTS_PATH}")
    # Remove all the csv files in the results directory
    for file in os.listdir(RESULTS_PATH):
        if file.endswith('.csv'):
            file_path = os.path.join(RESULTS_PATH, file)
            os.remove(file_path)
    optim_results_df = optim_results_df[optim_results_df['ml_model'] != ML_MODEL_NAME]


# Load the Neural Network model
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
model = FeedForwardNN(input_size, output_size, HIDDEN_LAYERS, DROPOUT)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
layers = getattr(model, 'layers')

# --- Load both DataFrames ---
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
cleaned_df['date'] = orig_df['date']


# Main optimization loop
print("\n" + "="*60)
print("STARTING FEED-FORWARD NEURAL NETWORK (FFNN) OPTIMIZATION")
print("="*60)

for day in DAYS:
    for delta in DELTA_VALUES:
        print(f"\n{'='*50}")
        print(f"OPTIMIZING: Day={day}, Delta=±{delta}")
        print(f"{'='*50}")
        
        # Prepare data for this day
        total_passengers = cleaned_df[cleaned_df['date'] == day]['passengers'].sum()
        print(f"Total expected passengers for {day}: {total_passengers}")
        
        # Get day-specific data
        day_context_matrix = cleaned_df[cleaned_df['date'] == day].drop(columns=['passengers', 'date'], errors='ignore').to_numpy()
        day_cleaned_df = cleaned_df.drop(columns=['passengers', 'date'], errors='ignore')
        
        feature_names = list(day_cleaned_df.columns)
        n_trains_context = day_context_matrix.shape[0]
        PRICE_COMP_M2_IDX = feature_names.index('price_competitor_-2')
        PRICE_COMP_M1_IDX = feature_names.index('price_competitor_-1')
        PRICE_COMP_P1_IDX = feature_names.index('price_competitor_1')
        PRICE_COMP_P2_IDX = feature_names.index('price_competitor_2')
        price_idx = feature_names.index('price')
        
        # Extract 'capacity' from original data
        capacity_values = orig_df['capacity'][orig_df['date'] == day].to_numpy()
        print("Capacity values from original data:", capacity_values[:5], "...")  # Show first 5

        # --- Set up Gurobi model for all trains in the day ---
        opt_m = gp.Model("FeedForwardNNOptimization")
        
        # Optimize Gurobi parameters for large MILPs
        opt_m.setParam('OutputFlag', 1)
        opt_m.setParam('Threads', 0)  # Use all available cores (0 = automatic)
        opt_m.setParam('MIPFocus', 3)  # Focus on improving bounds
        opt_m.setParam('Cuts', 2)  # Aggressive cuts
        opt_m.setParam('Presolve', 2)  # Aggressive presolve
        opt_m.setParam('Heuristics', 0.1)  # Spend 10% time on heuristics
        opt_m.setParam('NodefileStart', 12.0)  # Start writing node file after 12 GB

        # --- Create all price variables first (batch creation) ---
        price_vars = []
        price_bounds = []
        
        for train_idx in range(n_trains_context):
            context = day_context_matrix[train_idx]
            train_type_AVE = context[feature_names.index('train_type_AVE')]
            train_type_AVLO = context[feature_names.index('train_type_AVLO')]
            
            if train_type_AVE == 1 or train_type_AVLO == 1:
                lb = max(context[price_idx] - delta, RENFE_PRICES_INTERVAL[0])
                ub = min(context[price_idx] + delta, RENFE_PRICES_INTERVAL[1])
            else:
                lb = ub = context[price_idx]
            
            price_bounds.append((lb, ub))
        
        # Batch create all price variables
        price_vars = opt_m.addVars(
            n_trains_context,
            lb=[bounds[0] for bounds in price_bounds],
            ub=[bounds[1] for bounds in price_bounds],
            name="price"
        )

        opt_m.update()

        # --- Now embed the Feed-Forward Neural Network (FFNN) ---
        output_vars = []
        s_aux_vars = []
        ActualDemands = []

        for train_idx in range(n_trains_context):
            context = day_context_matrix[train_idx]
            capacity_value = capacity_values[train_idx]
            train_type_AVE = context[feature_names.index('train_type_AVE')]
            train_type_AVLO = context[feature_names.index('train_type_AVLO')]
            
            print(f"Train {train_idx}: train_type_AVE={train_type_AVE}, train_type_AVLO={train_type_AVLO}, price={context[price_idx]}, capacity={capacity_value}")

            # --- Build NN as MILP  (Mixed Integer Linear Programming) ---
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
                        price_var = price_vars[train_idx]
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

            # --- ActualDemand for this train ---
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
            # ActualDemand logic with strict enforcement
            ActualDemand = opt_m.addVar(lb=0, ub=cap, name=f"ActualDemand_{train_idx}")
            opt_m.addConstr(ActualDemand <= s_aux, name=f"ActualDemand_ge_output_{train_idx}")
            opt_m.addConstr(ActualDemand >= s_aux - M * (1 - bin2_aux), name=f"ActualDemand_le_output_{train_idx}")
            opt_m.addConstr(ActualDemand >= cap - M * bin2_aux, name=f"ActualDemand_le_cap_minus_M_{train_idx}")

            output_vars.append(output_var)
            s_aux_vars.append(s_aux)
            ActualDemands.append(ActualDemand)
            
            opt_m.update()

        # --- Add total demand constraints (relaxed to avoid infeasibility) ---
        opt_m.addConstr(gp.quicksum(s_aux_vars) >= 0.6 * total_passengers, name="min_total_demand")
        opt_m.addConstr(gp.quicksum(s_aux_vars) <= 1.4 * total_passengers, name="max_total_demand")

        # --- Objective: maximize revenue for AVE/AVLO trains ---
        total_revenue = gp.LinExpr()
        for i in range(n_trains_context):
            train_type_AVE = day_context_matrix[i][feature_names.index('train_type_AVE')]
            train_type_AVLO = day_context_matrix[i][feature_names.index('train_type_AVLO')]
            if train_type_AVE == 1 or train_type_AVLO == 1:
                total_revenue += price_vars[i] * ActualDemands[i]

        opt_m.setObjective(total_revenue, gp.GRB.MAXIMIZE)
        opt_m.update()

        # --- Optimization parameters for Feed-Forward Neural Network MILP ---
        opt_m.setParam('MIPGap', 0.01)  # Allow 1% optimality gap for faster solutions
        opt_m.setParam('MIPFocus', 3)   # Focus on finding good feasible solutions
        opt_m.setParam('TimeLimit', TIME_LIMIT)  # Set optimization time limit

        # Add infeasibility debugging
        #opt_m.setParam('DualReductions', 0)  # Disable dual reductions for better debugging

        print(f"Starting optimization with {opt_m.NumVars} variables and {opt_m.NumConstrs} constraints...")

        monitoring_flag = True
        monitor_thread = threading.Thread(target=monitor_ram, args=(process,))
        monitor_thread.start()

        opt_m.optimize()

        monitoring_flag = False
        monitor_thread.join()

        # Check for infeasibility and provide debugging info
        if opt_m.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible! Computing IIS...")
            opt_m.computeIIS()
            print("Irreducible Inconsistent Subsystem (IIS):")
            for c in opt_m.getConstrs():
                if c.IISConstr:
                    print(f"  Constraint: {c.constrName}")
            for v in opt_m.getVars():
                if v.IISLB:
                    print(f"  Variable LB: {v.varName} >= {v.lb}")
                if v.IISUB:
                    print(f"  Variable UB: {v.varName} <= {v.ub}")
            continue  # Skip to next scenario if infeasible

        # --- Print solution ---
        if opt_m.status == gp.GRB.OPTIMAL:
            print("Optimal solution found:")
            for train_idx in range(min(DISPLAY_LIMIT, n_trains_context)): # Display only a limited number of trains
                print(f"Train {train_idx}:")
                print(f"  Optimal price: {price_vars[train_idx].X:.2f}")
                print(f"  Predicted demand: {output_vars[train_idx].X:.2f}")
                print(f"  ActualDemand (capped): {ActualDemands[train_idx].X:.2f}")
            if n_trains_context > DISPLAY_LIMIT:
                print(f"... (showing {DISPLAY_LIMIT} out of {n_trains_context} trains)")
        elif opt_m.status == gp.GRB.INTERRUPTED:
            print("Optimization was interrupted.")
        elif opt_m.status == gp.GRB.TIME_LIMIT:
            print("Time limit reached.")
        else:
            print(f"Optimization failed with status: {opt_m.status}")

        # --- Save results to CSV ---
        if opt_m.status in [gp.GRB.OPTIMAL, gp.GRB.INTERRUPTED, gp.GRB.TIME_LIMIT]:
            try:
                # Get the objective value
                if opt_m.status == gp.GRB.OPTIMAL:
                    objective_value = opt_m.objVal
                else:
                    # For interrupted or time limit, use the best lower bound
                    objective_value = opt_m.objBound if opt_m.objBound < gp.GRB.INFINITY else 0
                
                # Get the service_ids for the selected day
                day_service_ids = orig_df[orig_df['date'] == day]['service_id'].tolist()
                
                # Prepare data for CSV
                results_data = []
                for train_idx in range(n_trains_context):
                    try:
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
                                train_type = "Other"
                        
                        results_data.append({
                            'train_idx': service_id,
                            'train_type': train_type,
                            'original_price': original_price,
                            'optimized_price': price_vars[train_idx].X,
                            'difference': price_vars[train_idx].X - original_price,
                            'predicted_demand': output_vars[train_idx].X,
                            'actual_demand': ActualDemands[train_idx].X,
                            'capacity': capacity_values[train_idx]
                        })
                    except:
                        print(f"Error processing train {train_idx}, skipping...")
                        continue
                
                # Create DataFrame
                results_df = pd.DataFrame(results_data)
                objective_str = f"{objective_value:.2f}".replace('.', '_')
                results_filename = f"{day}_delta_{delta}_obj_{objective_str}.csv"
                results_filepath = os.path.join(RESULTS_PATH, results_filename)
                
                # Save results to CSV
                results_df.to_csv(results_filepath, index=False)
                print(f"\nResults saved to: {results_filepath}")
                print(f"Total revenue (objective): {objective_value:.2f}")

                # Append to optimization results DataFrame
                if opt_m.status == gp.GRB.OPTIMAL:
                    status_str = "Optimal"
                elif opt_m.status == gp.GRB.INTERRUPTED:
                    status_str = "Interrupted"
                elif opt_m.status == gp.GRB.TIME_LIMIT:
                    status_str = "Time Limit"
                else:
                    status_str = "Other"
                new_row = {
                    'filename': results_filename,
                    'ml_model': ML_MODEL_NAME,
                    'day': day,
                    'delta': delta,
                    'total_revenue': objective_value,
                    'gap': opt_m.MIPGap,
                    'n_vars': opt_m.NumVars,
                    'n_bin_vars': opt_m.NumBinVars,
                    'n_cont_vars': opt_m.NumVars - opt_m.NumBinVars,
                    'n_constrs': opt_m.NumConstrs,
                    'peak_ram_usage_MB': peak_ram_mb,
                    'status': status_str,
                    'time_seconds': opt_m.Runtime,
                    'executed_on': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'device': os.uname().nodename,
                }
                optim_results_df = pd.concat([optim_results_df, pd.DataFrame([new_row])], ignore_index=True)
                optim_results_df.to_csv(OPTIM_RESULTS_PATH, index=False)
                print(f"Optimization results updated in: {OPTIM_RESULTS_PATH}")
                
            except Exception as e:
                print(f"Error saving results: {e}")
        else:
            print("Cannot save results - optimization failed.")

        # --- VALIDATION: Compare optimization predictions with ML model predictions ---
        PredVal = False  # Set to True to enable validation
        
        if PredVal and opt_m.status in [gp.GRB.OPTIMAL, gp.GRB.INTERRUPTED, gp.GRB.TIME_LIMIT]:
            print("\n" + "="*50)
            print("VALIDATION: Comparing Optimization vs ML Model Predictions")
            print("="*50)
            
            validation_results = []
            
            for train_idx in range(n_trains_context):
                try:
                    # Get optimized price from the optimization model
                    optimized_price = price_vars[train_idx].X
                    
                    # Get optimization model's demand prediction
                    opt_demand_prediction = output_vars[train_idx].X
                    
                    # Prepare context with optimized price for ML model prediction
                    context = day_context_matrix[train_idx].copy()
                    context[price_idx] = optimized_price
                    
                    # Handle competitor prices - same logic as in optimization
                    if train_idx >= 2:  # Update price_competitor_-2
                        context[PRICE_COMP_M2_IDX] = price_vars[train_idx - 2].X
                    if train_idx >= 1:  # Update price_competitor_-1
                        context[PRICE_COMP_M1_IDX] = price_vars[train_idx - 1].X
                    if train_idx < n_trains_context - 1:  # Update price_competitor_1
                        context[PRICE_COMP_P1_IDX] = price_vars[train_idx + 1].X
                    if train_idx < n_trains_context - 2:  # Update price_competitor_2
                        context[PRICE_COMP_P2_IDX] = price_vars[train_idx + 2].X
                    
                    # Scale features for ML model
                    scaled_context = (context - feature_means) / feature_stds
                    
                    # Get ML model prediction
                    ml_prediction_scaled = model.predict(scaled_context.reshape(1, -1))[0]
                    ml_demand_prediction = ml_prediction_scaled * target_std + target_mean
                    
                    # Calculate difference
                    difference = abs(opt_demand_prediction - ml_demand_prediction)
                    relative_error = (difference / max(ml_demand_prediction, 1e-6)) * 100
                    
                    validation_results.append({
                        'train_idx': train_idx,
                        'optimized_price': optimized_price,
                        'opt_demand_prediction': opt_demand_prediction,
                        'ml_demand_prediction': ml_demand_prediction,
                        'absolute_difference': difference,
                        'relative_error_pct': relative_error
                    })
                    
                    # Print first 5 trains for quick inspection
                    if train_idx < 5:
                        print(f"Train {train_idx}:")
                        print(f"  Optimized price: {optimized_price:.2f}")
                        print(f"  Opt model demand: {opt_demand_prediction:.2f}")
                        print(f"  ML model demand:  {ml_demand_prediction:.2f}")
                        print(f"  Absolute diff:    {difference:.2f}")
                        print(f"  Relative error:   {relative_error:.2f}%")
                        print()
                        
                except Exception as e:
                    print(f"Error validating train {train_idx}: {e}")
                    continue
            
            # Calculate validation statistics
            if validation_results:
                validation_df = pd.DataFrame(validation_results)
                
                mean_abs_diff = validation_df['absolute_difference'].mean()
                max_abs_diff = validation_df['absolute_difference'].max()
                mean_rel_error = validation_df['relative_error_pct'].mean()
                max_rel_error = validation_df['relative_error_pct'].max()
                
                print(f"VALIDATION SUMMARY:")
                print(f"  Mean absolute difference: {mean_abs_diff:.3f}")
                print(f"  Max absolute difference:  {max_abs_diff:.3f}")
                print(f"  Mean relative error:      {mean_rel_error:.2f}%")
                print(f"  Max relative error:       {max_rel_error:.2f}%")
                
                # Save validation results
                if opt_m.status == gp.GRB.OPTIMAL:
                    objective_value = opt_m.objVal
                else:
                    objective_value = opt_m.objBound if opt_m.objBound < gp.GRB.INFINITY else 0
                
                objective_str = f"{objective_value:.2f}".replace('.', '_')
                validation_filename = f"validation_ffnn_{day}_delta_{delta}_obj_{objective_str}.csv"
                validation_filepath = os.path.join(results_dir, validation_filename)
                
                validation_df.to_csv(validation_filepath, index=False)
                print(f"  Validation results saved to: {validation_filepath}")
                
                # Check if validation is acceptable (you can adjust these thresholds)
                if mean_rel_error < 5.0:
                    print("  ✅ VALIDATION PASSED: Optimization model predictions match ML model well!")
                else:
                    print("  ⚠️  VALIDATION WARNING: Large differences detected between optimization and ML predictions!")
            else:
                print("  ❌ VALIDATION FAILED: No valid predictions to compare!")

print("\n" + "="*60)
print("ALL FEED-FORWARD NEURAL NETWORK (FFNN) SCENARIOS COMPLETED!")
print("="*60)
