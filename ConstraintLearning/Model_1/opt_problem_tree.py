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

def monitor_ram(process, interval=0.1):
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
ML_MODEL_NAME = 'tree'
SAVED_MODEL_PATH = os.path.join(project_root, f"ConstraintLearning/saved_models/demand_{ML_MODEL_NAME}_model.pkl")
RESULTS_PATH = os.path.join(project_root, f'ConstraintLearning/Model_1/results_{ML_MODEL_NAME}/')
CLEAR_PREVIOUS_RESULTS = True  # Set to True to clear previous results
OPTIM_RESULTS_PATH = os.path.join(project_root, 'ConstraintLearning/Model_1/opt_results.csv')
RENFE_PRICES_INTERVAL = [10, 160]
TIME_LIMIT = 1 * 3600  # Time limit for optimization
DISPLAY_LIMIT = 5  # Limit for displaying train prices on console

# Define different scenarios to test
DELTA_VALUES = [5, 10, 20]
DAYS = [
    '2025-03-12',  # Weekday low demand day
    '2025-03-22',  # Weekend low demand day
    '2025-08-13',  # Weekday high demand day
    '2025-08-23'   # Weekend high demand day
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


# Load the tree model and scalers
model_data = joblib.load(SAVED_MODEL_PATH)
tree_model = model_data['model']
feature_scaler = model_data['feature_scaler']
target_scaler = model_data['target_scaler']
feature_means = model_data['feature_means']
feature_stds = model_data['feature_stds']
feature_mins = model_data['feature_mins']
feature_maxs = model_data['feature_maxs']
target_mean = model_data['target_mean']
target_std = model_data['target_std']

print(f"Loaded tree model with max_depth: {tree_model.max_depth}")
print(f"Tree has {tree_model.tree_.node_count} nodes")

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

# --- Helper functions to encode decision tree in MILP ---
def get_tree_structure(tree):
    """Extract tree structure for MILP encoding"""
    tree_ = tree.tree_
    
    # Get all nodes information
    nodes_info = []
    for node_id in range(tree_.node_count):
        if tree_.children_left[node_id] != tree_.children_right[node_id]:  # Internal node
            nodes_info.append({
                'node_id': node_id,
                'is_leaf': False,
                'feature': tree_.feature[node_id],
                'threshold': tree_.threshold[node_id],
                'left_child': tree_.children_left[node_id],
                'right_child': tree_.children_right[node_id]
            })
        else:  # Leaf node
            nodes_info.append({
                'node_id': node_id,
                'is_leaf': True,
                'value': tree_.value[node_id][0][0]  # Regression value
            })
    
    return nodes_info

def add_tree_constraints(opt_model, tree_model, scaled_features, train_idx):
    """Add decision tree constraints to the optimization model"""
    tree_info = get_tree_structure(tree_model)
    n_nodes = len(tree_info)
    
    # Create binary variables for each node (1 if the sample passes through this node)
    node_vars = {}
    for node in tree_info:
        node_vars[node['node_id']] = opt_model.addVar(
            vtype=gp.GRB.BINARY, 
            name=f"node_{node['node_id']}_train_{train_idx}"
        )
    
    opt_model.update()
    
    # Root node must be active
    opt_model.addConstr(node_vars[0] == 1, name=f"root_active_train_{train_idx}")
    
    # Add constraints for internal nodes
    for node in tree_info:
        if not node['is_leaf']:
            node_id = node['node_id']
            feature_idx = node['feature']
            threshold = node['threshold']
            left_child = node['left_child']
            right_child = node['right_child']
            
            # If current node is active, exactly one child must be active
            opt_model.addConstr(
                node_vars[left_child] + node_vars[right_child] == node_vars[node_id],
                name=f"split_consistency_{node_id}_train_{train_idx}"
            )
            
            # Big M constraint for left child (feature <= threshold)
            M = 10
            left_expr = scaled_features[feature_idx] - threshold - M * (1 - node_vars[left_child])
            opt_model.addConstr(
                left_expr <= 0,
                name=f"left_split_{node_id}_train_{train_idx}"
            )
            
            # Big M constraint for right child (feature > threshold)
            right_expr = scaled_features[feature_idx] - threshold + M * (1 - node_vars[right_child])
            opt_model.addConstr(
                right_expr >= 1e-6,
                name=f"right_split_{node_id}_train_{train_idx}"
            )
    
    # Create output variable as weighted sum of leaf values
    output_var = opt_model.addVar(
        lb=-gp.GRB.INFINITY, 
        name=f"tree_output_train_{train_idx}"
    )
    
    # Output is the sum of (leaf_value * leaf_node_binary) for all leaves
    leaf_expr = gp.LinExpr()
    for node in tree_info:
        if node['is_leaf']:
            # Scale back from standardized target to original scale
            leaf_value_original = node['value'] * target_std + target_mean
            leaf_expr += leaf_value_original * node_vars[node['node_id']]
    
    opt_model.addConstr(output_var == leaf_expr, name=f"tree_output_def_train_{train_idx}")
    
    # Exactly one leaf must be active
    leaf_sum = gp.quicksum(node_vars[node['node_id']] for node in tree_info if node['is_leaf'])
    opt_model.addConstr(leaf_sum == 1, name=f"one_leaf_active_train_{train_idx}")
    
    return output_var

# Main optimization loop
print("\n" + "="*60)
print("STARTING OPTIMIZATION FOR ALL SCENARIOS")
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
        opt_m = gp.Model("TreePricingOptimization")
        opt_m.setParam('OutputFlag', 1)

        # --- Create all price variables first ---
        price_vars = []
        for train_idx in range(n_trains_context):
            context = day_context_matrix[train_idx]
            train_type_AVE = context[feature_names.index('train_type_AVE')]
            train_type_AVLO = context[feature_names.index('train_type_AVLO')]
            
            if train_type_AVE == 1 or train_type_AVLO == 1:
                price_var = opt_m.addVar(
                    lb=max(context[price_idx] - delta, RENFE_PRICES_INTERVAL[0]),
                    ub=min(context[price_idx] + delta, RENFE_PRICES_INTERVAL[1]),
                    name=f"price_var_{train_idx}"
                )
            else:
                price_var = opt_m.addVar(
                    lb=context[price_idx], 
                    ub=context[price_idx], 
                    name=f"price_fixed_{train_idx}"
                )
            price_vars.append(price_var)

        opt_m.update()

        # --- Now embed the decision tree for each train ---
        output_vars = []
        s_aux_vars = []
        ActualDemands = []

        for train_idx in range(n_trains_context):
            context = day_context_matrix[train_idx]
            capacity_value = capacity_values[train_idx]
            train_type_AVE = context[feature_names.index('train_type_AVE')]
            train_type_AVLO = context[feature_names.index('train_type_AVLO')]
            
            print(f"Train {train_idx}: train_type_AVE={train_type_AVE}, train_type_AVLO={train_type_AVLO}, price={context[price_idx]}, capacity={capacity_value}")

            # --- Prepare scaled features for the tree ---
            scaled_features = {}
            
            for i in range(len(feature_names)):
                if i == price_idx:
                    if train_type_AVE == 1 or train_type_AVLO == 1:
                        # Use price variable and scale it
                        scaled_features[i] = (price_vars[train_idx] - feature_means[i]) / feature_stds[i]
                    else:
                        # Use fixed price and scale it
                        scaled_features[i] = (context[i] - feature_means[i]) / feature_stds[i]
                elif i in [PRICE_COMP_M2_IDX, PRICE_COMP_M1_IDX, PRICE_COMP_P1_IDX, PRICE_COMP_P2_IDX]:
                    # Handle competitor prices (same logic as NN version)
                    if i == PRICE_COMP_M2_IDX:
                        if train_idx < 2:  # First two trains
                            scaled_features[i] = (context[i] - feature_means[i]) / feature_stds[i]
                        else:
                            scaled_features[i] = (price_vars[train_idx - 2] - feature_means[i]) / feature_stds[i]
                    elif i == PRICE_COMP_M1_IDX:
                        if train_idx < 1:  # First train
                            scaled_features[i] = (context[i] - feature_means[i]) / feature_stds[i]
                        else:
                            scaled_features[i] = (price_vars[train_idx - 1] - feature_means[i]) / feature_stds[i]
                    elif i == PRICE_COMP_P1_IDX:
                        if train_idx >= n_trains_context - 1:  # Last train
                            scaled_features[i] = (context[i] - feature_means[i]) / feature_stds[i]
                        else:
                            scaled_features[i] = (price_vars[train_idx + 1] - feature_means[i]) / feature_stds[i]
                    else:  # PRICE_COMP_P2_IDX
                        if train_idx >= n_trains_context - 2:  # Last two trains
                            scaled_features[i] = (context[i] - feature_means[i]) / feature_stds[i]
                        else:
                            scaled_features[i] = (price_vars[train_idx + 2] - feature_means[i]) / feature_stds[i]
                else:
                    # Fixed feature, scale it
                    scaled_features[i] = (context[i] - feature_means[i]) / feature_stds[i]
            
            # --- Add tree constraints and get output ---
            output_var = add_tree_constraints(opt_m, tree_model, scaled_features, train_idx)
            
            # --- ActualDemand logic (same as NN version) ---
            cap = float(capacity_value)
            M = cap * 1000

            # First, handle max(0, output_var)
            s_aux = opt_m.addVar(lb=0, name=f"output_var_nonneg_{train_idx}")
            bin1_aux = opt_m.addVar(vtype=gp.GRB.BINARY, name=f"bin_output_nonneg1_{train_idx}")
            bin2_aux = opt_m.addVar(vtype=gp.GRB.BINARY, name=f"bin_output_nonneg2_{train_idx}")
            
            opt_m.addConstr(s_aux >= 0, name=f"output_var_nonneg_ge_zero_{train_idx}")
            opt_m.addConstr(s_aux >= output_var, name=f"output_var_nonneg_ge_output_{train_idx}")
            opt_m.addConstr(s_aux <= output_var + M * (1 - bin1_aux), name=f"output_var_nonneg_le_output_plus_M_{train_idx}")
            opt_m.addConstr(s_aux <= M * bin1_aux, name=f"output_var_nonneg_le_M_{train_idx}")
            opt_m.addConstr(output_var <= M * bin1_aux, name=f"output_var_le_M_{train_idx}")
            opt_m.addConstr(output_var >= -M * (1 - bin1_aux), name=f"output_var_ge_minus_M_{train_idx}")
            
            # ActualDemand logic with capacity constraints
            ActualDemand = opt_m.addVar(lb=0, ub=cap, name=f"ActualDemand_{train_idx}")
            opt_m.addConstr(ActualDemand <= s_aux, name=f"ActualDemand_le_output_{train_idx}")
            opt_m.addConstr(ActualDemand >= s_aux - M * (1 - bin2_aux), name=f"ActualDemand_ge_output_minus_M_{train_idx}")
            opt_m.addConstr(ActualDemand >= cap - M * bin2_aux, name=f"ActualDemand_ge_cap_minus_M_{train_idx}")

            output_vars.append(output_var)
            s_aux_vars.append(s_aux)
            ActualDemands.append(ActualDemand)
            
            opt_m.update()

        # --- Add total demand constraints ---
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

        # --- Optimization parameters ---
        opt_m.setParam('MIPGap', 0.01)  # Set a small MIP gap for faster convergence
        opt_m.setParam('MIPFocus', 3)  # Focus on improving the best bound
        opt_m.setParam('TimeLimit', TIME_LIMIT) # Set optimization time limit

        # Add infeasibility debugging
        #opt_m.setParam('DualReductions', 0)  # Disable dual reductions for better debugging

        print(f"Starting optimization with {opt_m.NumVars} variables and {opt_m.NumConstrs} constraints...")

        monitoring_flag = True
        monitor_thread = threading.Thread(target=monitor_ram, args=(process,))
        monitor_thread.start()

        opt_m.optimize()

        monitoring_flag = False
        monitor_thread.join()

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

print("\n" + "="*60)
print("ALL SCENARIOS COMPLETED!")
print("="*60)
