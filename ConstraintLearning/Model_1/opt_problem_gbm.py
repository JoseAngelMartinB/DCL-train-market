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
ML_MODEL_NAME = 'gbm'
DEMAND_DATASET = 'ConstraintLearning/preprocesed_data/demand_MAD-BCN_2025.csv'
UNUSED_COLS = ['service_id', 'capacity']
SAVED_MODEL_PATH = os.path.join(project_root, f"ConstraintLearning/saved_models/augmented_demand_{ML_MODEL_NAME}_model.pkl")
RESULTS_PATH = os.path.join(project_root, f'ConstraintLearning/Model_1/results_{ML_MODEL_NAME}/')
CLEAR_PREVIOUS_RESULTS = True  # Set to True to clear previous results
OPTIM_RESULTS_PATH = os.path.join(project_root, 'ConstraintLearning/Model_1/opt_results.csv')
RENFE_PRICES_INTERVAL = [10, 160]
MIPGap = 0.001  # Optimality gap for optimization
TIME_LIMIT = 3 * 3600  # Time limit for optimization
MAX_PREDICTED_DEMAND = 1000  # Maximum predicted demand for a train
DISPLAY_LIMIT = 5  # Limit for displaying train prices on console
FIXED_SPLIT_PRUNE_TOL = 1e-5

# Define different scenarios to test (start with just one scenario for testing)
DELTA_VALUES = [5,10,20] # Delta values indicate the price variation (+/-) in euros allowed
DAYS = [
    '2025-03-12',  # Weekday low demand day
    '2025-03-22',  # Weekend low demand day
    '2025-08-13',  # Weekday high demand day
    '2025-08-23',   # Weekend high demand day
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


# Load the Gradient Boosting model and scalers
model_data = joblib.load(SAVED_MODEL_PATH)
gbm_model = model_data['model']
feature_scaler = model_data['feature_scaler']
target_scaler = model_data['target_scaler']
feature_means = model_data['feature_means']
feature_stds = model_data['feature_stds']
feature_mins = model_data['feature_mins']
feature_maxs = model_data['feature_maxs']
target_mean = model_data['target_mean']
target_std = model_data['target_std']

print(f"Loaded Gradient Boosting model with {gbm_model.n_estimators} trees")
print(f"Learning rate: {gbm_model.learning_rate}")
print(f"Each tree has max_depth: {gbm_model.max_depth}")

# Calculate total nodes across all trees for complexity assessment
total_nodes = sum(tree[0].tree_.node_count for tree in gbm_model.estimators_)
print(f"Total nodes across all trees: {total_nodes}")

# Use all trees for full GBM power
trees_to_use = gbm_model.estimators_
n_trees_to_use = len(trees_to_use)
learning_rate = gbm_model.learning_rate
# Fix initial_prediction to be a scalar
if hasattr(gbm_model.init_, 'constant_'):
    initial_prediction = float(gbm_model.init_.constant_[0])
else:
    initial_prediction = 0.0

print(f"Using all {n_trees_to_use} trees for optimization")
print(f"Learning rate: {learning_rate}")
print(f"Initial prediction: {initial_prediction}")
trees_nodes = sum(tree[0].tree_.node_count for tree in trees_to_use)
print(f"Total nodes for optimization: {trees_nodes}")


# --- Load both DataFrames ---
cleaned_csv_path = os.path.join(project_root, DEMAND_DATASET)

# Check if file exist
if not os.path.exists(cleaned_csv_path):
    print(f"Cleaned CSV file not found at: {cleaned_csv_path}")
    print(f"Project root: {project_root}")
    sys.exit(1)

cleaned_df = pd.read_csv(cleaned_csv_path)

feature_names = [col for col in cleaned_df.columns if col not in UNUSED_COLS + ['passengers']]

# Extract date from service_id in the original DataFrame
def extract_date(service_id):
    # Example: '00003_01-01-2025-06.27'
    parts = service_id.split('_')
    date_part = parts[1].split('-')
    # date_part: ['01', '01', '2025', '06.27']
    return f"{date_part[2]}-{date_part[1]}-{date_part[0]}"  # YYYY-MM-DD

cleaned_df['date'] = cleaned_df['service_id'].apply(extract_date)

# --- Helper functions to encode GBM in MILP ---
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

def add_single_tree_constraints(opt_model, tree, scaled_features, fixed_scaled_features, tree_idx, train_idx):
    """Add constraints for a single tree in the Gradient Boosting model - optimized version"""
    tree_info = get_tree_structure(tree)
    
    # Create all binary variables at once for better efficiency
    node_vars = opt_model.addVars(
        [node['node_id'] for node in tree_info],
        vtype=gp.GRB.BINARY,
        name=f"tree_{tree_idx}_node_train_{train_idx}"
    )
    
    # Root node must be active
    opt_model.addConstr(
        node_vars[0] == 1, 
        name=f"tree_{tree_idx}_root_train_{train_idx}"
    )
    
    # Process internal nodes with optimized constraints
    for node in tree_info:
        if not node['is_leaf']:
            node_id = node['node_id']
            feature_idx = node['feature']
            threshold = node['threshold']
            left_child = node['left_child']
            right_child = node['right_child']
            
            # Flow conservation: if node is active, exactly one child is active
            opt_model.addConstr(
                node_vars[left_child] + node_vars[right_child] == node_vars[node_id],
                name=f"tree_{tree_idx}_flow_{node_id}_train_{train_idx}"
            )

            # Redundant bound propagation for splits on fixed features. The original
            # Big-M constraints already make the opposite branch infeasible; setting
            # UB=0 exposes that fact to presolve without changing the formulation.
            if feature_idx in fixed_scaled_features:
                fixed_value = fixed_scaled_features[feature_idx]
                if fixed_value <= threshold - FIXED_SPLIT_PRUNE_TOL:
                    node_vars[right_child].UB = 0
                elif fixed_value >= threshold + FIXED_SPLIT_PRUNE_TOL:
                    node_vars[left_child].UB = 0
            
            # Use robust Big M values - not too tight to avoid infeasibility
            # Features are standardized, so they typically range from -3 to +3
            # Use conservative Big M values
            M_left = 10  # Large enough for any reasonable standardized feature value
            M_right = 10
            
            # Left child constraints (feature <= threshold when left child is active)
            opt_model.addConstr(
                scaled_features[feature_idx] - threshold - M_left * (1 - node_vars[left_child]) <= 0,
                name=f"tree_{tree_idx}_left_{node_id}_train_{train_idx}"
            )
            
            # Right child constraints (feature > threshold when right child is active)
            opt_model.addConstr(
                scaled_features[feature_idx] - threshold - 1e-6 + M_right * (1 - node_vars[right_child]) >= 0,
                name=f"tree_{tree_idx}_right_{node_id}_train_{train_idx}"
            )
    
    # Create tree output as weighted sum of leaf values
    leaf_terms = []
    leaf_values = []
    for node in tree_info:
        if node['is_leaf']:
            leaf_terms.append(node['value'] * node_vars[node['node_id']])
            leaf_values.append(node['value'])
    
    # Exactly one leaf must be active (SOS1 constraint - more efficient)
    leaf_vars = [node_vars[node['node_id']] for node in tree_info if node['is_leaf']]
    opt_model.addSOS(gp.GRB.SOS_TYPE1, leaf_vars)
    
    # Tree output
    tree_output_var = opt_model.addVar(
        lb=min(leaf_values),
        ub=max(leaf_values),
        name=f"tree_{tree_idx}_out_train_{train_idx}"
    )
    
    opt_model.addConstr(
        tree_output_var == gp.quicksum(leaf_terms),
        name=f"tree_{tree_idx}_sum_train_{train_idx}"
    )
    
    return tree_output_var

def add_gbm_constraints(opt_model, trees_to_use, scaled_features, fixed_scaled_features, train_idx, learning_rate, initial_prediction):
    """Add Gradient Boosting Machine constraints - following OptiCL approach"""
    
    # Get individual tree outputs 
    tree_outputs = []
    for tree_idx, tree_estimator in enumerate(trees_to_use):
        # GBM estimators are stored as arrays, get the first (and only) tree
        tree = tree_estimator[0]
        tree_output = add_single_tree_constraints(
            opt_model, tree, scaled_features, fixed_scaled_features, tree_idx, train_idx
        )
        tree_outputs.append(tree_output)
    
    # GBM output follows the formula: initial_prediction + learning_rate * sum(tree_outputs)
    # This is the key difference from Random Forest which just averages the trees
    gbm_output_scaled = opt_model.addVar(
        lb=-gp.GRB.INFINITY, 
        name=f"gbm_scaled_train_{train_idx}"
    )
    
    # GBM prediction formula: y = initial_prediction + learning_rate * sum(tree_i)
    opt_model.addConstr(
        gbm_output_scaled == initial_prediction + learning_rate * gp.quicksum(tree_outputs),
        name=f"gbm_prediction_train_{train_idx}"
    )
    
    # Transform back to original scale
    gbm_output_original = opt_model.addVar(
        lb=-MAX_PREDICTED_DEMAND,
        ub=MAX_PREDICTED_DEMAND,
        name=f"gbm_out_train_{train_idx}"
    )
    
    opt_model.addConstr(
        gbm_output_original == gbm_output_scaled * target_std + target_mean,
        name=f"gbm_transform_train_{train_idx}"
    )
    
    return gbm_output_original

# Main optimization loop
print("\n" + "="*60)
print("STARTING GRADIENT BOOSTING OPTIMIZATION FOR ALL SCENARIOS")
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
        day_context_matrix = cleaned_df[cleaned_df['date'] == day].drop(columns=['passengers', 'date'] + UNUSED_COLS, errors='ignore').to_numpy()
        
        n_trains_context = day_context_matrix.shape[0]
        PRICE_COMP_M2_IDX = feature_names.index('price_competitor_-2')
        PRICE_COMP_M1_IDX = feature_names.index('price_competitor_-1')
        PRICE_COMP_P1_IDX = feature_names.index('price_competitor_1')
        PRICE_COMP_P2_IDX = feature_names.index('price_competitor_2')
        price_idx = feature_names.index('price')
        
        # Extract 'capacity' values
        capacity_values = cleaned_df['capacity'][cleaned_df['date'] == day].to_numpy()
        print("Capacity values from original data:", capacity_values[:5], "...")  # Show first 5

        # --- Set up Gurobi model for all trains in the day ---
        opt_m = gp.Model("GradientBoostingPricingOptimization")
        
        # Optimize Gurobi parameters for large MILPs
        opt_m.setParam('OutputFlag', 1)
        opt_m.setParam('Threads', 0)  # Use all available cores (0 = automatic)
        #opt_m.setParam('MIPFocus', 3)  # Focus on improving bounds
        #opt_m.setParam('Cuts', 2)  # Aggressive cuts
        opt_m.setParam('Presolve', 2)  # Aggressive presolve
        opt_m.setParam('Heuristics', 0.1)  # Spend 10% time on heuristics
        opt_m.setParam('NodefileStart', 12.0)  # Start writing node file after 12 GB
        opt_m.setParam('MIPGap', MIPGap)  # Set optimality gap
        opt_m.setParam('TimeLimit', TIME_LIMIT)  # Set optimization time limit
        
        # Add infeasibility debugging
        #opt_m.setParam('DualReductions', 0)  # Disable dual reductions for better debugging

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
        for train_idx in range(n_trains_context):
            lb, ub = price_bounds[train_idx]
            original_price = day_context_matrix[train_idx][price_idx]
            price_vars[train_idx].Start = min(max(original_price, lb), ub)

        opt_m.update()

        # --- Now embed the Gradient Boosting Machine for each train ---
        output_vars = []
        s_aux_vars = []
        ActualDemands = []

        for train_idx in range(n_trains_context):
            context = day_context_matrix[train_idx]
            capacity_value = capacity_values[train_idx]
            train_type_AVE = context[feature_names.index('train_type_AVE')]
            train_type_AVLO = context[feature_names.index('train_type_AVLO')]
            
            print(f"Train {train_idx}: train_type_AVE={train_type_AVE}, train_type_AVLO={train_type_AVLO}, price={context[price_idx]}, capacity={capacity_value}")

            # --- Prepare scaled features for the Gradient Boosting Machine ---
            scaled_features = {}
            fixed_scaled_features = {}
            
            for i in range(len(feature_names)):
                if i == price_idx:
                    if train_type_AVE == 1 or train_type_AVLO == 1:
                        # Use price variable and scale it
                        scaled_features[i] = (price_vars[train_idx] - feature_means[i]) / feature_stds[i]
                    else:
                        # Use fixed price and scale it
                        scaled_features[i] = (context[i] - feature_means[i]) / feature_stds[i]
                        fixed_scaled_features[i] = scaled_features[i]
                elif i in [PRICE_COMP_M2_IDX, PRICE_COMP_M1_IDX, PRICE_COMP_P1_IDX, PRICE_COMP_P2_IDX]:
                    # Handle competitor prices - CORRECTED to match validation logic
                    if i == PRICE_COMP_M2_IDX:
                        if train_idx < 2:  # First two trains - use original value
                            scaled_features[i] = (context[i] - feature_means[i]) / feature_stds[i]
                            fixed_scaled_features[i] = scaled_features[i]
                        else:
                            # Check if competitor train (train_idx-2) is AVE/AVLO
                            competitor_context = day_context_matrix[train_idx - 2] 
                            competitor_AVE = competitor_context[feature_names.index('train_type_AVE')]
                            competitor_AVLO = competitor_context[feature_names.index('train_type_AVLO')]
                            if competitor_AVE == 1 or competitor_AVLO == 1:
                                scaled_features[i] = (price_vars[train_idx - 2] - feature_means[i]) / feature_stds[i]
                            else:
                                # Competitor is not AVE/AVLO, use its fixed original price
                                scaled_features[i] = (competitor_context[price_idx] - feature_means[i]) / feature_stds[i]
                                fixed_scaled_features[i] = scaled_features[i]
                    elif i == PRICE_COMP_M1_IDX:
                        if train_idx < 1:  # First train - use original value
                            scaled_features[i] = (context[i] - feature_means[i]) / feature_stds[i]
                            fixed_scaled_features[i] = scaled_features[i]
                        else:
                            # Check if competitor train (train_idx-1) is AVE/AVLO
                            competitor_context = day_context_matrix[train_idx - 1]
                            competitor_AVE = competitor_context[feature_names.index('train_type_AVE')]
                            competitor_AVLO = competitor_context[feature_names.index('train_type_AVLO')]
                            if competitor_AVE == 1 or competitor_AVLO == 1:
                                scaled_features[i] = (price_vars[train_idx - 1] - feature_means[i]) / feature_stds[i]
                            else:
                                # Competitor is not AVE/AVLO, use its fixed original price
                                scaled_features[i] = (competitor_context[price_idx] - feature_means[i]) / feature_stds[i]
                                fixed_scaled_features[i] = scaled_features[i]
                    elif i == PRICE_COMP_P1_IDX:
                        if train_idx >= n_trains_context - 1:  # Last train - use original value
                            scaled_features[i] = (context[i] - feature_means[i]) / feature_stds[i]
                            fixed_scaled_features[i] = scaled_features[i]
                        else:
                            # Check if competitor train (train_idx+1) is AVE/AVLO
                            competitor_context = day_context_matrix[train_idx + 1]
                            competitor_AVE = competitor_context[feature_names.index('train_type_AVE')]
                            competitor_AVLO = competitor_context[feature_names.index('train_type_AVLO')]
                            if competitor_AVE == 1 or competitor_AVLO == 1:
                                scaled_features[i] = (price_vars[train_idx + 1] - feature_means[i]) / feature_stds[i]
                            else:
                                # Competitor is not AVE/AVLO, use its fixed original price
                                scaled_features[i] = (competitor_context[price_idx] - feature_means[i]) / feature_stds[i]
                                fixed_scaled_features[i] = scaled_features[i]
                    else:  # PRICE_COMP_P2_IDX
                        if train_idx >= n_trains_context - 2:  # Last two trains - use original value
                            scaled_features[i] = (context[i] - feature_means[i]) / feature_stds[i]
                            fixed_scaled_features[i] = scaled_features[i]
                        else:
                            # Check if competitor train (train_idx+2) is AVE/AVLO
                            competitor_context = day_context_matrix[train_idx + 2]
                            competitor_AVE = competitor_context[feature_names.index('train_type_AVE')]
                            competitor_AVLO = competitor_context[feature_names.index('train_type_AVLO')]
                            if competitor_AVE == 1 or competitor_AVLO == 1:
                                scaled_features[i] = (price_vars[train_idx + 2] - feature_means[i]) / feature_stds[i]
                            else:
                                # Competitor is not AVE/AVLO, use its fixed original price
                                scaled_features[i] = (competitor_context[price_idx] - feature_means[i]) / feature_stds[i]
                                fixed_scaled_features[i] = scaled_features[i]
                else:
                    # Fixed feature, scale it
                    scaled_features[i] = (context[i] - feature_means[i]) / feature_stds[i]
                    fixed_scaled_features[i] = scaled_features[i]
            
            # --- Add Gradient Boosting Machine constraints and get output ---
            output_var = add_gbm_constraints(
                opt_m, trees_to_use, scaled_features, fixed_scaled_features,
                train_idx, learning_rate, initial_prediction
            )


            # --- ActualDemand logic  ---
            cap = float(capacity_value)
            M = max(cap * 1.25, MAX_PREDICTED_DEMAND)  # Ensure M is large enough but not too large to avoid numerical issues 

            # First, handle max(0, output_var)
            s_aux = opt_m.addVar(lb=0, ub=M, name=f"output_var_nonneg_{train_idx}")
            bin1_aux = opt_m.addVar(vtype=gp.GRB.BINARY, name=f"bin_output_nonneg1_{train_idx}")
            opt_m.addConstr(s_aux >= 0, name=f"output_var_nonneg_ge_zero_{train_idx}")
            opt_m.addConstr(s_aux >= output_var, name=f"output_var_nonneg_ge_output_{train_idx}")
            opt_m.addConstr(s_aux <= output_var + M * (1 - bin1_aux), name=f"output_var_nonneg_le_output_plus_M_{train_idx}")
            opt_m.addConstr(s_aux <= M * bin1_aux, name=f"output_var_nonneg_le_M_{train_idx}")
            opt_m.addConstr(output_var <= M * bin1_aux, name=f"output_var_le_M_{train_idx}")
            opt_m.addConstr(output_var >= -M * (1 - bin1_aux), name=f"output_var_ge_minus_M_{train_idx}")

            # ActualDemand logic with strict enforcement
            ActualDemand = opt_m.addVar(lb=0, ub=cap, name=f"ActualDemand_{train_idx}")
            bin2_aux = opt_m.addVar(vtype=gp.GRB.BINARY, name=f"bin_output_nonneg2_{train_idx}")
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


        # --- Optimize the model using a separate thread for RAM monitoring ---
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
                # Use the incumbent solution value
                objective_value = opt_m.objVal
                
                # Get the service_ids for the selected day
                day_service_ids = cleaned_df[cleaned_df['date'] == day]['service_id'].tolist()
                
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
            
            # IMPORTANT FIX: Get all optimized prices first to ensure consistency
            all_optimized_prices = []
            for train_idx in range(n_trains_context):
                all_optimized_prices.append(price_vars[train_idx].X)
            
            # DEBUG: Print first few optimized prices and their train types
            print("DEBUG: First 5 optimized prices:")
            for i in range(min(5, n_trains_context)):
                train_context = day_context_matrix[i]
                is_ave = train_context[feature_names.index('train_type_AVE')] == 1
                is_avlo = train_context[feature_names.index('train_type_AVLO')] == 1
                original_price = train_context[price_idx]
                optimized_price = all_optimized_prices[i]
                print(f"  Train {i}: AVE={is_ave}, AVLO={is_avlo}, original={original_price:.2f}, optimized={optimized_price:.2f}, diff={optimized_price-original_price:.2f}")
            print()
            
            for train_idx in range(n_trains_context):
                try:
                    # Get optimized price from the optimization model
                    optimized_price = all_optimized_prices[train_idx]
                    
                    # Get optimization model's demand prediction
                    opt_demand_prediction = output_vars[train_idx].X
                    
                    # Prepare context with optimized price for ML model prediction
                    context = day_context_matrix[train_idx].copy()
                    
                    # CRITICAL FIX: Only update own price if this train is AVE/AVLO
                    current_train_context = day_context_matrix[train_idx]
                    current_AVE = current_train_context[feature_names.index('train_type_AVE')]
                    current_AVLO = current_train_context[feature_names.index('train_type_AVLO')]
                    
                    if current_AVE == 1 or current_AVLO == 1:
                        context[price_idx] = optimized_price
                        # Note: optimized_price should differ from original for AVE/AVLO when delta > 0
                    else:
                        # For non-AVE/AVLO trains, keep original price exactly (no change)
                        # context[price_idx] is already the original price from day_context_matrix
                        pass
                    
                    # CORRECTED: Handle competitor prices using the SAME logic as optimization
                    # Only update competitor prices if the competitor train is AVE/AVLO (has variable price)
                    if train_idx >= 2:  # Update price_competitor_-2
                        competitor_train_idx = train_idx - 2
                        competitor_context = day_context_matrix[competitor_train_idx]
                        competitor_AVE = competitor_context[feature_names.index('train_type_AVE')]
                        competitor_AVLO = competitor_context[feature_names.index('train_type_AVLO')]
                        if competitor_AVE == 1 or competitor_AVLO == 1:
                            context[PRICE_COMP_M2_IDX] = all_optimized_prices[competitor_train_idx]
                        # If competitor is not AVE/AVLO, keep original fixed price (already in context)
                        
                    if train_idx >= 1:  # Update price_competitor_-1
                        competitor_train_idx = train_idx - 1
                        competitor_context = day_context_matrix[competitor_train_idx]
                        competitor_AVE = competitor_context[feature_names.index('train_type_AVE')]
                        competitor_AVLO = competitor_context[feature_names.index('train_type_AVLO')]
                        if competitor_AVE == 1 or competitor_AVLO == 1:
                            context[PRICE_COMP_M1_IDX] = all_optimized_prices[competitor_train_idx]
                        
                    if train_idx < n_trains_context - 1:  # Update price_competitor_1
                        competitor_train_idx = train_idx + 1
                        competitor_context = day_context_matrix[competitor_train_idx]
                        competitor_AVE = competitor_context[feature_names.index('train_type_AVE')]
                        competitor_AVLO = competitor_context[feature_names.index('train_type_AVLO')]
                        if competitor_AVE == 1 or competitor_AVLO == 1:
                            context[PRICE_COMP_P1_IDX] = all_optimized_prices[competitor_train_idx]
                        
                    if train_idx < n_trains_context - 2:  # Update price_competitor_2
                        competitor_train_idx = train_idx + 2
                        competitor_context = day_context_matrix[competitor_train_idx]
                        competitor_AVE = competitor_context[feature_names.index('train_type_AVE')]
                        competitor_AVLO = competitor_context[feature_names.index('train_type_AVLO')]
                        if competitor_AVE == 1 or competitor_AVLO == 1:
                            context[PRICE_COMP_P2_IDX] = all_optimized_prices[competitor_train_idx]
                    
                    # Scale features for ML model
                    scaled_context = (context - feature_means) / feature_stds
                    
                    # Convert to DataFrame with proper feature names to avoid sklearn warning
                    scaled_context_df = pd.DataFrame(scaled_context.reshape(1, -1), columns=feature_names)
                    
                    # Get ML model prediction
                    ml_prediction_scaled = gbm_model.predict(scaled_context_df)[0]
                    ml_demand_prediction = ml_prediction_scaled * target_std + target_mean
                    
                    # Calculate difference
                    difference = abs(opt_demand_prediction - ml_demand_prediction)
                    relative_error = (difference / max(ml_demand_prediction, 1e-6)) * 100
                    
                    # Check if this is just numerical precision error
                    is_precision_error = difference < 1e-6
                    
                    validation_results.append({
                        'train_idx': train_idx,
                        'optimized_price': optimized_price,
                        'opt_demand_prediction': opt_demand_prediction,
                        'ml_demand_prediction': ml_demand_prediction,
                        'absolute_difference': difference,
                        'relative_error_pct': relative_error,
                        'is_precision_error': is_precision_error
                    })
                    
                    # Print first 5 trains for quick inspection
                    if train_idx < 5:
                        print(f"Train {train_idx}:")
                        print(f"  Train type: {day_context_matrix[train_idx][feature_names.index('train_type_AVE')]} (AVE), {day_context_matrix[train_idx][feature_names.index('train_type_AVLO')]} (AVLO)")
                        print(f"  Original price: {day_context_matrix[train_idx][price_idx]:.2f}")
                        print(f"  Optimized price: {optimized_price:.2f}")
                        print(f"  Context competitor prices: {context[PRICE_COMP_M2_IDX]:.2f}, {context[PRICE_COMP_M1_IDX]:.2f}, {context[PRICE_COMP_P1_IDX]:.2f}, {context[PRICE_COMP_P2_IDX]:.2f}")
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
                precision_errors = validation_df['is_precision_error'].sum()
                total_validations = len(validation_df)
                
                print(f"VALIDATION SUMMARY:")
                print(f"  Mean absolute difference: {mean_abs_diff:.3f}")
                print(f"  Max absolute difference:  {max_abs_diff:.3f}")
                print(f"  Mean relative error:      {mean_rel_error:.2f}%")
                print(f"  Max relative error:       {max_rel_error:.2f}%")
                print(f"  Precision errors (< 1e-6): {precision_errors}/{total_validations} ({100*precision_errors/total_validations:.1f}%)")
                
                # Save validation results
                objective_value = opt_m.objVal
                
                objective_str = f"{objective_value:.2f}".replace('.', '_')
                validation_filename = f"validation_gbm_{day}_delta_{delta}_obj_{objective_str}.csv"
                validation_filepath = os.path.join(RESULTS_PATH, validation_filename)
                
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
print("ALL GRADIENT BOOSTING SCENARIOS COMPLETED!")
print("="*60)
