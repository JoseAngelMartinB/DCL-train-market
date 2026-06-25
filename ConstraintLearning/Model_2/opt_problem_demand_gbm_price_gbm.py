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
ML_MODEL_NAME = 'demand_gbm-price_gbm'
DEMAND_DATASET = 'ConstraintLearning/preprocesed_data/demand_MAD-BCN_2025.csv'
DEMAND_UNUSED_COLS = ['service_id', 'capacity']
PRICE_DATASET = 'ConstraintLearning/preprocesed_data/price_RENFE_MAD-BCN_2025.csv'
PRICE_UNUSED_COLS = ['service_id']
DEMAND_MODEL_PATH = os.path.join(project_root, "ConstraintLearning/saved_models/augmented_demand_gbm_model.pkl")
PRICE_MODEL_PATH = os.path.join(project_root, "ConstraintLearning/saved_models/price_gbm_model.pkl")
RESULTS_PATH = os.path.join(project_root, f'ConstraintLearning/Model_2/results_{ML_MODEL_NAME}/')
CLEAR_PREVIOUS_RESULTS = True  # Set to True to clear previous results
OPTIM_RESULTS_PATH = os.path.join(project_root, 'ConstraintLearning/Model_2/opt_results.csv')
RENFE_PRICES_INTERVAL = [10, 160]
COMPETITORS_PRICES_INTERVAL = [10, 120]
MIPGap = 0.002  # Optimality gap for optimization
TIME_LIMIT = 6 * 3600  # Time limit for optimization
MAX_PREDICTED_DEMAND = 1000  # Maximum predicted demand for a train
DISPLAY_LIMIT = 5  # Limit for displaying train prices on console
FIXED_SPLIT_PRUNE_TOL = 1e-5

# Define different scenarios to test (start with smaller set for testing the new model)
DELTA_VALUES = [5,10,20] # Delta values indicate the price variation (+/-) in euros allowed
DAYS = [
    '2025-03-12',  # Weekday low demand day
    '2025-03-22',  # Weekend low demand day
    '2025-08-13',  # Weekday high demand day
    '2025-08-23'   # Weekend high demand day
]

# Check if model files exist
for model_path in [DEMAND_MODEL_PATH, PRICE_MODEL_PATH]:
    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}")
        print("Available files in the directory:")
        model_dir = os.path.dirname(model_path)
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


# --- Load the Demand GBM model ---
print("Loading Demand GBM model...")
demand_model_data = joblib.load(DEMAND_MODEL_PATH)
demand_gbm_model = demand_model_data['model']
demand_feature_scaler = demand_model_data['feature_scaler']
demand_target_scaler = demand_model_data['target_scaler']
demand_feature_means = demand_model_data['feature_means']
demand_feature_stds = demand_model_data['feature_stds']
demand_feature_mins = demand_model_data['feature_mins']
demand_feature_maxs = demand_model_data['feature_maxs']
demand_target_mean = demand_model_data['target_mean']
demand_target_std = demand_model_data['target_std']

print(f"Loaded Demand GBM model with {demand_gbm_model.n_estimators} trees")

# Calculate total nodes for demand model
demand_trees = demand_gbm_model.estimators_
demand_initial_prediction = float(demand_gbm_model.init_.constant_.item())  # Use .item() instead of [0]
demand_learning_rate = demand_gbm_model.learning_rate
demand_total_nodes = sum(tree[0].tree_.node_count for tree in demand_trees)

print(f"Demand model - Learning rate: {demand_learning_rate}, Initial prediction: {demand_initial_prediction}")
print(f"Demand model - Total nodes: {demand_total_nodes}")


# --- Load the Price GBM model ---
print("Loading Price GBM model...")
price_model_data = joblib.load(PRICE_MODEL_PATH)
price_gbm_model = price_model_data['model']
price_feature_scaler = price_model_data['feature_scaler']
price_target_scaler = price_model_data['target_scaler']
price_feature_means = price_model_data['feature_means']
price_feature_stds = price_model_data['feature_stds']
price_feature_mins = price_model_data['feature_mins']
price_feature_maxs = price_model_data['feature_maxs']
price_target_mean = price_model_data['target_mean']
price_target_std = price_model_data['target_std']
price_scaled_features_names = price_model_data['scaled_features']

print(f"Loaded Price GBM model with {price_gbm_model.n_estimators} trees")

# Calculate total nodes for price model
price_trees = price_gbm_model.estimators_
price_initial_prediction = float(price_gbm_model.init_.constant_.item())  # Use .item() instead of [0]
price_learning_rate = price_gbm_model.learning_rate
price_total_nodes = sum(tree[0].tree_.node_count for tree in price_trees)

print(f"Price model - Learning rate: {price_learning_rate}, Initial prediction: {price_initial_prediction}")
print(f"Price model - Total nodes: {price_total_nodes}")
print(f"Price model - Scaled features: {price_scaled_features_names}")

# Total complexity
total_nodes = demand_total_nodes + price_total_nodes
print(f"Total nodes across both models: {total_nodes}")


# --- Load both DataFrames ---
# Check if files exist
for csv_path in [DEMAND_DATASET, PRICE_DATASET]:
    csv_path = os.path.join(project_root, csv_path)
    if not os.path.exists(csv_path):
        print(f"CSV file not found at: {csv_path}")
        sys.exit(1)

demand_df = pd.read_csv(os.path.join(project_root, DEMAND_DATASET))
price_df = pd.read_csv(os.path.join(project_root, PRICE_DATASET))

# Extract date from service_id in the original DataFrame
def extract_date(service_id):
    # Example: '00003_01-01-2025-06.27'
    parts = service_id.split('_')
    date_part = parts[1].split('-')
    # date_part: ['01', '01', '2025', '06.27']
    return f"{date_part[2]}-{date_part[1]}-{date_part[0]}"  # YYYY-MM-DD

demand_df['date'] = demand_df['service_id'].apply(extract_date)
price_df['date'] = price_df['service_id'].apply(extract_date)

# Get feature names for both models
demand_features = [col for col in demand_df.columns if col not in DEMAND_UNUSED_COLS + ['passengers', 'date']]
price_features = [col for col in price_df.columns if col not in PRICE_UNUSED_COLS + ['price', 'date']]

print(f"Demand model features ({len(demand_features)}): {demand_features[:5]}...")
print(f"Price model features ({len(price_features)}): {price_features[:5]}...")

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

def add_single_tree_constraints(opt_model, tree, scaled_features, fixed_scaled_features, tree_idx, train_idx, model_name=""):
    """Add constraints for a single tree in a GBM model"""
    tree_info = get_tree_structure(tree)
    
    # Create all binary variables at once for better efficiency
    node_vars = opt_model.addVars(
        [node['node_id'] for node in tree_info],
        vtype=gp.GRB.BINARY,
        name=f"{model_name}_tree_{tree_idx}_node_train_{train_idx}"
    )
    
    # Root node must be active
    opt_model.addConstr(
        node_vars[0] == 1, 
        name=f"{model_name}_tree_{tree_idx}_root_train_{train_idx}"
    )
    
    # Process internal nodes
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
                name=f"{model_name}_tree_{tree_idx}_flow_{node_id}_train_{train_idx}"
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
            
            # Use robust Big M values
            M_left = 12  # Conservative Big M for standardized features
            M_right = 12
            
            # Left child constraints (feature <= threshold when left child is active)
            opt_model.addConstr(
                scaled_features[feature_idx] - threshold - M_left * (1 - node_vars[left_child]) <= 0,
                name=f"{model_name}_tree_{tree_idx}_left_{node_id}_train_{train_idx}"
            )
            
            # Right child constraints (feature > threshold when right child is active)
            opt_model.addConstr(
                scaled_features[feature_idx] - threshold - 1e-6 + M_right * (1 - node_vars[right_child]) >= 0,
                name=f"{model_name}_tree_{tree_idx}_right_{node_id}_train_{train_idx}"
            )
    
    # Create tree output as weighted sum of leaf values
    leaf_terms = []
    leaf_values = []
    for node in tree_info:
        if node['is_leaf']:
            leaf_terms.append(node['value'] * node_vars[node['node_id']])
            leaf_values.append(node['value'])
    
    # Exactly one leaf must be active (SOS1 constraint)
    leaf_vars = [node_vars[node['node_id']] for node in tree_info if node['is_leaf']]
    opt_model.addSOS(gp.GRB.SOS_TYPE1, leaf_vars)
    
    # Tree output
    tree_output_var = opt_model.addVar(
        lb=min(leaf_values),
        ub=max(leaf_values),
        name=f"{model_name}_tree_{tree_idx}_out_train_{train_idx}"
    )
    
    opt_model.addConstr(
        tree_output_var == gp.quicksum(leaf_terms),
        name=f"{model_name}_tree_{tree_idx}_sum_train_{train_idx}"
    )
    
    return tree_output_var

def add_gbm_constraints(opt_model, trees_to_use, scaled_features, fixed_scaled_features, train_idx, learning_rate, initial_prediction, model_name, target_scaler):
    """Add Gradient Boosting Machine constraints"""
    
    # Get individual tree outputs 
    tree_outputs = []
    for tree_idx, tree_estimator in enumerate(trees_to_use):
        tree = tree_estimator[0]
        tree_output = add_single_tree_constraints(
            opt_model, tree, scaled_features, fixed_scaled_features, tree_idx, train_idx, model_name
        )
        tree_outputs.append(tree_output)
    
    # GBM output (scaled)
    gbm_output_scaled = opt_model.addVar(
        lb=-gp.GRB.INFINITY, 
        name=f"{model_name}_scaled_train_{train_idx}"
    )
    
    # GBM prediction formula: y = initial_prediction + learning_rate * sum(tree_i)
    opt_model.addConstr(
        gbm_output_scaled == initial_prediction + learning_rate * gp.quicksum(tree_outputs),
        name=f"{model_name}_prediction_train_{train_idx}"
    )
    
    # Transform back to original scale
    gbm_output_original = opt_model.addVar(
        lb=-MAX_PREDICTED_DEMAND,
        ub=MAX_PREDICTED_DEMAND,
        name=f"{model_name}_out_train_{train_idx}"
    )
    
    opt_model.addConstr(
        gbm_output_original == gbm_output_scaled * target_scaler.scale_[0] + target_scaler.mean_[0],
        name=f"{model_name}_transform_train_{train_idx}"
    )
    
    return gbm_output_original

def create_scaled_feature_for_price_model(opt_model, feature_val, feature_idx, price_feature_means, price_feature_stds, idx, feature_name):
    """Create a scaled feature for the price model"""
    scaled_feature = opt_model.addVar(
        lb=-gp.GRB.INFINITY,
        name=f"price_scaled_{feature_name}_train_{idx}"
    )
    
    opt_model.addConstr(
        scaled_feature == (feature_val - price_feature_means[feature_idx]) / price_feature_stds[feature_idx],
        name=f"price_scale_{feature_name}_train_{idx}"
    )
    
    return scaled_feature

def is_fixed_value(value):
    # If value is a numeric type (int, float, np.integer, np.floating), consider it fixed,
    # otherwise, if it's a Gurobi variable, it's not fixed.
    return isinstance(value, (int, float, np.integer, np.floating))

# Main optimization loop
print("\n" + "="*80)
print("STARTING COMBINED DEMAND-PRICE GBM OPTIMIZATION FOR ALL SCENARIOS")
print("="*80)

for day in DAYS:
    for delta in DELTA_VALUES:
        print(f"\n{'='*60}")
        print(f"OPTIMIZING: Day={day}, Delta=±{delta}")
        print(f"{'='*60}")
        
        # Prepare data for this day
        day_demand_df = demand_df[demand_df['date'] == day].copy()
        day_price_df = price_df[price_df['date'] == day].copy()
        
        total_passengers = day_demand_df['passengers'].sum()
        print(f"Total expected passengers for {day}: {total_passengers}")
        
        # Get day-specific data matrices
        demand_context_matrix = day_demand_df.drop(columns=['passengers', 'date'] + DEMAND_UNUSED_COLS, errors='ignore').to_numpy()
        price_context_matrix = day_price_df.drop(columns=['price', 'date'] + PRICE_UNUSED_COLS, errors='ignore').to_numpy()
        
        # Get service IDs to match between datasets
        day_demand_service_ids = day_demand_df['service_id'].to_numpy()
        day_price_service_ids = day_price_df['service_id'].to_numpy()

        n_trains_context = demand_context_matrix.shape[0]
        
        # Get feature indices for demand model
        demand_price_idx = demand_features.index('price')
        demand_price_comp_indices = {
            -2: demand_features.index('price_competitor_-2'),
            -1: demand_features.index('price_competitor_-1'),
            1: demand_features.index('price_competitor_1'),
            2: demand_features.index('price_competitor_2')
        }
        
        # Get feature indices for price model  
        price_comp_indices = {
            -2: price_features.index('price_competitor_-2'),
            -1: price_features.index('price_competitor_-1'),
            1: price_features.index('price_competitor_1'),
            2: price_features.index('price_competitor_2')
        }

        # Extract capacity values from original data
        capacity_values = day_demand_df['capacity'].to_numpy()
        print(f"Processing {n_trains_context} trains for {day}")

        # --- Set up Gurobi model ---
        opt_m = gp.Model("CombinedDemandPriceGBMOptimization")
        
        # Optimize Gurobi parameters for very large MILPs
        opt_m.setParam('OutputFlag', 1)
        opt_m.setParam('Threads', 0)  # Use all available cores
        opt_m.setParam('Presolve', 2)  # Aggressive presolve
        opt_m.setParam('MIPGap', MIPGap)  # Set optimality gap
        opt_m.setParam('Heuristics', 0.1)  # 10% time on heuristics
        opt_m.setParam('NodefileStart', 16.0)  # Start writing node file after 16 GB
        opt_m.setParam('TimeLimit', TIME_LIMIT)
        
        # --- Create RENFE price variables ---
        renfe_price_vars = []
        renfe_price_bounds = []

        for train_idx in range(n_trains_context):
            demand_context = demand_context_matrix[train_idx]
            train_type_AVE = demand_context[demand_features.index('train_type_AVE')]
            train_type_AVLO = demand_context[demand_features.index('train_type_AVLO')]
            
            if train_type_AVE == 1 or train_type_AVLO == 1:
                # For RENFE trains, price is a variable that will be optimized
                original_price = demand_context[demand_price_idx]
                lb = max(original_price - delta, RENFE_PRICES_INTERVAL[0])
                ub = min(original_price + delta, RENFE_PRICES_INTERVAL[1])
            else:
                # For non-RENFE trains, price will be determined by price model
                lb = ub = demand_context[demand_price_idx]  # Fixed for now, will be replaced
            
            renfe_price_bounds.append((lb, ub))
        
        # Create RENFE price variables
        renfe_price_vars = opt_m.addVars(
            n_trains_context,
            lb=[bounds[0] for bounds in renfe_price_bounds],
            ub=[bounds[1] for bounds in renfe_price_bounds],
            name="renfe_price"
        )
        for train_idx in range(n_trains_context):
            lb, ub = renfe_price_bounds[train_idx]
            original_price = demand_context_matrix[train_idx][demand_price_idx]
            renfe_price_vars[train_idx].Start = min(max(original_price, lb), ub)

        # --- Create competitor price variables (IRYO, OUIGO) ---
        competitor_price_vars = []

        for train_idx in range(n_trains_context):
            demand_context = demand_context_matrix[train_idx]
            train_type_IRYO = demand_context[demand_features.index('train_type_IRYO')]
            train_type_OUIGO = demand_context[demand_features.index('train_type_OUIGO')]
            
            if train_type_IRYO == 1 or train_type_OUIGO == 1:
                # The competitor price will be determined by the Price GBM model as a response to the RENFE prices
                competitor_price_var = opt_m.addVar(
                    lb=COMPETITORS_PRICES_INTERVAL[0],
                    ub=COMPETITORS_PRICES_INTERVAL[1],
                    name=f"competitor_price_{train_idx}"
                )
                competitor_price_vars.append(competitor_price_var)
            else:
                # For non-competitor trains, use None as placeholder
                competitor_price_vars.append(None)

        opt_m.update()

        # --- Now embed, for each train, the Price GBM (if competitor) and Demand GBM ---
        demand_output_vars = []
        s_aux_vars = []
        ActualDemands = []

        for train_idx in range(n_trains_context):
            service_id = day_demand_service_ids[train_idx]
            print(f"Processing train {train_idx} (Service ID: {service_id})...")

            demand_context = demand_context_matrix[train_idx]
            capacity_value = capacity_values[train_idx]
            
            # Determine train types
            is_ave = demand_context[demand_features.index('train_type_AVE')] == 1
            is_avlo = demand_context[demand_features.index('train_type_AVLO')] == 1
            is_iryo = demand_context[demand_features.index('train_type_IRYO')] == 1
            is_ouigo = demand_context[demand_features.index('train_type_OUIGO')] == 1
            
            print(f"  Train {train_idx}: AVE={is_ave}, AVLO={is_avlo}, IRYO={is_iryo}, OUIGO={is_ouigo}")


            # --- 1. First, determine competitor prices using Price GBM for IRYO/OUIGO trains ---
            if is_iryo or is_ouigo:
                print(f"    Setting up Price GBM for train {train_idx}")
                # Find corresponding index in price data to match the service_id in both datasets
                day_price_idx = np.where(day_price_service_ids == service_id)[0]
                if len(day_price_idx) == 0:
                    print(f"    Warning: No matching price data found for service_id {service_id}. Skipping this train.")
                    continue
                # print(f"    Setting up Price GBM for competitor train {train_idx} (at price index {day_price_idx[0]})") # Uncomment for debugging
                day_price_idx = day_price_idx[0]
                price_context = price_context_matrix[day_price_idx]

                # Prepare scaled features for Price GBM
                price_scaled_features = {}
                price_fixed_scaled_features = {}
                
                for i, feature_name in enumerate(price_features):
                    if feature_name.startswith('price_competitor_'):
                        # Handle competitor prices for the price model
                        offset = int(feature_name.split('_')[-1])

                        # Find the competitor train index based on offset and ensure it's a RENFE train
                        competitor_train_idx = None
                        step = 1 if offset > 0 else -1
                        count = 0
                        for j in range(train_idx + step, n_trains_context if step > 0 else -1, step):
                            d_context = demand_context_matrix[j]
                            if d_context[demand_features.index('train_type_AVE')] == 1 or d_context[demand_features.index('train_type_AVLO')] == 1:
                                count += step
                                if count == offset:
                                    competitor_train_idx = j
                                    break

                        if competitor_train_idx is not None and 0 <= competitor_train_idx < n_trains_context:
                            # print(f"      Setting price feature '{feature_name}' for train {train_idx} using competitor train index {competitor_train_idx}") # Uncomment for debugging
                            # Check if this competitor is RENFE (AVE/AVLO)
                            comp_demand_context = demand_context_matrix[competitor_train_idx]
                            comp_is_ave = comp_demand_context[demand_features.index('train_type_AVE')] == 1
                            comp_is_avlo = comp_demand_context[demand_features.index('train_type_AVLO')] == 1
                            
                            if comp_is_ave or comp_is_avlo:
                                # Use RENFE price variable
                                feature_val = renfe_price_vars[competitor_train_idx]
                            else:
                                # Use competitor price variable if it exists, otherwise original price
                                if competitor_price_vars[competitor_train_idx] is not None:
                                    feature_val = competitor_price_vars[competitor_train_idx]
                                else:
                                    feature_val = comp_demand_context[demand_price_idx]  # Fixed original price
                        else:
                            # Out of bounds, use original context value
                            print(f"      Out of bounds for feature '{feature_name}' at train {train_idx}, using original context value.")
                            feature_val = price_context[i]      
                    else:
                        # Regular feature
                        feature_val = price_context[i]
                            
                    # Scale the feature
                    if feature_name in price_scaled_features_names:
                        if is_fixed_value(feature_val):
                            price_fixed_scaled_features[i] = (feature_val - price_feature_means[i]) / price_feature_stds[i]
                        price_scaled_features[i] = create_scaled_feature_for_price_model(
                            opt_m, feature_val, i, price_feature_means, price_feature_stds, 
                            day_price_idx, feature_name
                        )
                    else:
                        price_scaled_features[i] = feature_val
                        if is_fixed_value(feature_val):
                            price_fixed_scaled_features[i] = feature_val
                
                # Add Price GBM constraints and get predicted price
                predicted_price = add_gbm_constraints(
                    opt_m, price_trees, price_scaled_features, price_fixed_scaled_features, train_idx,
                    price_learning_rate, price_initial_prediction, 
                    "price", price_target_scaler
                )
                
                # Constraint: competitor price variable equals predicted price
                opt_m.addConstr(
                    competitor_price_vars[train_idx] == predicted_price,
                    name=f"competitor_price_equals_prediction_{train_idx}"
                )


            # --- 2. Now set up Demand GBM for all trains ---
            print(f"    Setting up Demand GBM for train {train_idx}")
            
            # Prepare scaled features for Demand GBM
            demand_scaled_features = {}
            demand_fixed_scaled_features = {}
            
            for i, feature_name in enumerate(demand_features):
                if feature_name == 'price':
                    # Use appropriate price variable based on train type
                    if is_ave or is_avlo:
                        feature_val = renfe_price_vars[train_idx]
                    elif is_iryo or is_ouigo:
                        feature_val = competitor_price_vars[train_idx]
                    else:
                        feature_val = demand_context[i]  # Fixed original price
                        
                    demand_scaled_features[i] = (feature_val - demand_feature_means[i]) / demand_feature_stds[i]
                    if is_fixed_value(feature_val):
                        demand_fixed_scaled_features[i] = demand_scaled_features[i]
                    
                elif feature_name.startswith('price_competitor_'):
                    # Handle competitor prices in demand model
                    offset = int(feature_name.split('_')[-1])
                    competitor_train_idx = train_idx + offset
                    
                    if 0 <= competitor_train_idx < n_trains_context:
                        # Check competitor type and use appropriate price
                        comp_demand_context = demand_context_matrix[competitor_train_idx]
                        comp_is_ave = comp_demand_context[demand_features.index('train_type_AVE')] == 1
                        comp_is_avlo = comp_demand_context[demand_features.index('train_type_AVLO')] == 1
                        comp_is_iryo = comp_demand_context[demand_features.index('train_type_IRYO')] == 1
                        comp_is_ouigo = comp_demand_context[demand_features.index('train_type_OUIGO')] == 1
                        
                        if comp_is_ave or comp_is_avlo:
                            feature_val = renfe_price_vars[competitor_train_idx]
                        elif comp_is_iryo or comp_is_ouigo:
                            feature_val = competitor_price_vars[competitor_train_idx]
                        else:
                            feature_val = comp_demand_context[demand_price_idx]  # Fixed price
                    else:
                        # Out of bounds, use original context value
                        feature_val = demand_context[i]
                        
                    demand_scaled_features[i] = (feature_val - demand_feature_means[i]) / demand_feature_stds[i]
                    if is_fixed_value(feature_val):
                        demand_fixed_scaled_features[i] = demand_scaled_features[i]
                else:
                    # Regular feature, scale it normally
                    demand_scaled_features[i] = (demand_context[i] - demand_feature_means[i]) / demand_feature_stds[i]
                    demand_fixed_scaled_features[i] = demand_scaled_features[i]
            
            # Add Demand GBM constraints and get predicted demand
            demand_output_var = add_gbm_constraints(
                opt_m, demand_trees, demand_scaled_features, demand_fixed_scaled_features, train_idx,
                demand_learning_rate, demand_initial_prediction,
                "demand", demand_target_scaler
            )


            # --- 3. Add ActualDemand logic with capacity constraints ---
            cap = float(capacity_value)
            M = max(cap * 1.25, MAX_PREDICTED_DEMAND)  # Ensure M is large enough but not too large to avoid numerical issues 

            # Handle max(0, output_var)
            s_aux = opt_m.addVar(lb=0, ub=M, name=f"demand_nonneg_{train_idx}")
            bin1_aux = opt_m.addVar(vtype=gp.GRB.BINARY, name=f"bin_demand_nonneg1_{train_idx}")
            opt_m.addConstr(s_aux >= 0, name=f"demand_nonneg_ge_zero_{train_idx}")
            opt_m.addConstr(s_aux >= demand_output_var, name=f"demand_nonneg_ge_output_{train_idx}")
            opt_m.addConstr(s_aux <= demand_output_var + M * (1 - bin1_aux), name=f"demand_nonneg_le_output_plus_M_{train_idx}")
            opt_m.addConstr(s_aux <= M * bin1_aux, name=f"demand_nonneg_le_M_{train_idx}")
            opt_m.addConstr(demand_output_var <= M * bin1_aux, name=f"demand_output_le_M_{train_idx}")
            opt_m.addConstr(demand_output_var >= -M * (1 - bin1_aux), name=f"demand_output_ge_minus_M_{train_idx}")
            
            # ActualDemand with capacity constraint
            ActualDemand = opt_m.addVar(lb=0, ub=cap, name=f"ActualDemand_{train_idx}")
            bin2_aux = opt_m.addVar(vtype=gp.GRB.BINARY, name=f"bin_demand_cap_{train_idx}")
            opt_m.addConstr(ActualDemand <= s_aux, name=f"ActualDemand_le_demand_{train_idx}")
            opt_m.addConstr(ActualDemand >= s_aux - M * (1 - bin2_aux), name=f"ActualDemand_ge_demand_{train_idx}")
            opt_m.addConstr(ActualDemand >= cap - M * bin2_aux, name=f"ActualDemand_le_cap_{train_idx}")
            
            demand_output_vars.append(demand_output_var)
            s_aux_vars.append(s_aux)
            ActualDemands.append(ActualDemand)
            
            opt_m.update()


        # --- Add total demand constraints ---
        opt_m.addConstr(gp.quicksum(s_aux_vars) >= 0.6 * total_passengers, name="min_total_demand")
        opt_m.addConstr(gp.quicksum(s_aux_vars) <= 1.4 * total_passengers, name="max_total_demand")

        # --- Objective: maximize revenue for RENFE trains (AVE/AVLO) ---
        total_revenue = gp.LinExpr()
        for i in range(n_trains_context):
            demand_context = demand_context_matrix[i]
            is_ave = demand_context[demand_features.index('train_type_AVE')] == 1
            is_avlo = demand_context[demand_features.index('train_type_AVLO')] == 1
            
            if is_ave or is_avlo:
                total_revenue += renfe_price_vars[i] * ActualDemands[i]

        opt_m.setObjective(total_revenue, gp.GRB.MAXIMIZE)
        opt_m.update()


        # --- Optimize the model using a separate thread for RAM monitoring ---
        print(f"Starting optimization with {opt_m.NumVars} variables and {opt_m.NumConstrs} constraints...")
        print(f"Including {opt_m.NumBinVars} binary variables...")

        monitoring_flag = True
        monitor_thread = threading.Thread(target=monitor_ram, args=(process,))
        monitor_thread.start()
        
        opt_m.optimize()

        monitoring_flag = False
        monitor_thread.join()

        # Check for infeasibility
        if opt_m.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible! Computing IIS...")
            opt_m.computeIIS()
            print("First few IIS constraints:")
            iis_count = 0
            for c in opt_m.getConstrs():
                if c.IISConstr:
                    print(f"  Constraint: {c.constrName}")
                    iis_count += 1
                    if iis_count >= 10:  # Limit output
                        print("  ... (more IIS constraints)")
                        break
            continue


        # --- Print solution ---
        if opt_m.status == gp.GRB.OPTIMAL:
            print("Optimal solution found!")
            for train_idx in range(min(DISPLAY_LIMIT, n_trains_context)):
                demand_context = demand_context_matrix[train_idx]
                is_ave = demand_context[demand_features.index('train_type_AVE')] == 1
                is_avlo = demand_context[demand_features.index('train_type_AVLO')] == 1
                is_iryo = demand_context[demand_features.index('train_type_IRYO')] == 1
                is_ouigo = demand_context[demand_features.index('train_type_OUIGO')] == 1
                
                print(f"Train {train_idx}:")
                if is_ave or is_avlo:
                    print(f"  RENFE optimal price: {renfe_price_vars[train_idx].X:.2f}")
                elif is_iryo or is_ouigo:
                    print(f"  Competitor predicted price: {competitor_price_vars[train_idx].X:.2f}")
                else:
                    print(f"  Other train price: {demand_context[demand_price_idx]:.2f}")
                print(f"  Predicted demand: {demand_output_vars[train_idx].X:.2f}")
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
                
                # Get service_ids for the selected day
                day_service_ids = day_demand_df['service_id'].tolist()
                
                # Prepare data for CSV
                results_data = []
                for train_idx in range(n_trains_context):
                    try:
                        service_id = day_service_ids[train_idx]
                        demand_context = demand_context_matrix[train_idx]
                        
                        # Determine train type
                        is_ave = demand_context[demand_features.index('train_type_AVE')] == 1
                        is_avlo = demand_context[demand_features.index('train_type_AVLO')] == 1 
                        is_iryo = demand_context[demand_features.index('train_type_IRYO')] == 1
                        is_ouigo = demand_context[demand_features.index('train_type_OUIGO')] == 1
                        
                        if is_ave:
                            train_type = "AVE"
                            optimized_price = renfe_price_vars[train_idx].X
                            original_price = demand_context[demand_price_idx]
                        elif is_avlo:
                            train_type = "AVLO"
                            optimized_price = renfe_price_vars[train_idx].X
                            original_price = demand_context[demand_price_idx]
                        elif is_iryo:
                            train_type = "IRYO"
                            optimized_price = competitor_price_vars[train_idx].X
                            original_price = demand_context[demand_price_idx]
                        elif is_ouigo:
                            train_type = "OUIGO"
                            optimized_price = competitor_price_vars[train_idx].X
                            original_price = demand_context[demand_price_idx]
                        else:
                            train_type = "Other"
                            optimized_price = demand_context[demand_price_idx]
                            original_price = demand_context[demand_price_idx]
                        
                        results_data.append({
                            'train_idx': service_id,
                            'train_type': train_type,
                            'original_price': original_price,
                            'optimized_price': optimized_price,
                            'difference': optimized_price - original_price,
                            'predicted_demand': demand_output_vars[train_idx].X,
                            'actual_demand': ActualDemands[train_idx].X,
                            'capacity': capacity_values[train_idx]
                        })
                    except Exception as e:
                        print(f"Error processing train {train_idx}: {e}")
                        continue
                
                # Create DataFrame and save results
                results_df = pd.DataFrame(results_data)
                objective_str = f"{objective_value:.2f}".replace('.', '_')
                results_filename = f"{day}_delta_{delta}_obj_{objective_str}.csv"
                results_filepath = os.path.join(RESULTS_PATH, results_filename)
                
                results_df.to_csv(results_filepath, index=False)
                print(f"\nResults saved to: {results_filepath}")
                print(f"Total revenue (objective): {objective_value:.2f}")

                # Update optimization results DataFrame
                status_str = {gp.GRB.OPTIMAL: "Optimal", gp.GRB.INTERRUPTED: "Interrupted", 
                             gp.GRB.TIME_LIMIT: "Time Limit"}.get(opt_m.status, "Other")
                
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

print("\n" + "="*80)
print("ALL COMBINED DEMAND-PRICE GBM SCENARIOS COMPLETED!")
print("="*80)
