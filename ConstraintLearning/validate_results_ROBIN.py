import os
import gc
import shutil
import pandas as pd
import numpy as np
import re
import yaml
import time
import multiprocessing

from tqdm import tqdm

from robin.kernel.entities import Kernel

# --- Config ---
optim_models = {
    'Model_1': ['tree', 'rf', 'gbm', 'ffnn'],
    #'Model_2': ['demand_gbm-price_gbm'],
}
delta_vect = [5, 10, 20]
days_to_test = ['2025-03-12', '2025-03-22', '2025-08-13', '2025-08-23']
num_simulations = 25 # 10 # 25 # 50 # 100
seed = 2025 # Initial random seed for reproducibility
num_processors = 5  # Number of processors for parallel execution
competitors_response = False # If True, competitors' prices are updated based on RENFE prices
keep_validation_results = True # If True, keeps the validation results after execution
reset_previous_output = False # If True, resets the final results file with all the previous results and starts from scratch
skip_previous_results = False # If True, skips the experiments that have already been processed in the final results file
restricted_service_providers = None # [2,4] # 1: AVLO, 2: IRIO, 3: AVE, 4: OUIGO # Do not update prices for these service providers
COMPETITORS_PRICES_INTERVAL = [10, 140] # Min and max price for competitors' services (same as in data generation)
clean_original_ROBIN_output = False # If True, removes the original ROBIN output files after execution
clean_modified_ROBIN_output = True # If True, removes the ROBIN output files for the modified supply configurations after execution

# --- Paths ---
path_config_supply = '../DataGenerationROBIN/data/MAD-BCN/supply_MAD-BCN_2025.yaml'
path_config_demand = '../DataGenerationROBIN/data/MAD-BCN/demand_data.yaml'
path_validation_results = 'validation_data/'
path_kernel_output = 'validation_data/ROBIN_output/'
path_original_results = '../DataGenerationROBIN/data/MAD-BCN/aggregated/aggregated_MAD-BCN_2025.csv'
final_results_path = os.path.join(path_validation_results, f"final_results.csv")

# Main function to run the ROBIN simulation
def run_sim(args):
    sim, output_supply_file, path_config_demand, robin_output_path = args
    kernel = Kernel(
        path_config_supply=output_supply_file,
        path_config_demand=path_config_demand,
        seed=seed + sim)
    kernel.simulate(
        robin_output_path,
        departure_time_hard_restriction=False,
        calculate_global_utility=False)
    services_df = pd.read_csv(robin_output_path, low_memory=False)
    services_df = services_df[services_df['arrival_day'].isin(days_to_test)]
    services_df.to_csv(robin_output_path, index=False)
    del kernel, services_df
    gc.collect()
    return sim

# Function to obtain the competitor's price based on RENFE prices using the same formulae as in the data generation step
def get_competitors_price(ref_price, t, PR1, PR2, t_R1, t_R2, rng=np.random.default_rng(0)) -> float:
    """
    Generate the price of a competitor's service using RENFE prices as a reference.

    This function generates a synthetic price for a competitor train service at time `t`, by 
    taking into account the reference/base competitor price (`price`), as well as the
    nearest previous (`PR1`) and next (`PR2`) RENFE prices. It weights these reference prices 
    inversely by the temporal distance (in minutes) to the competitor's departure time, adding 
    a small random perturbation to simulate pricing variability.
    """
    error = rng.normal(0, 10)

    d_tr1 = max(abs((t - t_R1).total_seconds() / 60), 1)
    d_tr2 = max(abs((t - t_R2).total_seconds() / 60), 1)
    denominator = 2/3 + 1/d_tr1 + 1/d_tr2

    price = 2/3 * ref_price/denominator + 1/d_tr1 * PR1/denominator + 1/d_tr2 * PR2/denominator + error
    
    # Ensure the price is within the specified interval
    price = max(min(price, COMPETITORS_PRICES_INTERVAL[1]), COMPETITORS_PRICES_INTERVAL[0])
    price = round(price, 2)

    return price

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

    # Initialize final results dataframe
    if not os.path.exists(final_results_path) or reset_previous_output:
        if os.path.exists(final_results_path):
            os.remove(final_results_path)
            print(f"Removed previous final results file: {final_results_path}")
        print("Creating new final results dataframe...")
        final_results = pd.DataFrame(columns=[
            'optim_model',
            'ml_model',
            'delta',
            'day',
            'original_passengers_mean',
            'original_passengers_std',
            'original_passengers_se',
            'new_passengers_mean',
            'new_passengers_std',
            'new_passengers_se',
            'original_revenue_mean',
            'original_revenue_std',
            'original_revenue_se',
            'optimized_revenue',
            'actual_revenue_mean',
            'actual_revenue_std',
            'actual_revenue_se',
            'revenue_difference_mean',
            'revenue_difference_std',
            'revenue_difference_se',
            'revenue_difference_percentage',
            'average_price',
            'average_original_price'
        ])
    else:
        final_results = pd.read_csv(final_results_path)
        print(f"Loaded existing final results from {final_results_path}")


    #%%
    # Start the validation process
    print("Starting validation process...")
    init_time = time.time()

    # %%
    # Execute the ROBIN simulation with the original supply configuration
    # If already executed, count if the number of simulations is correct, if not, raise a warning and re-execute. If not executed, run the simulations.
    original_output_path = os.path.join(path_kernel_output, 'Original_data/')
    run_original_simulation = True
    if os.path.exists(original_output_path):
        existing_files = [f for f in os.listdir(original_output_path) if f.startswith('kernel_output_sim') and f.endswith('.csv')]
        if len(existing_files) == num_simulations:
            print(f"Original ROBIN simulation already executed with {num_simulations} simulations.")
            run_original_simulation = False
        else:
            print(f"Warning: Original ROBIN simulation output found but with {len(existing_files)} simulations instead of {num_simulations}. Re-executing simulations.")
            shutil.rmtree(original_output_path)
            os.makedirs(original_output_path, exist_ok=True)
    
    if run_original_simulation:
        print("Starting ROBIN simulation with the original configuration...")

        #%% Modify the supply data
        # Load the original supply configuration file
        with open(path_config_supply, 'r') as file:
            supply_config = yaml.safe_load(file)
        # Filter only the days to test
        supply_config['service'] = [item for item in supply_config['service'] if item['date'] in days_to_test]

        # Update the capacity of the trains to the original capacity. This removes the infinite capacity that is set in the original supply configuration
        for rs in supply_config['rollingStock']:
            rs_capacity = int(rs['id'].split('_')[1].replace('C', ''))
            rs['seats'][0]['quantity'] = rs_capacity

        # Save the modified supply configuration
        os.makedirs(original_output_path, exist_ok=True)
        output_supply_file = os.path.join(original_output_path, 'supply_MAD-BCN_2025.yaml')
        with open(output_supply_file, 'w') as file:
            yaml.dump(supply_config, file, default_flow_style=False, sort_keys=False)
        print(f"New supply configuration saved to {output_supply_file}")

        args_list = [
                        (
                            sim,
                            output_supply_file,
                            path_config_demand,
                            os.path.join(
                                path_kernel_output,
                                'Original_data/',
                                f'kernel_output_sim{sim+1}.csv'
                            )
                        )
                        for sim in range(num_simulations)
                    ]
        # Use multiprocessing to run the simulations in parallel
        with multiprocessing.Pool(processes=num_processors) as pool:
            for _ in tqdm(pool.imap_unordered(run_sim, args_list), total=num_simulations):
                pass
        print(f"ROBIN simulation completed. Output saved to {original_output_path}")


    #%% Load the original demand data    
    original_agg_data = pd.read_csv(path_original_results)
    original_agg_data.rename(columns={'passengers': 'original_demand'}, inplace=True)

    # Filter only the days to test using the 'service_id' column, which has the format 'train_idx' (e.g., 'anumbergoeshere_12-03-2025-08.00') the date should be 2025-03-12 
    original_agg_data['date'] = original_agg_data['service_id'].str.split('_').str[1].str.split('-').str[2::-1].str.join('-')
    original_agg_data = original_agg_data[original_agg_data['date'].isin(days_to_test)].copy()
    original_agg_data.reset_index(drop=True, inplace=True)
    original_agg_data.drop(columns=['date'], inplace=True)

    # Obtain the original demand and revenue from the original ROBIN output, to compare with the optimized results later
    print("Post-processing ROBIN output for original configuration...")
    original_agg_data['original_passengers_list'] = original_agg_data.apply(lambda _: [], axis=1)

    for sim in range(num_simulations):
        robin_output_path = os.path.join(path_kernel_output, 'Original_data/', f'kernel_output_sim{sim+1}.csv')
        if not os.path.exists(robin_output_path):
            raise FileNotFoundError(f"ROBIN output file {robin_output_path} does not exist. The simulation might have failed.")

        # Aggregate the data
        robin_output = pd.read_csv(robin_output_path)
        robin_output_grouped = robin_output.groupby(['service']).size()

        # Calculate the number of passengers for this simulation
        expected_demand = original_agg_data['service_id'].map(robin_output_grouped).fillna(0).astype(int)
        # Limit the capacity of each train to the maximum capacity defined (if not already limited by ROBIN), to avoid unrealistic demand values
        if (expected_demand > original_agg_data['capacity']).any():
            print("Warning: Expected demand exceeds capacity for some trains. Limiting expected demand to capacity.")
            expected_demand = expected_demand.clip(upper=original_agg_data['capacity'])

        # Append the results to the aggregated dataset
        for idx in original_agg_data.index:
            original_agg_data.at[idx, 'original_passengers_list'].append(int(expected_demand.iloc[idx]))

        del robin_output, robin_output_grouped
        gc.collect()


    # %% Validate the results for each optimization model, ML model and delta value
    for optim_model in optim_models.keys():
        print(f"\n\n\nProcessing optimization model: {optim_model}")
        model_subpath = f'{optim_model}'

        # Check if the model subpath exists, if not raise an error
        if not os.path.exists(model_subpath):
            raise FileNotFoundError(f"The specified model subpath '{model_subpath}/' does not exist. Please check the path.")

        # Remove the previous output file if it exists
        if os.path.exists(os.path.join(path_kernel_output, model_subpath)):
            shutil.rmtree(os.path.join(path_kernel_output, model_subpath))
            print(f"Removed previous output directory: {os.path.join(path_kernel_output, model_subpath)}")

        for ml_model in optim_models[optim_model]:
            print(f"\n\nProcessing ML model: {ml_model}")

            # Load the files and execute the ROBIN simulation for each delta value
            for delta in delta_vect:
                print(f"\n\nProcessing delta: {delta}")

                if skip_previous_results and not final_results[(final_results['optim_model'] == optim_model) &
                                                                            (final_results['ml_model'] == ml_model) &
                                                                            (final_results['delta'] == delta)].empty:
                    print(f"Skipping delta {delta} for model {ml_model} as it has already been processed.")
                    continue

                # Load the optimized prices
                optimized_prices = pd.DataFrame()
                best_optim_revenue = {}
                results_files = os.listdir(os.path.join(model_subpath, f"results_{ml_model}"))
                for day in days_to_test:
                    print(f"Processing day: {day}")
                    # Filter files for the current day using regex
                    pattern = re.compile(rf'{day}_delta_{delta}_obj_(\d+)_(\d+)\.csv')
                    matching_files = [f for f in results_files if pattern.match(f)]
                    if not matching_files:
                        print(f"No results found for day {day}. Skipping.")
                        continue
                    # Sort files by objective value (the part after 'obj_') and take the one with the highest value
                    matching_files.sort(key=lambda f: float(pattern.match(f).group(1) + '.' + pattern.match(f).group(2)), reverse=True)
                    best_file = matching_files[0]
                    print(f"Best file for day {day}: {best_file}")

                    # Load the data from the best file
                    df = pd.read_csv(os.path.join(os.path.join(model_subpath, f"results_{ml_model}"), best_file))
                    df['day'] = day
                    df['departure_time'] = pd.to_datetime(df['train_idx'].str.split('_').str[1], format='%d-%m-%Y-%H.%M')

                    # Simulate competitors' response to new prices
                    if competitors_response:
                        renfe_prices = df[df['train_type'].isin(['AVE', 'AVLO'])][['departure_time', 'optimized_price']].set_index('departure_time')
                        competitors_indices = df[~df['train_type'].isin(['AVE', 'AVLO'])].index
                        for idx in competitors_indices:
                            departure_time = df.at[idx, 'departure_time']
                            # Find nearest RENFE prices before and after the competitor's departure time
                            renfe_before = renfe_prices[renfe_prices.index <= departure_time]
                            # Obtain the reference prices (PR1, PR2) for get_competitors_price function
                            if renfe_before.empty:
                                # Mean RENFE price at that day
                                PR1 = renfe_prices['optimized_price'].mean()
                                t_R1 = departure_time - pd.Timedelta(hours=12)
                            else:
                                PR1 = renfe_before.iloc[-1]['optimized_price']
                                t_R1 = renfe_before.index[-1]
                            renfe_after = renfe_prices[renfe_prices.index >= departure_time]
                            if renfe_after.empty:
                                # Mean RENFE price at that day
                                PR2 = renfe_prices['optimized_price'].mean()
                                t_R2 = departure_time + pd.Timedelta(hours=12)
                            else:
                                PR2 = renfe_after.iloc[0]['optimized_price']
                                t_R2 = renfe_after.index[0]
                            # Update competitor's price
                            new_price = get_competitors_price(
                                ref_price=df.at[idx, 'optimized_price'],
                                t=departure_time,
                                PR1=PR1,
                                PR2=PR2,
                                t_R1=t_R1,
                                t_R2=t_R2,
                                rng=np.random.default_rng(seed)
                            )
                            df.at[idx, 'optimized_price'] = new_price
                            df.at[idx, 'difference'] = new_price - df.at[idx, 'original_price']

                    # Concatenate that day to the optimized prices dataframe
                    optimized_prices = pd.concat([optimized_prices, df], ignore_index=True)
                    best_optim_revenue[day] = float(pattern.match(best_file).group(1) + '.' + pattern.match(best_file).group(2))

                if optimized_prices.empty:
                    raise ValueError("No optimized prices found for the specified days.")


                #%% Modify the supply data to include the optimized prices
                # Load the original supply configuration file
                with open(path_config_supply, 'r') as file:
                    supply_config = yaml.safe_load(file)
                # Filter only the days to test
                supply_config['service'] = [item for item in supply_config['service'] if item['date'] in days_to_test]

                # Update the prices in the supply configuration
                for item in supply_config['service']:
                    # Find the corresponding optimized price for this train
                    train_id = item['id']
                    price_row = optimized_prices[optimized_prices['train_idx'] == train_id]
                    if not price_row.empty:
                        if restricted_service_providers is None or int(item['train_service_provider']) not in restricted_service_providers:
                            # Update the price in the supply configuration
                            for od in item['origin_destination_tuples']:
                                for seat in od['seats']:
                                    # Store price as a two-decimal float
                                    seat['price'] = float(np.round(price_row['optimized_price'].values[0], 2))
                    else:
                        print(f"Warning: No optimized price found for train {train_id} on day {item['date']}.")

                # Update the capacity of the trains to the original capacity. This removes the infinite capacity that is set in the original supply configuration
                for rs in supply_config['rollingStock']:
                    rs_capacity = int(rs['id'].split('_')[1].replace('C', ''))
                    rs['seats'][0]['quantity'] = rs_capacity

                # Save the modified supply configuration
                output_supply_path = os.path.join(path_kernel_output + model_subpath)
                os.makedirs(output_supply_path, exist_ok=True)
                output_supply_file = os.path.join(output_supply_path, 'supply_MAD-BCN_2025.yaml')
                with open(output_supply_file, 'w') as file:
                    yaml.dump(supply_config, file, default_flow_style=False, sort_keys=False)

                print(f"New supply configuration saved to {output_supply_file}")


                # %%
                # Execute the ROBIN simulation with the modified supply configuration
                print("Starting ROBIN simulation with modified prices...")
                args_list = [
                    (
                        sim,
                        output_supply_file,
                        path_config_demand,
                        os.path.join(
                            path_kernel_output,
                            model_subpath,
                            f'kernel_output_delta_{delta}_sim{sim+1}.csv'
                        )
                    )
                    for sim in range(num_simulations)
                ]
                # Use multiprocessing to run the simulations in parallel
                with multiprocessing.Pool(processes=num_processors) as pool:
                    for _ in tqdm(pool.imap_unordered(run_sim, args_list), total=num_simulations):
                        pass
                print(f"ROBIN simulation completed. Output saved to {os.path.join(path_kernel_output, model_subpath)}")


                #%%
                # Pos-process the ROBIN output
                print("Post-processing ROBIN output...")
                agg_dataset = optimized_prices.copy()
                agg_dataset['passengers_list'] = agg_dataset.apply(lambda _: [], axis=1)

                for sim in range(num_simulations):
                    robin_output_path = os.path.join(path_kernel_output, model_subpath, f'kernel_output_delta_{delta}_sim{sim+1}.csv')
                    if not os.path.exists(robin_output_path):
                        raise FileNotFoundError(f"ROBIN output file {robin_output_path} does not exist. The simulation might have failed.")

                    # Aggregate the data
                    robin_output = pd.read_csv(robin_output_path)
                    robin_output_grouped = robin_output.groupby(['service']).size()

                    # Calculate the expected number of passengers for this simulation
                    expected_demand = agg_dataset['train_idx'].map(robin_output_grouped).fillna(0).astype(int)
                    # Limit the capacity of each train to the maximum capacity defined (if not already limited by ROBIN), to avoid unrealistic demand values
                    if (expected_demand > agg_dataset['capacity']).any():
                        print("Warning: Expected demand exceeds capacity for some trains. Limiting expected demand to capacity.")
                        expected_demand = expected_demand.clip(upper=agg_dataset['capacity'])

                    # Append the results to the aggregated dataset
                    for idx in agg_dataset.index:
                        agg_dataset.at[idx, 'passengers_list'].append(int(expected_demand.iloc[idx]))

                    del robin_output, robin_output_grouped
                    gc.collect()
                    
                # Save the aggregated dataset
                os.makedirs(os.path.join(path_validation_results, optim_model), exist_ok=True)
                output_agg_path = os.path.join(path_validation_results, optim_model, f"{ml_model}_validation_delta_{delta}.csv")
                agg_dataset.to_csv(output_agg_path, index=False)
                print(f"Aggregated dataset saved to {output_agg_path}")


                # %%
                # Compute the total revenue per day for Renfe [AVE + AVLO]
                agg_dataset_RENFE = agg_dataset[(agg_dataset['train_type'] == 'AVE') | (agg_dataset['train_type'] == 'AVLO')]
                agg_dataset_RENFE = agg_dataset_RENFE.merge(original_agg_data[['service_id', 'original_passengers_list']],
                                                            left_on='train_idx', right_on='service_id', how='left')

                agg_dataset_RENFE['original_revenue_list'] = agg_dataset_RENFE.apply(lambda row: [p * row['original_price'] for p in row['original_passengers_list']], axis=1)
                
                # Calculate the new revenue
                agg_dataset_RENFE['revenue_list'] = agg_dataset_RENFE.apply(lambda row: [p * row['optimized_price'] for p in row['passengers_list']], axis=1)

                # Save the Renfe revenue data
                renfe_revenue_path = os.path.join(path_validation_results, optim_model, f"{ml_model}_validation_delta_{delta}_renfe_revenue.csv")
                agg_dataset_RENFE.to_csv(renfe_revenue_path, index=False)
                print(f"Renfe revenue data saved to {renfe_revenue_path}")


                #%%
                print("Total revenue for Renfe (AVE + AVLO):")
                for day in days_to_test:
                    day_data = agg_dataset_RENFE[agg_dataset_RENFE['day'] == day]

                    original_revenue_matrix = np.array(day_data['original_revenue_list'].tolist(), dtype=float)
                    original_revenue = original_revenue_matrix.sum(axis=0)
                    original_revenue_mean = original_revenue.mean()
                    original_revenue_std = original_revenue.std()
                    original_revenue_se = original_revenue_std / np.sqrt(len(original_revenue))

                    revenue_matrix = np.array(day_data['revenue_list'].tolist(), dtype=float)
                    actual_revenue = revenue_matrix.sum(axis=0)
                    actual_revenue_mean = actual_revenue.mean()
                    actual_revenue_std = actual_revenue.std()
                    actual_revenue_se = actual_revenue_std / np.sqrt(len(actual_revenue))

                    revenue_difference = actual_revenue - original_revenue
                    revenue_difference_mean = revenue_difference.mean()
                    revenue_difference_std = revenue_difference.std()
                    revenue_difference_se = revenue_difference_std / np.sqrt(len(revenue_difference))

                    original_passengers_matrix = np.array(day_data['original_passengers_list'].tolist(), dtype=int)
                    original_passengers = original_passengers_matrix.sum(axis=0)
                    original_passengers_mean = original_passengers.mean()
                    original_passengers_std = original_passengers.std()
                    original_passengers_se = original_passengers_std / np.sqrt(len(original_passengers))

                    passengers_matrix = np.array(day_data['passengers_list'].tolist(), dtype=int)
                    new_passengers = passengers_matrix.sum(axis=0)
                    new_passengers_mean = new_passengers.mean()
                    new_passengers_std = new_passengers.std()
                    new_passengers_se = new_passengers_std / np.sqrt(len(new_passengers))

                    print(f"Day {day}: Actual revenue (mean): {actual_revenue_mean:,.2f}€, Original Revenue (mean): {original_revenue_mean:,.2f}€, "
                        f"Difference: {actual_revenue_mean - original_revenue_mean:,.2f}€ ({(actual_revenue_mean - original_revenue_mean) / original_revenue_mean * 100:.2f}%)")
                    

                    # Append the results to the final results dataframe
                    final_results.loc[len(final_results)] = {
                        'optim_model': optim_model,
                        'ml_model': ml_model,
                        'delta': delta,
                        'day': day,
                        'original_passengers_mean': original_passengers_mean,
                        'original_passengers_std': original_passengers_std,
                        'original_passengers_se': original_passengers_se,
                        'new_passengers_mean': new_passengers_mean,
                        'new_passengers_std': new_passengers_std,
                        'new_passengers_se': new_passengers_se,
                        'original_revenue_mean': original_revenue_mean,
                        'original_revenue_std': original_revenue_std,
                        'original_revenue_se': original_revenue_se,
                        'optimized_revenue': best_optim_revenue[day],
                        'actual_revenue_mean': actual_revenue_mean,
                        'actual_revenue_std': actual_revenue_std,
                        'actual_revenue_se': actual_revenue_se,
                        'revenue_difference_mean': revenue_difference_mean,
                        'revenue_difference_std': revenue_difference_std,
                        'revenue_difference_se': revenue_difference_se,
                        'revenue_difference_percentage': (actual_revenue_mean - original_revenue_mean) / original_revenue_mean * 100,
                        'average_price': day_data['optimized_price'].mean(),
                        'average_original_price': day_data['original_price'].mean()
                    }

                #%%
                # Save partial final results to a CSV file
                final_results.to_csv(final_results_path, index=False)

    # %%
    # Clean up ROBIN output files
    if os.path.exists(os.path.join(path_kernel_output, 'Original_data/')) and clean_original_ROBIN_output:
        simulation_size_gb = sum(os.path.getsize(os.path.join(path_kernel_output, 'Original_data/', f)) for f in os.listdir(os.path.join(path_kernel_output, 'Original_data/')) if os.path.isfile(os.path.join(path_kernel_output, 'Original_data/', f))) / (1024 ** 3)
        print(f"Total size of original ROBIN output files: {simulation_size_gb:.2f} GB") 
        shutil.rmtree(os.path.join(path_kernel_output, 'Original_data/'))
        print(f"Removed original ROBIN output directory: {os.path.join(path_kernel_output, 'Original_data/')}")

    if os.path.exists(os.path.join(path_kernel_output, model_subpath)) and clean_modified_ROBIN_output:
        simulation_size_gb = sum(os.path.getsize(os.path.join(path_kernel_output, model_subpath, f)) for f in os.listdir(os.path.join(path_kernel_output, model_subpath)) if os.path.isfile(os.path.join(path_kernel_output, model_subpath, f))) / (1024 ** 3)
        print(f"Total size of modified ROBIN output files: {simulation_size_gb:.2f} GB") 
        shutil.rmtree(os.path.join(path_kernel_output, model_subpath))
        print(f"Removed modified ROBIN output directory: {os.path.join(path_kernel_output, model_subpath)}")

    if not keep_validation_results:
        for optim_model in optim_models.keys():
            model_path = os.path.join(path_validation_results, optim_model)
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                print(f"Removed validation results directory: {model_path}")

    # Measure total execution time
    total_time = time.time() - init_time

    print("\nResults validated susccessfully!")
    print(f"Total execution time: {total_time / 60:.2f} minutes")
    print(f"Final results saved to {final_results_path}")