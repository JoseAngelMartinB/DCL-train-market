#%%
import os
import pandas as pd
import numpy as np
import re
import yaml

from robin.kernel.entities import Kernel


# --- Config ---
model_subpath = 'Model_1/'
ml_model = 'tree'
days_to_test = ['2025-03-12', '2025-03-22', '2025-08-13', '2025-08-23']
delta_vect = [5, 10, 20]

path_config_supply = '../DataGenerationROBIN/data/MAD-BCN/supply_MAD-BCN_2025.yaml'
path_config_demand = '../DataGenerationROBIN/data/MAD-BCN/demand_data.yaml'
path_kernel_output = 'validation_data/ROBIN_output/'
path_validation_results = 'validation_data/'
path_aggregated_results = '../DataGenerationROBIN/data/MAD-BCN/aggregated/aggregated_MAD-BCN_2025.csv'

seed = 2025 # Initial random seed for reproducibility


# Initialize final results dataframe
final_results = pd.DataFrame(columns=[
    'model',
    'delta',
    'day',
    'original_passengers',
    'new_passengers',
    'optimized_revenue',
    'original_revenue',
    'revenue_difference',
    'revenue_difference_percentage'
])


#%%
# Load the files and execute the ROBIN simulation for each delta value
for delta in delta_vect:
    print(f"\n\nProcessing delta: {delta}")

    # Load the optimized prices
    optimized_prices = pd.DataFrame()
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
        # Concatenate the data from the best file
        df = pd.read_csv(os.path.join(os.path.join(model_subpath, f"results_{ml_model}"), best_file))
        df['day'] = day
        optimized_prices = pd.concat([optimized_prices, df], ignore_index=True)

    if optimized_prices.empty:
        raise ValueError("No optimized prices found for the specified days.")


    #%%
    # Modify the supply data to include the optimized prices
    with open(path_config_supply, 'r') as file:
        supply_config = yaml.safe_load(file)
    supply_config['service'] = [item for item in supply_config['service'] if item['date'] in days_to_test]

    # Update the prices in the supply configuration
    for item in supply_config['service']:
        # Find the corresponding optimized price for this train
        train_id = item['id']
        price_row = optimized_prices[optimized_prices['train_idx'] == train_id]
        if not price_row.empty:
            # Update the price in the supply configuration
            for od in item['origin_destination_tuples']:
                for seat in od['seats']:
                    # Store price as a two-decimal float
                    seat['price'] = float(np.round(price_row['optimized_price'].values[0], 2))
        else:
            print(f"Warning: No optimized price found for train {train_id} on day {item['date']}.")

    # Save the modified supply configuration
    output_supply_path = os.path.join(path_kernel_output + model_subpath)
    os.makedirs(output_supply_path, exist_ok=True)
    output_supply_file = os.path.join(output_supply_path, 'supply_MAD-BCN_2025.yaml')
    with open(output_supply_file, 'w') as file:
        yaml.dump(supply_config, file, default_flow_style=False, sort_keys=False)

    print(f"New supply configuration saved to {output_supply_file}")


    # %%
    # Execute the ROBIN simulation with the modified supply configuration
    print("Starting ROBIN simulation...")
    kernel = Kernel(
        output_supply_file,
        path_config_demand,
        seed)

    robin_output_path = os.path.join(path_kernel_output, model_subpath, f'kernel_output_delta_{delta}.csv')
    services = kernel.simulate(
        robin_output_path,
        departure_time_hard_restriction=False,
        calculate_global_utility=False)
    print(f"ROBIN simulation completed. Output saved to {robin_output_path}")


    #%%
    # Pos-process the ROBIN output
    robin_output = pd.read_csv(robin_output_path)
    # Remove all the rows where arrival_day is not in the days_to_test
    robin_output = robin_output[robin_output['arrival_day'].isin(days_to_test)]

    # Aggregate the data: 
    # Copy the optimized_prices dataset and add a new column with the expected passengers
    agg_dataset = optimized_prices.copy()
    robin_output_grouped = robin_output.groupby(['service']).size()
    agg_dataset['expected_passengers'] = agg_dataset['train_idx'].map(robin_output_grouped).fillna(0).astype(int)

    # Limit the capacity of each train to the maximum capacity defined
    agg_dataset['passengers'] = agg_dataset['expected_passengers'].clip(upper=agg_dataset['capacity'])

    # Save the aggregated dataset
    model_name = model_subpath.split('/')[0]
    os.makedirs(os.path.join(path_validation_results, model_name), exist_ok=True)
    output_agg_path = os.path.join(path_validation_results, model_name, f"validation_delta_{delta}.csv")
    agg_dataset.to_csv(output_agg_path, index=False)
    print(f"Aggregated dataset saved to {output_agg_path}")

    # %%
    # Compute the total revenue per day for Renfe [AVE + AVLO]
    original_agg_data = pd.read_csv(path_aggregated_results)
    original_agg_data['original_passengers'] = original_agg_data['passengers']

    agg_dataset_RENFE = agg_dataset[(agg_dataset['train_type'] == 'AVE') | (agg_dataset['train_type'] == 'AVLO')]
    agg_dataset_RENFE = agg_dataset_RENFE.merge(original_agg_data[['service_id', 'original_passengers']],
                                                left_on='train_idx', right_on='service_id', how='left')
    agg_dataset_RENFE['original_passengers'] = agg_dataset_RENFE['original_passengers'].clip(upper=agg_dataset_RENFE['capacity'])

    agg_dataset_RENFE['original_revenue'] = agg_dataset_RENFE['original_passengers'] * agg_dataset_RENFE['original_price']
    agg_dataset_RENFE['revenue'] = agg_dataset_RENFE['passengers'] * agg_dataset_RENFE['optimized_price']

    # Save the Renfe revenue data
    renfe_revenue_path = os.path.join(path_validation_results, model_name, f"validation_delta_{delta}_renfe_revenue.csv")
    agg_dataset_RENFE.to_csv(renfe_revenue_path, index=False)
    print(f"Renfe revenue data saved to {renfe_revenue_path}")

    print("Total revenue for Renfe (AVE + AVLO):")
    for day in days_to_test:
        day_data = agg_dataset_RENFE[agg_dataset_RENFE['day'] == day]
        total_revenue = day_data['revenue'].sum()
        total_original_revenue = day_data['original_revenue'].sum()
        print(f"Day {day}: Optimized Revenue: {total_revenue:.2f}€, Original Revenue: {total_original_revenue:.2f}€, "
            f"Difference: {total_revenue - total_original_revenue:.2f}€ ({(total_revenue - total_original_revenue) / total_original_revenue * 100:.2f}%)")

        # Append the results to the final results dataframe
        final_results.loc[len(final_results)] = {
            'model': model_name,
            'delta': delta,
            'day': day,
            'original_passengers': day_data['original_passengers'].sum(),
            'new_passengers': day_data['passengers'].sum(),
            'optimized_revenue': total_revenue,
            'original_revenue': total_original_revenue,
            'revenue_difference': total_revenue - total_original_revenue,
            'revenue_difference_percentage': (total_revenue - total_original_revenue) / total_original_revenue * 100
        }

# Save the final results to a CSV file
final_results_path = os.path.join(path_validation_results, f"final_results_{ml_model}.csv")
final_results.to_csv(final_results_path, index=False)
print(f"Final results saved to {final_results_path}")


# %%
