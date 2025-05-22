import sys
import re
import os
import csv

from compute_sample_cost import compute_total_sample_cost

def get_labeled_set_test_r2(file_path):
    with open(file_path, 'r') as f:
        log_data = f.read()

    # Find all labeled set sizes (including "New Labeled Set" or plain "Labeled Set")
    labeled_set_sizes = re.findall(r'(New )?Labeled Set: (\d+)', log_data)

    # Find all Test R² values (allowing negatives)
    test_r2_values = re.findall(r'Test Accuracy (-?[0-9]*\.?[0-9]+)', log_data)

    # Extract initial and final labeled set sizes
    initial_labeled_set_size = int(labeled_set_sizes[0][1]) if labeled_set_sizes else None
    last_labeled_set_size = int(labeled_set_sizes[-1][1]) if labeled_set_sizes else None

    # Extract initial and final test R² values
    initial_test_r2 = float(test_r2_values[0]) if test_r2_values else None
    last_test_r2 = float(test_r2_values[-1]) if test_r2_values else None

    print(f"Initial Labeled Set size: {initial_labeled_set_size}, Initial Test R²: {initial_test_r2}")
    print(f"Final Labeled Set size: {last_labeled_set_size}, Final Test R²: {last_test_r2}")

    return initial_labeled_set_size, initial_test_r2, last_labeled_set_size, last_test_r2

def iterate_log_files_and_extract_data(dataset_name, task, initial_set_str):
    """Iterate through new-style experiment directories and extract relevant data."""
    data_rows = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    log_dir = os.path.join(project_root, 'output', dataset_name, task, initial_set_str)

    if not os.path.exists(log_dir):
        print(f"{log_dir} does not exist.")
        return None

    # Traverse all directories under log_dir
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file == 'stdout.log':
                file_path = os.path.join(root, file)
            else:
                continue

            # Try to match the expected directory structure
            parts = file_path.split(os.sep)

            try:         
                method = parts[8].lower()                              # e.g., 'random', 'leverage', etc.
                budget = int(parts[9].split('_')[1])                   # e.g., 'budget_500' → 500
                al_seed = int(parts[10].split('_')[1])                  # e.g., 'seed_0' → 0
            except Exception as e:
                print(f"Skipping path {file_path} due to parse error: {e}")
                continue

            try:
                initial_labeled_set_size, initial_test_r2, last_labeled_set_size, last_test_r2 = get_labeled_set_test_r2(file_path)

                data_rows.append([
                    method,
                    al_seed,
                    initial_labeled_set_size,
                    initial_test_r2,
                    budget,
                    last_test_r2
                ])
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return data_rows

def iterate_log_files_and_extract_data_with_cost(log_dir, cost_path, label):
    """Iterate through new-style experiment directories and extract relevant data."""
    data_rows = []
    
    for root, dirs, _ in os.walk(log_dir):
        for dir in dirs:
            exp_path = os.path.join(root, dir)
            if not os.path.isdir(exp_path):
                continue

            # Try to match the expected directory structure
            parts = os.path.relpath(exp_path, log_dir).split(os.sep)
            if len(parts) < 4:
                continue  # Not a valid experiment path

            try:
                initial_set_desc = parts[0]                            # e.g., 'IDs_clustered_100_counties_10_radius_seed_0'
                method = parts[1].lower()                              # e.g., 'random', 'leverage', etc.
                budget = int(parts[2].split('_')[1])                   # e.g., 'budget_500' → 500
                al_seed = int(parts[3].split('_')[1])                  # e.g., 'seed_0' → 0
            except Exception as e:
                print(f"Skipping path {exp_path} due to parse error: {e}")
                continue

            file_path = os.path.join(exp_path, "stdout.log")
            if not os.path.isfile(file_path):
                continue

            try:
                initial_labeled_set_size, initial_test_r2, last_labeled_set_size, last_test_r2 = get_labeled_set_test_r2(file_path)

                activeset_path = os.path.join(exp_path, "episode_0", "activeSet.npy")
                if label == 'TC':
                    full_label = 'treecover'
                elif label == 'POP':
                    full_label = 'population'
                else:
                    full_label = label.lower()

                sample_cost = compute_total_sample_cost(activeset_path, full_label, cost_path)

                data_rows.append([
                    'USAVARS',
                    label,
                    initial_set_desc,
                    method,
                    budget,
                    al_seed,
                    initial_labeled_set_size,
                    initial_test_r2,
                    last_labeled_set_size,
                    last_test_r2,
                    sample_cost
                ])
            except Exception as e:
                print(f"Error processing {exp_path}: {e}")

    return data_rows

def save_to_csv():
    dataset_name = "USAVARS"
    labels = ['treecover', 'population']

    for task in labels:
        for type_str in ['density', 'clustered']:
            for num_counties in [25, 50, 75, 100]:
                for radius in [10]:
                    initial_set_str = f'{type_str}_{num_counties}_counties_{radius}_radius'

                    data = iterate_log_files_and_extract_data(dataset_name, task, initial_set_str)
                    if data is None:
                        continue

                    # Sort by: Label, Budget, Method
                    data.sort(key=lambda row: (row[0], row[4]))

                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
                    
                    csv_dir = f'results/csv/{dataset_name}/{task}/{initial_set_str}'
                    os.makedirs(csv_dir, exist_ok=True)

                    csv_filepath = os.path.join(project_root, csv_dir, 'results.csv')


                    header = ['Method', 'Seed', 'Initial Set Size', 'Initial Test R2', 'Budget', 'Test R2']

                    with open(csv_filepath, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(header)
                        writer.writerows(data)

                    print(f"Data has been written to {csv_filepath}")

if __name__ == '__main__':
    save_to_csv()