import sys
import re
import os
import csv

from compute_sample_cost import compute_total_sample_cost

def get_labeled_set_test_r2(file_path, cost_aware=False):
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

    if cost_aware:
        total_cost = re.findall(r'Total Cost of New Labeled Set: (-?[0-9]*\.?[0-9]+)', log_data)
        total_cost = float(total_cost[0]) if total_cost else None

        return initial_labeled_set_size, initial_test_r2, last_labeled_set_size, last_test_r2, total_cost

    return initial_labeled_set_size, initial_test_r2, last_labeled_set_size, last_test_r2

def iterate_log_files_and_extract_data(dataset_name, task, initial_set_str, cost_aware=False):
    """Iterate through new-style experiment directories and extract relevant data."""
    data_rows = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    log_dir = os.path.join(project_root, 'output', dataset_name, task, initial_set_str, 'cost_aware') if cost_aware else os.path.join(project_root, 'output', dataset_name, task, initial_set_str)

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

            if not cost_aware and "cost_aware" in parts:
                continue

            try:         
                if "representative" in parts:
                    rep_idx = parts.index("representative")
                    rep_type = parts[rep_idx + 1]  # 'nlcd' or 'state'
                    method = f"representative_{rep_type}"
                    budget = int(parts[rep_idx + 2].split('_')[1])  # budget_#
                    al_seed = int(parts[rep_idx + 3].split('_')[1])  # seed_#
                else:
                    cost_func = parts[9].lower() if cost_aware else None
                    method = parts[10].lower() if cost_aware else parts[8].lower()
                    budget = int(parts[11].split('_')[1]) if cost_aware else int(parts[9].split('_')[1])
                    al_seed = int(parts[12].split('_')[1]) if cost_aware else int(parts[10].split('_')[1])
            except Exception as e:
                from IPython import embed; embed()
                print(f"Skipping path {file_path} due to parse error: {e}")
                continue

            try:
                if cost_aware:
                    initial_labeled_set_size, initial_test_r2, last_labeled_set_size, last_test_r2, total_cost = get_labeled_set_test_r2(file_path, cost_aware=cost_aware)

                    data_rows.append([
                        method,
                        al_seed,
                        initial_labeled_set_size,
                        initial_test_r2,
                        budget,
                        last_labeled_set_size,
                        last_test_r2,
                        cost_func,
                        total_cost
                    ])
                else:
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

def save_to_csv(dataset_name, labels, cost_aware):
    for task in labels:
        for type_str in ['clustered', 'density']:
            for num_counties in [25, 50, 75, 100, 125, 150, 175, 200]:
                for radius in [10]:
                    initial_set_str = f'{type_str}_{num_counties}_counties_{radius}_radius'

                    data = iterate_log_files_and_extract_data(dataset_name, task, initial_set_str, cost_aware=cost_aware)
                    if data is None:
                        continue

                    # Sort by: Label, Budget, Method
                    data.sort(key=lambda row: (row[0], row[4]))

                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

                    csv_dir = os.path.join(project_root, f'results/csv/{dataset_name}/{task}/{initial_set_str}/cost_aware') if cost_aware else os.path.join(project_root, f'results/csv/{dataset_name}/{task}/{initial_set_str}')
                    os.makedirs(csv_dir, exist_ok=True)

                    csv_filepath = os.path.join(csv_dir, 'results.csv')

                    header = ['Method', 'Seed', 'Initial Set Size', 'Initial Test R2', 'Budget', 'Labeled Set Size', 'Test R2', 'Cost Function', 'Total Cost'] if cost_aware else ['Method', 'Seed', 'Initial Set Size', 'Initial Test R2', 'Budget', 'Test R2']

                    with open(csv_filepath, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(header)
                        writer.writerows(data)

                    print(f"Data has been written to {csv_filepath}")

if __name__ == '__main__':
    dataset_name = "USAVARS"
    labels = ['treecover', 'population']
    cost_aware = True


    save_to_csv(dataset_name, labels, cost_aware)