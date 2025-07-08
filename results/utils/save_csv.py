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
    import os
    data_rows = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    base_dir = os.path.join(project_root, 'output', dataset_name, task, initial_set_str)

    dirs_to_check = []

    if cost_aware:
        cost_aware_dir = os.path.join(base_dir, 'cost_aware')
        if not os.path.exists(cost_aware_dir):
            print(f"{cost_aware_dir} does not exist.")
            return None
        dirs_to_check.append(cost_aware_dir)
    else:
        dirs_to_check.append(base_dir)
        uniform_dir = os.path.join(base_dir, 'cost_aware', 'uniform')
        if os.path.exists(uniform_dir):
            dirs_to_check.append(uniform_dir)

    for log_dir in dirs_to_check:
        for root, _, files in os.walk(log_dir):
            for file in files:
                if file != 'stdout.log':
                    continue

                if not cost_aware:
                    if "cost_aware" in root and "uniform" not in root:
                        continue

                file_path = os.path.join(root, file)
                parts = file_path.split(os.sep)

                try:
                    if "cost_aware" in parts:
                        cost_aware_flag = True

                        cost_aware_idx = parts.index('cost_aware')
                        cost_func = parts[cost_aware_idx + 1].lower()
                        method = parts[cost_aware_idx + 2].lower()

                        possible_group_type = parts[cost_aware_idx + 3].lower()
                        if possible_group_type in ['nlcd', 'state']:
                            group_type = possible_group_type
                            budget_idx = cost_aware_idx + 4
                            seed_idx = cost_aware_idx + 5
                        else:
                            group_type = None
                            budget_idx = cost_aware_idx + 3
                            seed_idx = cost_aware_idx + 4

                        budget = int(parts[budget_idx].split('_')[1])
                        al_seed = int(parts[seed_idx].split('_')[1])
                    else:
                        cost_aware_flag = False
                        cost_func = None

                        # Budget and seed folders:
                        budget_idx = -3
                        seed_idx = -2

                        # Start by assuming the common case: method with optional group_type before it
                        possible_group_type = parts[-4].lower()
                        if possible_group_type in ['nlcd', 'state']:
                            group_type = possible_group_type
                            method = parts[-5].lower()
                        else:
                            group_type = None
                            method = parts[-4].lower()

                        budget = int(parts[budget_idx].split('_')[1])
                        al_seed = int(parts[seed_idx].split('_')[1])

                        budget = int(parts[budget_idx].split('_')[1])
                        al_seed = int(parts[seed_idx].split('_')[1])
                except Exception as e:
                    print(f"Skipping {file_path} due to parse error: {e}")
                    continue

                try:
                    # Append group_type to method if applicable
                    if group_type:
                        method = f"{method}_{group_type}"

                    if cost_aware_flag and cost_func != 'uniform':
                        initial_labeled_set_size, initial_test_r2, last_labeled_set_size, last_test_r2, total_cost = get_labeled_set_test_r2(
                            file_path, cost_aware=True
                        )

                        row = [
                            method,
                            al_seed,
                            initial_labeled_set_size,
                            initial_test_r2,
                            budget,
                            last_labeled_set_size,
                            last_test_r2,
                            cost_func,
                            total_cost
                        ]
                    else:
                        initial_labeled_set_size, initial_test_r2, last_labeled_set_size, last_test_r2 = get_labeled_set_test_r2(
                            file_path, cost_aware=False
                        )

                        row = [
                            method,
                            al_seed,
                            initial_labeled_set_size,
                            initial_test_r2,
                            budget,
                            last_test_r2
                        ]

                    data_rows.append(row)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

    return data_rows if data_rows else None


def save_to_csv(dataset_name, labels, cost_aware):
    points_per_cluster_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    desired_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    top_urban_points = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    for task in labels:
        # --- Original clusters ---
        for num_points_per_cluster in points_per_cluster_list:
            for desired_size in desired_sizes:
                initial_set_str = f"state_strata_county_clusters_{num_points_per_cluster}_points_per_cluster_{desired_size}_size"
                _write_csv_from_logs(dataset_name, task, initial_set_str, cost_aware)

        # --- Top 50 urban areas ---
        for num in top_urban_points:
            initial_set_str = f"top50_urban_areas_{num}_points"
            _write_csv_from_logs(dataset_name, task, initial_set_str, cost_aware)


def _write_csv_from_logs(dataset_name, task, initial_set_str, cost_aware):
    data = iterate_log_files_and_extract_data(
        dataset_name,
        task,
        initial_set_str,
        cost_aware=cost_aware
    )
    if data is None:
        print(f"No data found for {initial_set_str} ({task})")
        return

    # Sort by: Method, Budget (assuming method = row[0], budget = row[4])
    data.sort(key=lambda row: (row[0], row[4]))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    csv_dir = os.path.join(
        project_root,
        f'results/csv/{dataset_name}/{task}/{initial_set_str}/cost_aware'
    ) if cost_aware else os.path.join(
        project_root,
        f'results/csv/{dataset_name}/{task}/{initial_set_str}'
    )
    os.makedirs(csv_dir, exist_ok=True)

    csv_filepath = os.path.join(csv_dir, 'results.csv')

    header = (
        ['Method', 'Seed', 'Initial Set Size', 'Initial Test R2', 'Budget',
         'Labeled Set Size', 'Test R2', 'Cost Function', 'Total Cost']
        if cost_aware else
        ['Method', 'Seed', 'Initial Set Size', 'Initial Test R2', 'Budget', 'Test R2']
    )

    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

    print(f"Data written to {csv_filepath}")

if __name__ == '__main__':
    dataset_name = "USAVARS"
    labels = ['population', 'treecover', 'income']
    cost_aware = False


    save_to_csv(dataset_name, labels, cost_aware)