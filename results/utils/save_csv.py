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

def iterate_log_files_and_extract_data(log_dir):
    """Iterate through log files and extract relevant data."""
    data_rows = []

    # Pattern for main ID subdirectories
    dir_pattern = re.compile(r'IDs_(clustered|density)_(\d+)_counties_(\d+)_radius_seed_(\d+)')

    # Pattern for AL run subdirectory names
    al_subdir_pattern = re.compile(
        r'USAVARS_(\w+)_AL_(\w+)_BUDGET_(\d+)_SEED_(\d+)_IDPATH_(IDs_\w+_\d+_counties_\d+_radius_seed_\d+)'
    )

    for root, dirs, _ in os.walk(log_dir):
        for dir in dirs:
            if dir_pattern.fullmatch(dir):
                id_dir = os.path.join(root, dir)

                # Now iterate over AL run directories inside this ID directory
                for al_run_dir in os.listdir(id_dir):
                    al_run_path = os.path.join(id_dir, al_run_dir)

                    if os.path.isdir(al_run_path):
                        match = al_subdir_pattern.fullmatch(al_run_dir)
                        if match:
                            label = match.group(1)            # full directory name
                            method = match.group(2).lower()  # clustered or density
                            budget = int(match.group(3))    
                            al_seed = int(match.group(4))  
                            initial_set_desc = match.group(5)

                        file_path = os.path.join(al_run_path, "stdout.log")
                        initial_labeled_set_size, initial_test_r2, last_labeled_set_size, last_test_r2 = get_labeled_set_test_r2(file_path)

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
                            last_test_r2
                        ])

    return data_rows

def iterate_log_files_and_extract_data_with_cost(log_dir, cost_path, density=False, label='TC'):
    """Iterate through log files and extract relevant data."""
    data_rows = []

    # Pattern for main ID subdirectories
    if density:
        dir_pattern = re.compile(r'IDs_density_(\d+)_counties_(\d+)_radius_seed_(\d+)')
    else:
        dir_pattern = re.compile(r'IDs_clustered_(\d+)_counties_(\d+)_radius_seed_(\d+)')

    # Pattern for AL run subdirectory names
    al_subdir_pattern = re.compile(
        r'USAVARS_(\w+)_AL_(\w+)_BUDGET_(\d+)_SEED_(\d+)_IDPATH_(IDs_\w+_\d+_counties_\d+_radius_seed_\d+)(.*)'
    )

    for root, dirs, _ in os.walk(log_dir):
        for dir in dirs:
            if dir_pattern.fullmatch(dir):
                id_dir = os.path.join(root, dir)

                # Now iterate over AL run directories inside this ID directory
                for al_run_dir in os.listdir(id_dir):
                    al_run_path = os.path.join(id_dir, al_run_dir)

                    if os.path.isdir(al_run_path):
                        match = al_subdir_pattern.fullmatch(al_run_dir)
                        if match:
                            label = match.group(1)            # full directory name
                            method = match.group(2).lower()  # clustered or density
                            budget = int(match.group(3))    
                            al_seed = int(match.group(4))  
                            initial_set_desc = match.group(5)

                        file_path = os.path.join(al_run_path, "stdout.log")
                        initial_labeled_set_size, initial_test_r2, last_labeled_set_size, last_test_r2 = get_labeled_set_test_r2(file_path)

                        activeset_path = os.path.join(al_run_path, "episode_0", "activeSet.npy")
                        if label == 'TC':
                            full_label = 'treecover'
                        elif label == 'POP':
                            full_label = 'population'
                        try:
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
                            print(e)

    return data_rows

def save_to_csv():
    labels = ['TC']
    density = False
    type_str = "density" if density else "clustered"
    all_data = []

    for label in labels:
        log_dir = f'output/USAVARS_{label}/ridge'

        full_label = 'treecover' if label == 'TC' else 'population'
        cost_path = f"/home/libe2152/deep-al/usavars/cost_{full_label}/cost_{type_str}_500_counties_r1_10_r2_20_seed_42_cost_1_vs_2.pkl"

        data = iterate_log_files_and_extract_data_with_cost(log_dir, cost_path, density=density, label=label)
        all_data.extend(data)

    # Sort by: Label, Budget, Method
    all_data.sort(key=lambda row: (row[1], row[2], row[4], row[3]))

    csv_filename = f'log_extracted_data_sorted_{type_str}_with_cost_cost_1_vs_2.csv'
    header = ['Dataset', 'Label', 'Initial Set Description', 'Method', 'Budget', 'Seed', 'Initial Set Size', 'Initial Test R2', 'Labeled Set Size', 'Test R2', 'Sample Cost']

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(all_data)

    print(f"Data has been written to {csv_filename}")

if __name__ == '__main__':
    save_to_csv()