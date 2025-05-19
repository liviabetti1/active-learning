import re
import os
import csv

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

    for subdir, _, files in os.walk(log_dir):
        for file in files:
            if file == 'stdout.log':
                file_path = os.path.join(subdir, file)

                # Get labeled/test R² from this file
                initial_labeled_set_size, initial_test_r2, last_labeled_set_size, last_test_r2 = get_labeled_set_test_r2(file_path)

                # Try first pattern: includes SEED
                match = re.search(
                    r'USAVARS_(\w+)_AL_(\w+)_BUDGET_(\d+)_SEED_(\d+)_IDPATH_(IDs_\d+_\w+_\w+_seed_\d+)', 
                    file_path
                )

                if match:
                    label = match.group(1)
                    method = match.group(2).lower()
                    budget = int(match.group(3))
                    al_seed = int(match.group(4))
                    initial_set_desc = match.group(5)
                else:
                    # Try second pattern: no SEED
                    match = re.search(
                        r'USAVARS_(\w+)_AL_(\w+)_BUDGET_(\d+)_IDPATH_(IDs_\d+_\w+_\w+_seed_\d+)', 
                        file_path
                    )

                    if match:
                        label = match.group(1)
                        method = match.group(2).lower()
                        budget = int(match.group(3))
                        al_seed = None  # Not applicable
                        initial_set_desc = match.group(4)
                    else:
                        print(f"Warning: Could not parse path: {file_path}")
                        continue

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

def save_to_csv():
    labels = ['POP', 'TC']
    all_data = []

    for label in labels:
        base_dir = f'output/USAVARS_{label}/ridge'

        # Only iterate through subdirectories starting with "IDs"
        for subfolder in os.listdir(base_dir):
            if subfolder.startswith("IDs"):
                log_dir = os.path.join(base_dir, subfolder)
                data = iterate_log_files_and_extract_data(log_dir)
                all_data.extend(data)

    # Sort by: Label, Budget, Method
    all_data.sort(key=lambda row: (row[1], row[2], row[4], row[3]))

    csv_filename = 'log_extracted_data_sorted.csv'
    header = ['Dataset', 'Label', 'Initial Set Description', 'Method', 'Budget', 'Seed', 'Initial Set Size', 'Initial Test R2', 'Labeled Set Size', 'Test R2']

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(all_data)

    print(f"Data has been written to {csv_filename}")

if __name__ == '__main__':
    save_to_csv()