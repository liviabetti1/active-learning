import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

from pycls.datasets.usavars import USAVars

label_abbrev = {
    'POP': "population",
    'TC': "treecover"
}

def load_lat_lons(label, file_path):
    indices = np.load(file_path, allow_pickle=True)
    dataset = USAVars(root='/share/usavars', isTrain=True, label=label_abbrev[label])
    return [dataset[i][2] for i in indices]

def plot_labeled(label, lset_path, activeset_path, method_name, active_learning_method, al_seed, **kwargs):
    # Load the two sets of lat/lons
    lset_latlons = load_lat_lons(label, lset_path)
    active_latlons = load_lat_lons(label, activeset_path)

    # Unpack coordinates
    lset_lats, lset_lons = zip(*lset_latlons)
    active_lats, active_lons = zip(*active_latlons)

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 10))

    world = gpd.read_file("../../country_boundaries/ne_110m_admin_1_states_provinces.shp", engine="pyogrio")

    # Filter to contiguous US
    exclude_states = ["Alaska", "Hawaii", "Puerto Rico"]
    contiguous_us = world[~world["name"].isin(exclude_states)]
    contiguous_outline = contiguous_us.dissolve()
    contiguous_outline.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=4, alpha=0.8)

    # Plot both sets
    ax.scatter(lset_lons, lset_lats, color='#d62728', s=5, alpha=0.9, label=f'lSet ({len(lset_lons)} Initial Points)', zorder=5)
    ax.scatter(active_lons, active_lats, color='black', marker='s', s=10, alpha=0.8, label=f'activeSet ({len(active_lons)} Queried Points)', zorder=6)

    # Set title, labels, and legend
    ax.set_title(f'Geospatial Points for Label: {label_abbrev[label]}\n'
                 f'Initial Set: {method_name}\n'
                 f'Active Learning: {active_learning_method} (Seed {al_seed})', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    ax.legend(loc='lower left', fontsize=10, title='Legend', title_fontsize=11, frameon=True)

    # Additional formatting
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Save the plot with a specific filename
    plot_filename = f'plots/usavars_points_{method_name}_{label}_active_learning_{active_learning_method.lower()}_seed_{al_seed}.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {plot_filename}")

if __name__ == "__main__":
    initial_seed = 42
    budget = 100
    al_seed = 1

    initial_set_method_list = ['100_counties_5_points',
                               '500_centers_5000m_radius',
                               '500_counties_4_points']
    
    active_learning_method_list = ['RANDOM',
                                   'TYPICLUST',
                                   'INVERSETYPICLUST',
                                   'ENSEMBLE_VARIANCE']

    for label in ['TC']:
        for initial_set_method in initial_set_method_list:
            for active_learning_method in active_learning_method_list:
                for al_seed in [1]:
                    try:
                        exp_dir = f'/home/libe2152/deep-al/output/USAVARS_{label}/ridge/IDs_{initial_set_method}_seed_{initial_seed}/USAVARS_{label}_AL_{active_learning_method}_BUDGET_{budget}_SEED_{al_seed}_IDPATH_IDs_{initial_set_method}_seed_{initial_seed}.pkl/episode_0/'
                        
                        lset_path = f'{exp_dir}lSet.npy'
                        activeset_path = f'{exp_dir}activeSet.npy'
                        plot_labeled(label, lset_path, activeset_path, initial_set_method, active_learning_method, al_seed)
                    except Exception as e:
                        try: 
                            exp_dir = f'/home/libe2152/deep-al/output/USAVARS_{label}/ridge/IDs_{initial_set_method}_seed_{initial_seed}/USAVARS_{label}_AL_{active_learning_method}_BUDGET_{budget}_IDPATH_IDs_{initial_set_method}_seed_{initial_seed}.pkl/episode_0/'
                            lset_path = f'{exp_dir}lSet.npy'
                            activeset_path = f'{exp_dir}activeSet.npy'
                            plot_labeled(label, lset_path, activeset_path, initial_set_method, active_learning_method, al_seed)
                        except Exception as e:
                            print(e)
