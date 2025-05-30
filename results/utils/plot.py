import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm

sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

custom_palette = {
    "random": "red",
    "representative_nlcd": "blue",
    "representative_state": "green",
    "typiclust": "#9467bd",         # purple from matplotlib default
    "inversetypiclust": "#ff7f0e"   # orange from matplotlib default
}

def nice_initial_set_desc(desc):
    """
    Turn something like 'IDs_100_counties_5_points_seed_42'
    or 'IDs_500_centers_5000m_radius_seed_42'
    into '100 Counties, 5 Points (Seed 42)'
    or '500 Centers, 5000m Radius (Seed 42)'
    """
    # Match description like 'IDs_100_counties_5_points_seed_42'
    match1 = re.match(r"IDs_(\d+)_([a-zA-Z]+)_(\d+)_([a-zA-Z]+)_seed_(\d+)", desc)
    # Match description like 'IDs_500_centers_5000m_radius_seed_42'
    match2 = re.match(r'IDs_(\d+)_([a-zA-Z]+)_(\d+)([a-zA-Z0-9]+)_radius_seed_(\d+)', desc)
    # Match description like 'IDs_clustered_500_counties_10_radius_seed_42'
    match3 = re.match(r'IDs_([a-zA-Z]+)_(\d+)_([a-zA-Z]+)_(\d+)_radius_seed_(\d+)', desc)
    # Match 'clustered|density_num_counties_num_radis
    match4 = re.match(r'(clustered|density)_(\d+)_([a-zA-Z]+)_(\d+)_radius', desc)

    if match1:
        count, region, number, unit, seed = match1.groups()
        return f"{count} {region.capitalize()}, {number} {unit.capitalize()} [Seed {seed}]"
    elif match2:
        count, region, number, unit, seed = match2.groups()
        return f"{count} {region.capitalize()}, {number}{unit} radius [Seed {seed}]"
    elif match3:
        type_str, count, region, radius, seed = match3.groups()
        return f"{type_str} {count} {region.capitalize()}, {radius} km radius [Seed {seed}]"
    elif match4:
        type_str, count, region, radius = match4.groups()
        return f"{type_str} {count} {region.capitalize()}, {radius} km radius"

    return desc

def plot_r2_vs_budget(
    df,
    label,
    init_set_str,
    save_path,
    budget_bound=np.inf,
    methods_to_include=None,
    log=True,
    jitter=True,
    cost_aware=False
):
    if cost_aware:
        jitter = False

    if methods_to_include:
        df = df[df["Method"].isin(methods_to_include)]

    x_val = "Total Cost" if cost_aware else "Budget"
    df = df[df[x_val] <= budget_bound]

    if cost_aware:
        # Use raw values directly
        plot_df = df.copy()
    else:
        # Group and aggregate separately
        grouped = df.groupby(["Method", x_val, "Initial Test R2", "Initial Set Size"])["Test R2"]
        r2_mean = grouped.mean().reset_index(name="Mean R2")
        r2_std = grouped.std().reset_index(name="Std R2")

        # Combine
        plot_df = r2_mean.merge(r2_std, on=["Method", x_val, "Initial Test R2", "Initial Set Size"])

    if not log:
        methods = plot_df["Method"].unique()
        new_rows = []
        for method in methods:
            initial_r2_value = plot_df["Initial Test R2"].dropna().iloc[0]  # Assume all same
            initial_set_size = plot_df["Initial Set Size"].dropna().iloc[0]
            x = initial_set_size if cost_aware else 0
            row = {
                "Method": method,
                x_val: x,
                "Initial Test R2": initial_r2_value,
                "Initial Set Size": initial_set_size,
            }
            if cost_aware:
                row["Test R2"] = initial_r2_value
            else:
                row["Mean R2"] = initial_r2_value
                row["Std R2"] = 0
            new_rows.append(row)

        plot_df = pd.concat([plot_df, pd.DataFrame(new_rows)], ignore_index=True)

    # Method ordering
    method_order = ["random", "greedycost"] if cost_aware else ["random", "typiclust", "inversetypiclust"]
    plot_df["Method"] = pd.Categorical(plot_df["Method"], categories=method_order, ordered=True)
    plot_df = plot_df.sort_values(by=["Method", x_val])

    # Jitter setup
    jitter_strength = 0.03
    Budget_str = "Budget"
    if jitter:
        method_to_jitter = {
            method: i * jitter_strength - jitter_strength for i, method in enumerate(plot_df["Method"].cat.categories)
        }
        plot_df["Budget_jittered"] = plot_df.apply(
            lambda row: row["Budget"] + row['Budget'] * method_to_jitter[row["Method"]], axis=1
        )
        Budget_str = "Budget_jittered"
    x_val = x_val if cost_aware else Budget_str

    # Plotting
    plt.figure(figsize=(12, 7))
    sns.set(style="whitegrid")

    if cost_aware:
        y_val = "Test R2"
    else:
        y_val = "Mean R2"

    ax = sns.scatterplot(
        data=plot_df,
        x=x_val,
        y=y_val,
        hue="Method",
        style="Method",
        markers=True,
        palette=custom_palette,
        alpha=0.9
    )

    if not cost_aware:
        for method in plot_df["Method"].cat.categories:
            method_df = plot_df[plot_df["Method"] == method]
            color = custom_palette.get(method, "black")

            plt.errorbar(
                x=method_df[x_val],
                y=method_df["Mean R2"],
                yerr=method_df["Std R2"],
                fmt='none',               # No connecting lines or markers
                capsize=4,
                elinewidth=1.5,
                color=color
            )

    # Update init set description
    nice_init_set_desc = nice_initial_set_desc(init_set_str)

    # Adjust title and labels with improved font sizes
    ax.set_title(f"Test R² vs Budget\nLabel = {label}\nInit Set: {nice_init_set_desc}", fontsize=18, fontweight='bold')
    ax.set_xlabel(x_val, fontsize=15)
    ax.set_ylabel("Test R²", fontsize=15)

    # Adjust legend outside the plot to avoid overlap
    ax.legend(title="Methods", loc='upper left', fontsize=12)

    # Apply y-limits if specified
    # plt.ylim(0.1, 0.5)

    ax.grid(True)

    if log:
        ax.set_xscale('log')

    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def plot_r2_vs_num_samples(
    df_list,
    label,
    init_set_str,
    save_path,
    budget_bound=np.inf,
    methods_to_include=None,
    log=False,
    jitter=True
):

    plot_df_list = []
    plt.figure(figsize=(36, 30))

    for df in df_list:
        if methods_to_include:
            df = df[df["Method"].isin(methods_to_include)]

        df = df[df['Budget'] <= budget_bound]

        # Group and aggregate separately
        grouped= df.groupby(["Method", "Budget", "Initial Test R2", "Initial Set Size"])["Test R2"]
        r2_mean= grouped.mean().reset_index(name="Mean R2")
        r2_std = grouped.std().reset_index(name="Std R2")

        # Combine
        plot_df = r2_mean.merge(r2_std, on=["Method", "Budget", "Initial Test R2", "Initial Set Size"])

        if not log: #budget 0 won't show up on log scale
            # Create new rows for Budget=0 with Initial Test R2 values
            methods = plot_df["Method"].unique()
            new_rows = []
            for method in methods:
                initial_r2_value = plot_df["Initial Test R2"].dropna().iloc[0] #just take any bc they're all the same
                initial_set_size = plot_df["Initial Set Size"].dropna().iloc[0]
                if initial_r2_value is not None:
                    new_rows.append({
                        "Method": method,
                        "Budget": 0,
                        "Mean R2": initial_r2_value,
                        "Std R2": 0,  # Std R2 at 0 budget assumed 0 or NaN
                        "Initial Test R2": initial_r2_value,  # if this column exists
                        "Initial Set Size": initial_set_size
                    })
            plot_df = pd.concat([plot_df, pd.DataFrame(new_rows)], ignore_index=True)

        method_order = methods_to_include if methods_to_include is not None else ["random", "typiclust", "inversetypiclust"]
        plot_df["Method"] = pd.Categorical(plot_df["Method"], categories=method_order, ordered=True)
        plot_df = plot_df.sort_values(by=["Method", "Budget"])

        jitter_strength = 0.03
        Budget_str = "Budget"
        if jitter:
            method_to_jitter = {
                method: i * jitter_strength - jitter_strength for i, method in enumerate(plot_df["Method"].cat.categories)
            }
            plot_df["Budget_jittered"] = plot_df.apply(lambda row: row["Budget"] + row['Budget']*method_to_jitter[row["Method"]], axis=1)
            Budget_str = "Budget_jittered"

        # Plotting
        plot_df["Num Samples jittered"] = plot_df["Initial Set Size"] + plot_df["Budget_jittered"]

        plot_df_list.append(plot_df)

        #plot_df = pd.concat(plot_df_list, ignore_index=True)

        zero_budget_df = plot_df[plot_df["Budget"] == 0]
        ax = sns.lineplot(
            data=plot_df,
            x="Num Samples jittered",
            y="Mean R2",
            hue="Method",
            style="Method",
            markers=True,
            palette=custom_palette,
            alpha=0.9
        )
        color_palette = sns.color_palette("Set1")
        plt.scatter(
            zero_budget_df["Num Samples jittered"],
            zero_budget_df["Mean R2"],
            color="black",
            marker="x",
            s=200,  # size of the marker
        )

        for method in plot_df["Method"].cat.categories:
            method_df = plot_df[plot_df["Method"] == method]
            color = custom_palette.get(method, "black")

            x = method_df["Num Samples jittered"]
            y = method_df["Mean R2"]
            yerr = method_df["Std R2"]

            plt.fill_between(
                x,
                y - yerr,
                y + yerr,
                color=color,
                alpha=0.2,  # transparency of the band
                label=None  # don't add to legend
            )

    # Adjust title and labels with improved font sizes
    ax.set_title(f"Test R² vs Num Samples\nLabel = {label}\nInit Set: {init_set_str}", fontsize=18, fontweight='bold')
    ax.set_xlabel("Number of Samples", fontsize=15)
    ax.set_ylabel("Test R²", fontsize=15)

    # Adjust legend outside the plot to avoid overlap
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))  # later duplicates overwrite earlier ones
    ax.legend(unique.values(), unique.keys(), title="Method", loc="upper left")

    # Apply y-limits if specified
    # plt.ylim(0.1, 0.5)

    ax.grid(True)

    if log:
        ax.set_xscale('log')

    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

def plot_r2_grid(base_path_template, type_str, nums):
    """
    Loads and displays 4 images in a 2x2 grid from the specified path template.
    
    Parameters:
    - base_path_template (str): File path template with `{num}` as placeholder for county count.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, num in enumerate(nums):
        image_path = base_path_template.format(type_str=type_str, num=num)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = mpimg.imread(image_path)
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout(pad=0.5)
    base_dir = os.path.dirname(image_path)
    parent_dir = os.path.dirname(base_dir)
    plt.savefig(os.path.join(parent_dir, f"{type_str}_combined_{nums}.png"), dpi=300)

def plot_prob_map(latlons, probabilities, cmap="viridis", title="Probability Heat Map", save_path=None):
    fig, ax = plt.subplots(figsize=(14, 10))

    # Load and plot contiguous US outline
    world = gpd.read_file(
        "/home/libe2152/optimizedsampling/country_boundaries/ne_110m_admin_1_states_provinces.shp", 
        engine="pyogrio"
    )
    exclude_states = ["Alaska", "Hawaii", "Puerto Rico"]
    contiguous_us = world[~world["name"].isin(exclude_states)]
    contiguous_outline = contiguous_us.dissolve()

    contiguous_outline.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=3, alpha=0.8)

    # Prepare and sort data
    latlons = np.array(latlons)
    probabilities = np.array(probabilities)

    epsilon = 1e-6
    probabilities = np.clip(probabilities, epsilon, None)

    # Sort to plot low probabilities first
    sort_idx = np.argsort(probabilities)
    latlons = latlons[sort_idx]
    probabilities = probabilities[sort_idx]
    lons, lats = latlons[:, 1], latlons[:, 0]

    # Create scatter plot with LogNorm
    norm = LogNorm(vmin=probabilities.min(), vmax=probabilities.max())
    scatter = ax.scatter(
        lons, lats,
        c=probabilities,
        cmap=cmap,
        norm=norm,
        s=12,
        edgecolor='k',
        linewidth=0.1,
        alpha=0.85,
        zorder=5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label="Inclusion Probability", shrink=0.7, pad=0.01)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_offset_position('left')
    cbar.outline.set_visible(False)

    # Title and axis
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    # Ticks and grid
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.4, zorder=2)

    # Set axis limits and aspect ratio
    ax.set_xlim([-130, -65])
    ax.set_ylim([23, 50])
    ax.set_aspect('equal', adjustable='box')

    # Tight layout and save
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # base_path_template = '/home/libe2152/deep-al/results/plots/USAVARS/population/{type_str}_{num}_counties_10_radius/R2 vs budget.png'
    # for type_str in ['density', 'clustered']:
    #     plot_r2_grid(base_path_template, type_str, [25, 50, 75, 100])
    #     plot_r2_grid(base_path_template, type_str, [125, 150, 175, 200])
    
    dataset_name = "USAVARS"
    labels = ['treecover', 'population']
    cost_aware=False

    for task in labels:
        for type_str in ['clustered', 'density']:
            df_list = []
            for num_counties in [25, 50, 75, 100, 125, 150, 175, 200]:
                for radius in [10]:
                    initial_set_str = f'{type_str}_{num_counties}_counties_{radius}_radius'

                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

                    csv_dir = os.path.join(project_root, f'results/csv/{dataset_name}/{task}/{initial_set_str}/cost_aware') if cost_aware else os.path.join(project_root, f'results/csv/{dataset_name}/{task}/{initial_set_str}')
                    plot_dir = os.path.join(project_root, f'results/plots/{dataset_name}/{task}/{initial_set_str}/cost_aware') if cost_aware else os.path.join(project_root, f'results/plots/{dataset_name}/{task}/{initial_set_str}')
                    os.makedirs(plot_dir, exist_ok=True)

                    csv_filepath = os.path.join(csv_dir, 'results.csv')
                    plot_filepath = os.path.join(plot_dir, 'R2 vs budget.png')

                    if not os.path.exists(csv_filepath):
                        print(f"{csv_filepath} does not exist.")
                        continue

                    df = pd.read_csv(csv_filepath)
                    df_list.append(df)

                    # plot_r2_vs_budget(
                    #     df,
                    #     task,
                    #     initial_set_str,
                    #     plot_filepath,
                    #     budget_bound=100,
                    #     methods_to_include=None,
                    #     log=False,
                    #     cost_aware=cost_aware
                    # )
            
            if len(df_list) == 0:
                continue
            plot_filepath = os.path.join(project_root, f"results/plots/{dataset_name}/{task}", f"{type_str}.png")

            plot_r2_vs_num_samples(
                df_list,
                task,
                type_str,
                plot_filepath,
                budget_bound=200,
                methods_to_include=["random", "representative_state"],
                log=False
            )
