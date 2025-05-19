import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

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

    if match1:
        count, region, number, unit, seed = match1.groups()
        return f"{count} {region.capitalize()}, {number} {unit.capitalize()} [Seed {seed}]"
    elif match2:
        count, region, number, unit, seed = match2.groups()
        return f"{count} {region.capitalize()}, {number}{unit} radius [Seed {seed}]"

    return desc

def nice_label_name(label):
    """Map label codes to human-readable names."""
    mapping = {
        "TC": "Tree Cover",
        "POP": "Population",
    }
    return mapping.get(label, label)

def plot_initial_final_r2_comparison(
    csv_path,
    label="TC",
    budget=100,
    init_set_desc="IDs_500_centers_5000m_radius_seed_42",
    save_path=None,
    ylim=(0.7, 0.88)
):
    df = pd.read_csv(csv_path)

    # Filter for specific conditions
    subset = df[
        (df["Label"] == label) &
        (df["Budget"] == budget) &
        (df["Initial Set Description"] == init_set_desc)
    ]

    if subset.empty:
        raise ValueError("Filtered DataFrame is empty. Check your inputs.")

    methods = subset["Method"].unique()
    plot_df = pd.DataFrame()

    for method in methods:
        method_df = subset[subset["Method"] == method]
        initial_r2 = method_df["Initial Test R2"].iloc[0]
        final_r2_values = method_df.groupby("Seed")["Test R2"].max()

        plot_df = pd.concat([plot_df, pd.DataFrame({
            "Method": [method, method],
            "Stage": ["Initial", "Final"],
            "Test R2": [initial_r2, final_r2_values.mean()],
            "Std": [0, final_r2_values.std()]
        })], ignore_index=True)

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=plot_df, x="Method", y="Test R2", hue="Stage", palette="Set2")

    # Optional error bars
    for i, bar in enumerate(ax.patches):
        std = plot_df["Std"].iloc[i]
        if std > 0:
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                yerr=std,
                fmt='none',
                c='black',
                capsize=4
            )

    plt.title(f"Initial vs Final Test R²\nLabel = {label}, Budget = {budget}", fontsize=16)
    plt.ylim(*ylim)
    plt.xlabel("Method", fontsize=14)
    plt.ylabel("Test R²", fontsize=14)
    plt.legend(title="Stage")
    plt.tight_layout()

    if save_path is None:
        safe_desc = init_set_desc.replace("/", "_").replace(" ", "_")
        save_path = Path(f"initial_final_r2_{label}_{safe_desc}.png")

    plt.savefig(save_path)
    plt.close()


def plot_r2_vs_budget(
    csv_path,
    label="TC",
    init_set_desc="IDs_500_centers_5000m_radius_seed_42",
    methods_to_include=None,
    save_path=None,
    ylim=(0.7, 0.88)
):
    df = pd.read_csv(csv_path)

    # Filter by label and initial set
    subset = df[
        (df["Label"] == label) &
        (df["Initial Set Description"] == init_set_desc)
    ]

    if methods_to_include:
        subset = subset[subset["Method"].isin(methods_to_include)]

    if subset.empty:
        raise ValueError("Filtered DataFrame is empty. Check your inputs.")

    # Handle ensemble_variance: only keep seed 1
    ensemble_df = subset[(subset["Method"] == "ensemble_variance")]
    other_df = subset[subset["Method"] != "ensemble_variance"]

    # Group and aggregate separately
    grouped_other = other_df.groupby(["Method", "Budget"])["Test R2"]
    r2_mean_other = grouped_other.mean().reset_index(name="Mean R2")
    r2_std_other = grouped_other.std().reset_index(name="Std R2")

    ensemble_df = ensemble_df.rename(columns={"Test R2": "Mean R2"})
    ensemble_df["Std R2"] = 0.0  # No std since only one seed

    # Keep only necessary columns
    ensemble_df = ensemble_df[["Method", "Budget", "Mean R2", "Std R2"]]

    # Combine
    plot_df = pd.concat([r2_mean_other.merge(r2_std_other, on=["Method", "Budget"]), ensemble_df], ignore_index=True)

    initial_test_r2_df = subset[["Method", "Initial Test R2"]].copy()
    initial_test_r2_df["Budget"] = 0
    initial_test_r2_df["Std R2"] = 0.0  # No std for Initial Test R²
    initial_test_r2_df = initial_test_r2_df.rename(columns={"Initial Test R2": "Mean R2"})
    
    # Add this to the plot dataframe
    plot_df = pd.concat([plot_df, initial_test_r2_df], ignore_index=True)

    method_order = ["random", "typiclust", "inversetypiclust", "ensemble_variance"]
    plot_df["Method"] = pd.Categorical(plot_df["Method"], categories=method_order, ordered=True)
    plot_df = plot_df.sort_values(by=["Method", "Budget"])

    # Plotting
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(
        data=plot_df,
        x="Budget",
        y="Mean R2",
        hue="Method",
        style="Method",
        markers=True,
        dashes=False,
        palette="Set1",
        err_style=None
    )

    #Plot std
    for method in plot_df["Method"].unique():
        method_df = plot_df[plot_df["Method"] == method]
        ax.errorbar(
            method_df["Budget"],
            method_df["Mean R2"],
            yerr=method_df["Std R2"],
            fmt="none",
            capsize=4,
            c="black"
        )

    # Update init set description
    nice_init_set_desc = nice_initial_set_desc(init_set_desc)
    if not subset.empty and "Initial Set Size" in subset.columns:
        init_set_size = subset["Initial Set Size"].iloc[0]
        nice_init_set_desc += f" (Total: {init_set_size})"

    # Adjust title and labels with improved font sizes
    ax.set_title(f"Test R² vs Budget\nLabel = {nice_label_name(label)}\nInit Set: {nice_init_set_desc}", fontsize=18, fontweight='bold')
    ax.set_xlabel("Budget", fontsize=15)
    ax.set_ylabel("Test R²", fontsize=15)

    # Adjust legend outside the plot to avoid overlap
    ax.legend(title="Methods", loc='upper left', fontsize=12)

    # Apply y-limits if specified
    #plt.ylim(*ylim)

    ax.grid(True)
    plt.tight_layout()

    # Handle save path
    if save_path is None:
        safe_desc = init_set_desc.replace("/", "_").replace(" ", "_")
        save_path = Path(f"r2_vs_budget_{label}_{safe_desc}.png")

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == '__main__':

    for init_set_desc in ["IDs_100_counties_5_points_seed_42", "IDs_500_counties_4_points_seed_42", "IDs_500_centers_5000m_radius_seed_42"]:
        plot_r2_vs_budget(
            "log_extracted_data_sorted.csv",
            label="TC",
            init_set_desc=init_set_desc,
            methods_to_include=None,
            save_path=None,
            ylim=(0.7, 0.88)
        )
