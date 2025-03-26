import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict


def plot_metric_contour(
        results: List[Dict[str, any]],
        metric: str,
        save_path: str | None = None
    ) -> None:
    # Get unique sorted hyperparameters
    unique_lrs = sorted(list({res["learning_rate"] for res in results}))
    unique_bs = sorted(list({res["batch_size"] for res in results}))
    
    # Create metric grid
    metric_grid = np.zeros((len(unique_lrs), len(unique_bs)))
    
    # Populate grid with final epoch's metric values
    for res in results:
        lr_idx = unique_lrs.index(res["learning_rate"])
        bs_idx = unique_bs.index(res["batch_size"])
        metric_grid[lr_idx, bs_idx] = res["result"][-1][metric]
    
    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        metric_grid, 
        xticklabels=unique_bs, 
        yticklabels=unique_lrs, 
        annot=True, 
        fmt=".2f",
        cmap="viridis"
    )
    plt.xlabel("Batch Size")
    plt.ylabel("Learning Rate")
    plt.title(f"{metric} vs Learning Rate and Batch Size (Final Epoch)")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_single_epoch_vs_metric(
        results: List[Dict[str, any]],
        metric: str,
        save_path: str | None = None
    ) -> None:
    plt.figure(figsize=(14, 8))
    
    # Create consistent color mapping for experiments
    unique_configs = sorted(
        {(res["learning_rate"], res["batch_size"]) for res in results},
        key=lambda x: (x[0], x[1])
    )
    color_palette = sns.color_palette("husl", n_colors=len(unique_configs))
    
    # Plot each experiment with full config in label
    for idx, res in enumerate(results):
        lr = res["learning_rate"]
        bs = res["batch_size"]
        config = (lr, bs)
        
        epochs = range(len(res["result"]))
        metric_values = [r[metric] for r in res["result"]]
        
        color_idx = unique_configs.index(config)
        plt.plot(epochs, metric_values,
                 color=color_palette[color_idx],
                 linewidth=2.5,
                 marker='o' if len(results) < 15 else '',
                 markersize=6,
                 label=f"LR={lr:.0e}, BS={bs}")

    # Create comprehensive legend
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f"{metric} Progression by Experiment Configuration", fontsize=14, pad=20)
    plt.grid(True, alpha=0.2)
    
    # Position legend outside with scrollable option if needed
    legend = plt.legend(
        title="Experiment Configurations",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=9,
        ncol=1 if len(results) < 15 else 2,
        frameon=False
    )
    
    # Improve legend readability
    plt.setp(legend.get_title(), fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make space for legend
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
