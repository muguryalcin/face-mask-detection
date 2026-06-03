from pathlib import Path

import matplotlib.pyplot as plt


def write_training_plots(history, plots_dir):
    # Plots the curves and saves them to specified paths, returning the list of saved plot paths.
    # Ensure the plots directory exists
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        # Training curves (loss, loss_step, loss_epoch) and validation curves (loss, loss_epoch)
        _plot_series(
            history,
            ["train/loss", "train/loss_step", "train/loss_epoch"],
            plots_dir / "train_loss.png",
            "Training Loss",
            "loss",
        ),
        _plot_series(
            history,
            ["val/loss", "val/loss_epoch"],
            plots_dir / "val_loss.png",
            "Validation Loss",
            "loss",
        ),
        # Loss components for both training and validation
        _plot_components(history, plots_dir / "loss_components.png"),
    ]
    return paths


def _plot_series(history, keys, path, title, ylabel):
    # Generic plot function
    plt.figure(figsize=(8, 5))
    plotted = False

    for key in keys:
        values = history.get(key, [])
        if values:
            plt.plot(values, label=key)
            plotted = True

    if not plotted:
        plt.plot([0], [0], label="no data")

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_components(history, path):
    # Plots the individual loss components for both training and validation in a single plot for easier comparison.
    component_keys = [
        "train/loss_classifier",
        "train/loss_box_reg",
        "train/loss_objectness",
        "train/loss_rpn_box_reg",
        "val/loss_classifier",
        "val/loss_box_reg",
        "val/loss_objectness",
        "val/loss_rpn_box_reg",
    ]
    return _plot_series(history, component_keys, path, "Loss Components", "loss")
