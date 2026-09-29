"""Generate validation metric comparisons from completed experiments."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENTS = (
    ("baseline", "Modelo de referência", "0.15", "-", "o"),
    ("batchnorm", "BN estatístico", "0.45", "--", "s"),
    ("weighted_batchnorm", "BN treinável", "0.25", "-.", "^"),
    ("stochastic_depth", "BN treinável + SD", "0.55", ":", "D"),
)
METRICS = (
    ("validation_losses", "Perda de validação", "perda_validacao", 1),
    ("validation_accuracy_top1", r"Acurácia top-1 (\%)", "acuracia_top1_validacao", 100),
    ("validation_accuracy_top5", r"Acurácia top-5 (\%)", "acuracia_top5_validacao", 100),
)


def draw(ax, data, metric):
    key, ylabel, _, scale = metric
    for name, label, color, linestyle, marker in EXPERIMENTS:
        ax.plot(
            range(1, 11), [value * scale for value in data[name][key]],
            label=label, color=color,
            linestyle=linestyle, marker=marker, linewidth=1.5, markersize=3,
        )
    ax.set_xlabel("Época")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.8, 10.2)
    ax.grid(axis="y", color="0.85", linewidth=0.5)
    ax.margins(y=0.12)


def save(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png", dpi=200)
    plt.close(fig)


def main():
    report_dir = Path(__file__).resolve().parent
    project_dir = report_dir.parent.parent
    output_dir = report_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, *_ in EXPERIMENTS:
        path = project_dir / "outputs" / name / "training_results.json"
        data[name] = json.loads(path.read_text())
        for key, *_ in METRICS:
            if len(data[name][key]) != 10:
                raise ValueError(f"Expected 10 epochs for {name}: {key}")

    with plt.rc_context({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }):
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(15 / 2.54, 6.5 / 2.54), layout="constrained")
            draw(ax, data, metric)
            ax.legend(loc="best", fontsize=10)
            save(fig, output_dir, metric[2])

        # Compact layout for the report; standalone plots retain full size.
        fig = plt.figure(figsize=(15 / 2.54, 9 / 2.54), layout="constrained")
        grid = fig.add_gridspec(2, 2)
        axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
        titles = ("(a)", "(b)", "(c)")
        for ax, metric, title in zip(axes, METRICS, titles):
            draw(ax, data, metric)
            ax.set_title(title, fontsize=10, loc="left")
            ax.tick_params(labelsize=9)
            ax.xaxis.label.set_size(10)
            ax.yaxis.label.set_size(10)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=10)
        save(fig, output_dir, "metricas_validacao")


if __name__ == "__main__":
    main()
