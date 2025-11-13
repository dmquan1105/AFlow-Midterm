# -*- coding: utf-8 -*-
# @Date    : 2025-11-12
# @Desc    : Compare AFlow results with baselines and paper results

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paper results (baseline from AFlow paper)
PAPER_RESULTS = {
    "MATH": {
        "baseline": {"CoT": 48.8, "CoT SC": 50.4},
        "aflow": {"validation": 57.0, "test": 56.2},
        "model": "GPT-4o mini",
    },
    "HumanEval": {
        "baseline": {"Direct": 67.0, "CoT": 68.9},
        "aflow": {"validation": 89.0, "test": 87.8},
        "model": "GPT-4o mini",
    },
    "GSM8K": {
        "baseline": {"CoT": 87.1, "Self-Consistency": 92.0},
        "aflow": {"validation": 94.2, "test": 93.8},
        "model": "GPT-4o mini",
    },
}


def load_our_results(workspace_path, dataset):
    """Load our results"""
    val_path = f"{workspace_path}/{dataset}/workflows/results.json"
    test_path = f"{workspace_path}/{dataset}/workflows_test/results.json"

    results = {}

    try:
        with open(val_path, "r") as f:
            val_data = json.load(f)
            results["validation"] = (
                max(val_data, key=lambda x: x["score"])["score"] * 100
            )
    except:
        results["validation"] = None

    try:
        with open(test_path, "r") as f:
            test_data = json.load(f)
            results["test"] = test_data[-1]["score"] * 100
    except:
        results["test"] = None

    return results


def create_comparison_chart(
    datasets=["MATH", "HumanEval", "GSM8K"], workspace_path="workspace", save_path=None
):
    """Create comparison chart with paper results"""

    fig, axes = plt.subplots(1, len(datasets), figsize=(15, 6))
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        if dataset not in PAPER_RESULTS:
            continue

        paper = PAPER_RESULTS[dataset]
        our_results = load_our_results(workspace_path, dataset)

        # Data
        methods = []
        scores = []
        colors = []

        # Baselines
        for method, score in paper["baseline"].items():
            methods.append(f"{method}\n(Paper)")
            scores.append(score)
            colors.append("#95a5a6")

        # Paper AFlow
        if paper["aflow"]["test"]:
            methods.append(f"AFlow\n(Paper-Test)")
            scores.append(paper["aflow"]["test"])
            colors.append("#3498db")

        # Our AFlow
        # if our_results["validation"]:
        #     methods.append(f"AFlow\n(Ours-Val)")
        #     scores.append(our_results["validation"])
        #     colors.append("#2ecc71")

        if our_results["test"]:
            methods.append(f"AFlow\n(Ours-Test)")
            scores.append(our_results["test"])
            colors.append("#27ae60")

        # Plot
        bars = ax.bar(
            range(len(methods)),
            scores,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        # Styling
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=9, fontweight="bold")
        ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            f'{dataset}\n({paper["model"]} vs DeepSeek)', fontsize=13, fontweight="bold"
        )
        ax.set_ylim([0, 100])
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison chart saved to: {save_path}")
    else:
        plt.show()


def print_comparison_table(
    datasets=["MATH", "HumanEval", "GSM8K"], workspace_path="workspace"
):
    """Print comparison table"""

    print(f"\n{'='*100}")
    print(f"COMPARISON: PAPER vs OUR IMPLEMENTATION")
    print(f"{'='*100}\n")

    for dataset in datasets:
        if dataset not in PAPER_RESULTS:
            continue

        paper = PAPER_RESULTS[dataset]
        our_results = load_our_results(workspace_path, dataset)

        print(f"{dataset} Dataset:")
        print(f"   Model: {paper['model']} (Paper) vs DeepSeek-V3.2 (Ours)\n")

        print(f"   Baselines (Paper):")
        for method, score in paper["baseline"].items():
            print(f"      • {method}: {score:.1f}%")

        print(f"\n   AFlow (Paper):")
        print(f"      • Validation: {paper['aflow']['validation']:.1f}%")
        print(f"      • Test: {paper['aflow']['test']:.1f}%")

        print(f"\n   AFlow (Ours - DeepSeek):")
        if our_results["validation"]:
            print(f"      • Validation: {our_results['validation']:.2f}%")
            val_diff = our_results["validation"] - paper["aflow"]["validation"]
            print(f"        Δ vs Paper: {val_diff:+.2f}%")

        if our_results["test"]:
            print(f"      • Test: {our_results['test']:.2f}%")
            test_diff = our_results["test"] - paper["aflow"]["test"]
            print(f"        Δ vs Paper: {test_diff:+.2f}%")
        else:
            print(f"      • Test: Not yet evaluated")

        print(f"\n   {'-'*80}")

        if our_results["validation"] and our_results["test"]:
            gap = our_results["validation"] - our_results["test"]
            print(f"   Generalization Gap: {gap:.2f}%")
            if abs(gap) < 2:
                print(f"      Excellent generalization!")
            elif abs(gap) < 5:
                print(f"      Good generalization")
            else:
                print(f"      Consider larger validation set")

        print(f"\n{'='*100}\n")


def create_presentation_summary(
    dataset="MATH", workspace_path="workspace", save_path=None
):
    """Create a single summary image for presentation"""

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Load data
    val_path = f"{workspace_path}/{dataset}/workflows/results.json"
    with open(val_path, "r") as f:
        val_data = json.load(f)

    df = pd.DataFrame(val_data).sort_values("round")

    # 1. Optimization Progress (Top Left - Large)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(
        df["round"],
        df["score"] * 100,
        "o-",
        linewidth=3,
        markersize=10,
        color="#2E86AB",
        label="Validation Score",
    )
    best_idx = df["score"].idxmax()
    best = df.iloc[best_idx]
    ax1.plot(
        best["round"],
        best["score"] * 100,
        "r*",
        markersize=25,
        label=f'Best: Round {best["round"]}',
    )
    ax1.axhline(y=best["score"] * 100, color="red", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Optimization Round", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Validation Score (%)", fontsize=14, fontweight="bold")
    ax1.set_title(
        f"{dataset} - AFlow Optimization Progress", fontsize=16, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)

    # 2. Success/Failure Rounds (Top Right)
    ax2 = fig.add_subplot(gs[0, 2])
    success_rounds = df[df["score"] > 0.5]["round"].tolist()
    fail_rounds = df[df["score"] <= 0.5]["round"].tolist()

    categories = ["Success", "Failure"]
    counts = [len(success_rounds), len(fail_rounds)]
    colors_pie = ["#27ae60", "#e74c3c"]

    ax2.pie(
        counts,
        labels=categories,
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=90,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )
    ax2.set_title("Round Outcomes", fontsize=14, fontweight="bold")

    # 3. Cost Analysis (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(
        df["round"], df["total_cost"], color="#f39c12", alpha=0.7, edgecolor="black"
    )
    ax3.set_xlabel("Round", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Total Cost ($)", fontsize=12, fontweight="bold")
    ax3.set_title("Cost per Round", fontsize=14, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # 4. Score Distribution (Bottom Middle)
    ax4 = fig.add_subplot(gs[1, 1])
    scores_nonzero = df[df["score"] > 0]["score"] * 100
    if len(scores_nonzero) > 0:
        ax4.hist(scores_nonzero, bins=10, color="#9b59b6", alpha=0.7, edgecolor="black")
        ax4.axvline(
            scores_nonzero.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {scores_nonzero.mean():.1f}%",
        )
        ax4.set_xlabel("Score (%)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax4.set_title("Score Distribution", fontsize=14, fontweight="bold")
        ax4.legend()

    # 5. Summary Stats (Bottom Right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    stats_text = f"""
    SUMMARY STATISTICS
    
    Total Rounds: {len(df)}
    Success Rounds: {len(success_rounds)}
    Failed Rounds: {len(fail_rounds)}
    
    Best Score: {df['score'].max()*100:.2f}%
    Best Round: {best['round']}
    
    Total Cost: ${df['total_cost'].sum():.4f}
    Avg Cost/Round: ${df['total_cost'].mean():.4f}
        """

    ax5.text(
        0.1,
        0.5,
        stats_text,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(
        f"AFlow Results Summary - {dataset} Dataset",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Presentation summary saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare with paper results")
    parser.add_argument("--dataset", type=str, default="MATH")
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--output", type=str, default="reports")

    args = parser.parse_args()

    import os

    os.makedirs(args.output, exist_ok=True)

    # Generate all visualizations
    print_comparison_table([args.dataset], args.workspace)

    create_comparison_chart(
        [args.dataset],
        args.workspace,
        save_path=f"{args.output}/comparison_{args.dataset}.png",
    )

    create_presentation_summary(
        args.dataset,
        args.workspace,
        save_path=f"{args.output}/summary_{args.dataset}.png",
    )
