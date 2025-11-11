# -*- coding: utf-8 -*-
# @Date    : 2025-11-12
# @Desc    : Generate comprehensive reports for AFlow results

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


class AFlowReporter:
    def __init__(self, workspace_path: str, dataset: str):
        self.workspace_path = workspace_path
        self.dataset = dataset
        self.workflows_path = f"{workspace_path}/{dataset}/workflows"
        self.workflows_test_path = f"{workspace_path}/{dataset}/workflows_test"

    def load_results(self, mode="validation"):
        """Load results from validation or test runs"""
        if mode == "validation":
            results_file = f"{self.workflows_path}/results.json"
        else:
            results_file = f"{self.workflows_test_path}/results.json"

        if not os.path.exists(results_file):
            return []

        with open(results_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_experience(self):
        """Load processed experience"""
        exp_file = f"{self.workflows_path}/processed_experience.json"
        if not os.path.exists(exp_file):
            return {}

        with open(exp_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_summary_table(self):
        """Generate summary table for all rounds"""
        validation_results = self.load_results("validation")

        if not validation_results:
            print("No validation results found!")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(validation_results)
        df["score_pct"] = df["score"] * 100
        df["time"] = pd.to_datetime(df["time"])

        # Summary statistics
        summary = df[["round", "score_pct", "avg_cost", "total_cost"]].copy()
        summary.columns = ["Round", "Score (%)", "Avg Cost ($)", "Total Cost ($)"]
        summary = summary.round(2)

        return summary

    def plot_optimization_curve(self, save_path=None):
        """Plot optimization curve over rounds"""
        validation_results = self.load_results("validation")

        if not validation_results:
            return

        df = pd.DataFrame(validation_results)
        df = df.sort_values("round")

        plt.figure(figsize=(12, 6))

        # Plot 1: Score progression
        plt.subplot(1, 2, 1)
        plt.plot(
            df["round"],
            df["score"] * 100,
            "o-",
            linewidth=2,
            markersize=8,
            color="#2E86AB",
        )
        plt.axhline(
            y=df["score"].max() * 100,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Best Score",
        )
        plt.xlabel("Round", fontsize=12, fontweight="bold")
        plt.ylabel("Validation Score (%)", fontsize=12, fontweight="bold")
        plt.title(
            f"{self.dataset} - Optimization Progress", fontsize=14, fontweight="bold"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Highlight best round
        best_idx = df["score"].idxmax()
        best_round = df.loc[best_idx]
        plt.plot(
            best_round["round"],
            best_round["score"] * 100,
            "r*",
            markersize=20,
            label="Best",
        )

        # Plot 2: Cost vs Score
        plt.subplot(1, 2, 2)
        plt.scatter(
            df["total_cost"],
            df["score"] * 100,
            s=200,
            alpha=0.6,
            c=df["round"],
            cmap="viridis",
        )
        plt.xlabel("Total Cost ($)", fontsize=12, fontweight="bold")
        plt.ylabel("Validation Score (%)", fontsize=12, fontweight="bold")
        plt.title("Cost vs Performance Trade-off", fontsize=14, fontweight="bold")
        plt.colorbar(label="Round")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def generate_workflow_evolution_report(self):
        """Generate report showing workflow evolution"""
        experience = self.load_experience()

        print(f"\n{'='*80}")
        print(f"WORKFLOW EVOLUTION ANALYSIS - {self.dataset}")
        print(f"{'='*80}\n")

        evolution_data = []

        for round_str, data in sorted(experience.items(), key=lambda x: int(x[0])):
            round_num = int(round_str)
            score = data.get("score", 0)

            print(f"Round {round_num}: Score = {score*100:.2f}%")

            # Successes
            if data.get("success"):
                for target_round, mod_data in data["success"].items():
                    evolution_data.append(
                        {
                            "Base Round": round_num,
                            "Target Round": int(target_round),
                            "Status": "Success",
                            "Score Improvement": f"{mod_data['score']*100:.2f}%",
                            "Modification": mod_data["modification"][:80] + "...",
                        }
                    )
                    print(f"  Round {target_round}: {mod_data['score']*100:.2f}%")
                    print(f"  Mod: {mod_data['modification'][:100]}...")

            # Failures
            if data.get("failure"):
                for target_round, mod_data in data["failure"].items():
                    evolution_data.append(
                        {
                            "Base Round": round_num,
                            "Target Round": int(target_round),
                            "Status": "Failure",
                            "Score Improvement": f"{mod_data['score']*100:.2f}%",
                            "Modification": mod_data["modification"][:80] + "...",
                        }
                    )
                    print(
                        f"Round {target_round}: {mod_data['score']*100:.2f}% (Failed)"
                    )
                    print(f"Mod: {mod_data['modification'][:100]}...")

            print()

        return pd.DataFrame(evolution_data)

    def generate_final_comparison(self):
        """Generate final comparison: Validation vs Test"""
        val_results = self.load_results("validation")
        test_results = self.load_results("test")

        print(f"\n{'='*80}")
        print(f"FINAL RESULTS COMPARISON - {self.dataset}")
        print(f"{'='*80}\n")

        if val_results:
            best_val = max(val_results, key=lambda x: x["score"])
            print(f"VALIDATION SET (Development):")
            print(f"Best Round: {best_val['round']}")
            print(f"Score: {best_val['score']*100:.2f}%")
            print(f"Avg Cost: ${best_val['avg_cost']:.6f}")
            print(f"Total Cost: ${best_val['total_cost']:.4f}")
        print()

        if test_results:
            test_result = test_results[-1]  # Latest test result
            print(f"TEST SET (Final Evaluation):")
            print(f"Round: {test_result['round']}")
            print(f"Score: {test_result['score']*100:.2f}%")
            print(f"Avg Cost: ${test_result['avg_cost']:.6f}")
            print(f"Total Cost: ${test_result['total_cost']:.4f}")

            if val_results:
                print(f"\nPerformance Gap:")
                gap = (test_result["score"] - best_val["score"]) * 100
                print(f"Test vs Validation: {gap:+.2f}%")
                if gap < 0:
                    print(f"Test score lower (possible overfitting on validation)")
                elif gap > 0:
                    print(f"Test score higher (good generalization)")
                else:
                    print(f"Consistent performance")
        else:
            print(f"No test results found. Run test mode first!")

        print(f"\n{'='*80}\n")

    def export_to_latex_table(self, save_path=None):
        """Export results as LaTeX table for papers/slides"""
        summary = self.generate_summary_table()

        if summary is None:
            return

        latex_table = summary.to_latex(
            index=False,
            column_format="c|c|c|c",
            caption=f"{self.dataset} Optimization Results",
            label=f"tab:{self.dataset.lower()}_results",
        )

        if save_path:
            with open(save_path, "w") as f:
                f.write(latex_table)
            print(f"LaTeX table saved to: {save_path}")
        else:
            print("\nLATEX TABLE:")
            print(latex_table)

    def generate_full_report(self, output_dir=None):
        """Generate comprehensive report with all visualizations"""
        if output_dir is None:
            output_dir = f"{self.workspace_path}/{self.dataset}/reports"

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*80}")
        print(f"GENERATING COMPREHENSIVE REPORT - {self.dataset}")
        print(f"{'='*80}\n")

        # 1. Summary table
        print("Generating summary table...")
        summary = self.generate_summary_table()
        if summary is not None:
            csv_path = f"{output_dir}/summary_{timestamp}.csv"
            summary.to_csv(csv_path, index=False)
            print(f"Saved to: {csv_path}")
            print(f"\n{summary.to_string(index=False)}\n")

        # 2. Optimization curve
        print("Generating optimization curve...")
        plot_path = f"{output_dir}/optimization_curve_{timestamp}.png"
        self.plot_optimization_curve(save_path=plot_path)

        # 3. Workflow evolution
        print("Analyzing workflow evolution...")
        evolution_df = self.generate_workflow_evolution_report()
        if not evolution_df.empty:
            evolution_path = f"{output_dir}/evolution_{timestamp}.csv"
            evolution_df.to_csv(evolution_path, index=False)
            print(f"Saved to: {evolution_path}")

        # 4. Final comparison
        print("Generating final comparison...")
        self.generate_final_comparison()

        # 5. LaTeX table
        print("Exporting LaTeX table...")
        latex_path = f"{output_dir}/results_table_{timestamp}.tex"
        self.export_to_latex_table(save_path=latex_path)

        print(f"\n{'='*80}")
        print(f"FULL REPORT GENERATED!")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}\n")


def main():
    """Main function to generate reports"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate AFlow results report")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MATH",
        help="Dataset name (MATH, HumanEval, etc.)",
    )
    parser.add_argument(
        "--workspace", type=str, default="workspace", help="Workspace path"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory for reports"
    )

    args = parser.parse_args()

    reporter = AFlowReporter(workspace_path=args.workspace, dataset=args.dataset)
    reporter.generate_full_report(output_dir=args.output)


if __name__ == "__main__":
    main()
