import io
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class ReportVisualizationService:
    """
    Generates publication-quality figures using matplotlib from actual stored experiment & XAI data.
    All methods return raw PNG bytes.
    """

    @staticmethod
    def set_publication_style():
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300,
        })

    @classmethod
    def generate_fig1_model_performance(cls, aggregated_stats: Dict[str, Any]) -> bytes:
        """
        FIGURE 1: Grouped Model Performance Comparison for F1, Accuracy, and ROC-AUC (Mean ± Std).
        """
        cls.set_publication_style()
        variants = list(aggregated_stats.keys())
        labels = [
            v.replace("CENTRALIZED (BASELINE)", "Centralized")
             .replace("FEDAVG (IID)", "FedAvg (IID)")
             .replace("FEDAVG (NON_IID_ALPHA_0.5)", "FedAvg (Non-IID)")
             .replace("FEDPROX (NON_IID_ALPHA_0.5_MU_0.001)", "FedProx (μ=0.001)")
             .replace("FEDPROX (NON_IID_ALPHA_0.5_MU_0.01)", "FedProx (μ=0.01)")
             .replace("FEDPROX (NON_IID_ALPHA_0.5_MU_0.1)", "FedProx (μ=0.1)")
            for v in variants
        ]

        f1_means = [aggregated_stats[v]["metrics_summary"]["f1_mean"] for v in variants]
        f1_stds = [aggregated_stats[v]["metrics_summary"]["f1_std"] for v in variants]

        acc_means = [aggregated_stats[v]["metrics_summary"]["accuracy_mean"] for v in variants]
        acc_stds = [aggregated_stats[v]["metrics_summary"]["accuracy_std"] for v in variants]

        auc_means = [aggregated_stats[v]["metrics_summary"]["roc_auc_mean"] for v in variants]
        auc_stds = [aggregated_stats[v]["metrics_summary"]["roc_auc_std"] for v in variants]

        x = np.arange(len(labels))
        width = 0.26

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(x - width, f1_means, width, yerr=f1_stds, capsize=4, label="F1 Score", color="#2b5c8f", alpha=0.9)
        ax.bar(x, acc_means, width, yerr=acc_stds, capsize=4, label="Accuracy", color="#3b8b68", alpha=0.9)
        ax.bar(x + width, auc_means, width, yerr=auc_stds, capsize=4, label="ROC-AUC", color="#d95f02", alpha=0.9)

        ax.set_ylabel("Metric Value (Mean ± Sample Std)")
        ax.set_title("Figure 1: Model Performance Comparison Across 5 Random Seeds")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper left")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close(fig)
        return buf.getvalue()

    @classmethod
    def generate_fig2_f1_across_seeds(cls, aggregated_stats: Dict[str, Any], seeds: List[int]) -> bytes:
        """
        FIGURE 2: F1 Metric Trajectory Across Seeds (Demonstrating Stability & Collapses).
        """
        cls.set_publication_style()
        fig, ax = plt.subplots(figsize=(9, 5))

        markers = ["o", "s", "^", "D", "v", "P"]
        colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

        for i, (key, data) in enumerate(aggregated_stats.items()):
            label = (
                key.replace("CENTRALIZED (BASELINE)", "Centralized")
                   .replace("FEDAVG (IID)", "FedAvg (IID)")
                   .replace("FEDAVG (NON_IID_ALPHA_0.5)", "FedAvg (Non-IID)")
                   .replace("FEDPROX (NON_IID_ALPHA_0.5_MU_0.001)", "FedProx (μ=0.001)")
                   .replace("FEDPROX (NON_IID_ALPHA_0.5_MU_0.01)", "FedProx (μ=0.01)")
                   .replace("FEDPROX (NON_IID_ALPHA_0.5_MU_0.1)", "FedProx (μ=0.1)")
            )
            f1_vals = [r["metrics"]["f1"] for r in data["seed_runs"]]
            ax.plot(
                range(len(seeds)),
                f1_vals,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2,
                markersize=7,
                label=label,
            )

        ax.set_xlabel("Random Seed")
        ax.set_ylabel("Test Split F1 Score")
        ax.set_title("Figure 2: Test F1 Stability Across Independent Random Seeds")
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f"Seed {s}" for s in seeds])
        ax.set_ylim(-0.05, 0.8)
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close(fig)
        return buf.getvalue()

    @classmethod
    def generate_fig3_roc_auc_across_seeds(cls, aggregated_stats: Dict[str, Any], seeds: List[int]) -> bytes:
        """
        FIGURE 3: ROC-AUC Across Seeds.
        """
        cls.set_publication_style()
        fig, ax = plt.subplots(figsize=(9, 5))

        markers = ["o", "s", "^", "D", "v", "P"]
        colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

        for i, (key, data) in enumerate(aggregated_stats.items()):
            label = (
                key.replace("CENTRALIZED (BASELINE)", "Centralized")
                   .replace("FEDAVG (IID)", "FedAvg (IID)")
                   .replace("FEDAVG (NON_IID_ALPHA_0.5)", "FedAvg (Non-IID)")
                   .replace("FEDPROX (NON_IID_ALPHA_0.5_MU_0.001)", "FedProx (μ=0.001)")
                   .replace("FEDPROX (NON_IID_ALPHA_0.5_MU_0.01)", "FedProx (μ=0.01)")
                   .replace("FEDPROX (NON_IID_ALPHA_0.5_MU_0.1)", "FedProx (μ=0.1)")
            )
            auc_vals = [r["metrics"]["roc_auc"] for r in data["seed_runs"]]
            ax.plot(
                range(len(seeds)),
                auc_vals,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2,
                markersize=7,
                label=label,
            )

        ax.set_xlabel("Random Seed")
        ax.set_ylabel("Test Split ROC-AUC")
        ax.set_title("Figure 3: Test ROC-AUC Discriminative Ranking Across 5 Seeds")
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f"Seed {s}" for s in seeds])
        ax.set_ylim(0.4, 0.8)
        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close(fig)
        return buf.getvalue()

    @classmethod
    def generate_fig4_fedprox_mu_sensitivity(cls, aggregated_stats: Dict[str, Any]) -> bytes:
        """
        FIGURE 4: FedProx μ Sensitivity (X-axis μ, Y-axis F1 and ROC-AUC Mean ± Std).
        """
        cls.set_publication_style()
        mu_keys = [
            ("0.001", "FEDPROX (NON_IID_ALPHA_0.5_MU_0.001)"),
            ("0.01", "FEDPROX (NON_IID_ALPHA_0.5_MU_0.01)"),
            ("0.1", "FEDPROX (NON_IID_ALPHA_0.5_MU_0.1)"),
        ]

        mu_labels = [k[0] for k in mu_keys]
        f1_means = [aggregated_stats[k[1]]["metrics_summary"]["f1_mean"] for k in mu_keys]
        f1_stds = [aggregated_stats[k[1]]["metrics_summary"]["f1_std"] for k in mu_keys]

        auc_means = [aggregated_stats[k[1]]["metrics_summary"]["roc_auc_mean"] for k in mu_keys]
        auc_stds = [aggregated_stats[k[1]]["metrics_summary"]["roc_auc_std"] for k in mu_keys]

        # Compare with FedAvg Non-IID baseline
        fedavg_f1_mean = aggregated_stats["FEDAVG (NON_IID_ALPHA_0.5)"]["metrics_summary"]["f1_mean"]
        fedavg_auc_mean = aggregated_stats["FEDAVG (NON_IID_ALPHA_0.5)"]["metrics_summary"]["roc_auc_mean"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

        # Subplot 1: F1 vs mu
        ax1.errorbar(
            range(len(mu_labels)),
            f1_means,
            yerr=f1_stds,
            fmt="o-",
            capsize=5,
            color="#2b5c8f",
            linewidth=2,
            markersize=8,
            label="FedProx (Mean ± Std)",
        )
        ax1.axhline(fedavg_f1_mean, color="#d62728", linestyle="--", label="FedAvg Non-IID Baseline")
        ax1.set_xticks(range(len(mu_labels)))
        ax1.set_xticklabels(mu_labels)
        ax1.set_xlabel("Proximal Regularization Parameter (μ)")
        ax1.set_ylabel("Test F1 Score")
        ax1.set_title("(a) F1 Sensitivity to μ")
        ax1.set_ylim(0.0, 0.7)
        ax1.legend(loc="lower left")

        # Subplot 2: ROC-AUC vs mu
        ax2.errorbar(
            range(len(mu_labels)),
            auc_means,
            yerr=auc_stds,
            fmt="s-",
            capsize=5,
            color="#3b8b68",
            linewidth=2,
            markersize=8,
            label="FedProx (Mean ± Std)",
        )
        ax2.axhline(fedavg_auc_mean, color="#d62728", linestyle="--", label="FedAvg Non-IID Baseline")
        ax2.set_xticks(range(len(mu_labels)))
        ax2.set_xticklabels(mu_labels)
        ax2.set_xlabel("Proximal Regularization Parameter (μ)")
        ax2.set_ylabel("Test ROC-AUC")
        ax2.set_title("(b) ROC-AUC Sensitivity to μ")
        ax2.set_ylim(0.45, 0.75)
        ax2.legend(loc="lower left")

        plt.suptitle("Figure 4: FedProx Proximal Regularization (μ) Sensitivity Analysis (N=5 Seeds)")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close(fig)
        return buf.getvalue()

    @classmethod
    def generate_fig5_iid_vs_non_iid(cls, aggregated_stats: Dict[str, Any]) -> bytes:
        """
        FIGURE 5: IID vs Non-IID Performance Degradation Analysis (F1 & Recall).
        """
        cls.set_publication_style()
        fig, ax = plt.subplots(figsize=(8, 4.8))

        compare_keys = [
            ("FedAvg (IID)", "FEDAVG (IID)"),
            ("FedAvg (Non-IID)", "FEDAVG (NON_IID_ALPHA_0.5)"),
            ("FedProx (μ=0.01)", "FEDPROX (NON_IID_ALPHA_0.5_MU_0.01)"),
        ]

        labels = [c[0] for c in compare_keys]
        f1_vals = [aggregated_stats[c[1]]["metrics_summary"]["f1_mean"] for c in compare_keys]
        f1_err = [aggregated_stats[c[1]]["metrics_summary"]["f1_std"] for c in compare_keys]
        rec_vals = [aggregated_stats[c[1]]["metrics_summary"]["recall_mean"] for c in compare_keys]
        rec_err = [aggregated_stats[c[1]]["metrics_summary"]["recall_std"] for c in compare_keys]

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width/2, f1_vals, width, yerr=f1_err, capsize=4, label="F1 Score", color="#1f77b4", alpha=0.85)
        ax.bar(x + width/2, rec_vals, width, yerr=rec_err, capsize=4, label="Recall (Dropout)", color="#ff7f0e", alpha=0.85)

        ax.set_ylabel("Metric Value (Mean ± Std across 5 seeds)")
        ax.set_title("Figure 5: Impact of Client Non-IID Label Skew on F1 and Dropout Recall")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 0.8)
        ax.legend(loc="upper right")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close(fig)
        return buf.getvalue()

    @classmethod
    def generate_fig6_convergence(cls, trajectories: Dict[str, Any]) -> bytes:
        """
        FIGURE 6: Validation Convergence Trajectories.
        """
        cls.set_publication_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

        color_map = {
            "CENTRALIZED": "#1f77b4",
            "FEDAVG_IID": "#2ca02c",
            "FEDAVG_NON_IID_ALPHA_0.5": "#d62728",
            "FEDPROX_MU_0.001": "#9467bd",
            "FEDPROX_MU_0.01": "#8c564b",
            "FEDPROX_MU_0.1": "#e377c2",
        }

        for model_name, traj in trajectories.items():
            c = color_map.get(model_name, "#333333")
            label = (
                model_name.replace("FEDAVG_IID", "FedAvg (IID)")
                          .replace("FEDAVG_NON_IID_ALPHA_0.5", "FedAvg (Non-IID)")
                          .replace("FEDPROX_MU_", "FedProx μ=")
                          .replace("CENTRALIZED", "Centralized")
            )

            x_steps = traj.get("rounds") or traj.get("epochs", [])
            val_loss = traj.get("loss", [])
            val_acc = traj.get("validation_accuracy", [])

            if x_steps and val_loss:
                ax1.plot(x_steps, val_loss, marker="o", label=label, color=c, linewidth=1.8, markersize=5)
            if x_steps and val_acc:
                ax2.plot(x_steps, val_acc, marker="s", label=label, color=c, linewidth=1.8, markersize=5)

        ax1.set_xlabel("Rounds / Epochs")
        ax1.set_ylabel("Validation Loss")
        ax1.set_title("(a) Validation Loss Trajectory")
        ax1.legend(loc="upper right", fontsize=8)

        ax2.set_xlabel("Rounds / Epochs")
        ax2.set_ylabel("Validation Accuracy")
        ax2.set_title("(b) Validation Accuracy Trajectory")
        ax2.set_ylim(0.2, 0.85)
        ax2.legend(loc="lower right", fontsize=8)

        plt.suptitle("Figure 6: Model Convergence Dynamics on Real OULAD Splits")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close(fig)
        return buf.getvalue()

    @classmethod
    def generate_fig7_client_heterogeneity(cls, client_dists: Dict[str, Any]) -> bytes:
        """
        FIGURE 7: Client Heterogeneity Breakdown (IID vs Dirichlet Non-IID).
        """
        cls.set_publication_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        iid_clients = client_dists.get("iid", {}).get("client_breakdown", {})
        non_iid_clients = client_dists.get("non_iid", {}).get("client_breakdown", {})

        def plot_breakdown(ax, clients, title):
            c_names = list(clients.keys())
            c0_counts = [clients[c]["class_0_count"] for c in c_names]
            c1_counts = [clients[c]["class_1_count"] for c in c_names]

            x = np.arange(len(c_names))
            width = 0.5

            ax.bar(x, c0_counts, width, label="Non-Dropout (0)", color="#2b5c8f", alpha=0.85)
            ax.bar(x, c1_counts, width, bottom=c0_counts, label="Dropout (1)", color="#d95f02", alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels([c.replace("_", " ").title() for c in c_names])
            ax.set_ylabel("Sample Count")
            ax.set_title(title)
            ax.set_ylim(0, 65)
            ax.legend(loc="upper right")

        plot_breakdown(ax1, iid_clients, "(a) IID Partitioning (Uniform)")
        plot_breakdown(ax2, non_iid_clients, "(b) Non-IID Partitioning (Dirichlet α=0.5)")

        plt.suptitle("Figure 7: Client Sample & Label Distribution across 3 Federated Partitions")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close(fig)
        return buf.getvalue()

    @classmethod
    def generate_fig8_xai_global_importance(cls, global_shap_features: List[Dict[str, Any]]) -> bytes:
        """
        FIGURE 8: Top 10 Global SHAP Feature Importance.
        """
        cls.set_publication_style()
        fig, ax = plt.subplots(figsize=(9, 5.2))

        top_features = global_shap_features[:10]
        # Reverse order for horizontal bar chart (highest at top)
        top_features = list(reversed(top_features))

        names = [f["feature_name"] for f in top_features]
        vals = [f["mean_abs_attribution"] for f in top_features]

        y = np.arange(len(names))
        bars = ax.barh(y, vals, color="#2b5c8f", alpha=0.9, height=0.6)

        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("Mean |SHAP Attribution|")
        ax.set_title("Figure 8: Global Feature Importance via Kernel SHAP")

        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va="center", ha="left", fontsize=9)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        plt.close(fig)
        return buf.getvalue()
