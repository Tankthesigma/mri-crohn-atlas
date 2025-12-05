#!/usr/bin/env python3
"""
Error Analysis and Visualization for MRI Report Parser
Generates confusion matrices, scatter plots, and error distribution charts

Part of MRI-Crohn Atlas ISEF 2026 Project
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


class ParserErrorAnalysis:
    """Generate error analysis visualizations"""

    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.data = None
        self.output_dir = None

    def load_results(self):
        """Load validation results"""
        with open(self.results_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data['results'])} validation results")

    def setup_output_dir(self, output_dir: str):
        """Setup output directory for plots"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_feature_accuracy_bar(self):
        """Bar chart of per-feature accuracy"""
        feature_acc = self.data['summary']['feature_accuracy']

        features = list(feature_acc.keys())
        accuracies = [feature_acc[f] * 100 for f in features]

        # Sort by accuracy
        sorted_pairs = sorted(zip(features, accuracies), key=lambda x: -x[1])
        features, accuracies = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Color bars based on accuracy
        colors = ['#2ecc71' if a >= 80 else '#f39c12' if a >= 60 else '#e74c3c' for a in accuracies]

        bars = ax.barh(features, accuracies, color=colors, edgecolor='white', linewidth=0.7)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{acc:.1f}%', va='center', fontsize=10, fontweight='bold')

        ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title('MRI Report Parser - Per-Feature Extraction Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 110)
        ax.axvline(x=80, color='#27ae60', linestyle='--', linewidth=2, label='Target (80%)')
        ax.legend(loc='lower right')

        # Add legend for colors
        legend_elements = [
            mpatches.Patch(color='#2ecc71', label='High (>=80%)'),
            mpatches.Patch(color='#f39c12', label='Medium (60-80%)'),
            mpatches.Patch(color='#e74c3c', label='Low (<60%)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: feature_accuracy.png")

    def plot_score_scatter(self):
        """Scatter plot of extracted vs expected scores"""
        results = self.data['results']

        vai_extracted = [r['vai_extracted'] for r in results if r['vai_extracted'] is not None]
        vai_expected = [r['vai_expected'] for r in results if r['vai_expected'] is not None]

        magnifi_extracted = [r['magnifi_extracted'] for r in results if r['magnifi_extracted'] is not None]
        magnifi_expected = [r['magnifi_expected'] for r in results if r['magnifi_expected'] is not None]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # VAI scatter
        ax1 = axes[0]
        ax1.scatter(vai_expected, vai_extracted, c='#3498db', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
        ax1.plot([0, 22], [0, 22], 'k--', linewidth=2, label='Perfect Agreement')
        ax1.plot([0, 22], [2, 24], 'r--', linewidth=1, alpha=0.5, label='+/- 2 points')
        ax1.plot([0, 22], [-2, 20], 'r--', linewidth=1, alpha=0.5)
        ax1.fill_between([0, 22], [-2, 20], [2, 24], alpha=0.1, color='green')
        ax1.set_xlabel('Expected VAI Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Extracted VAI Score', fontsize=12, fontweight='bold')
        ax1.set_title('VAI Score Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlim(-1, 23)
        ax1.set_ylim(-1, 23)
        ax1.legend(loc='upper left')
        ax1.set_aspect('equal')

        # Add MAE annotation
        vai_mae = self.data['summary']['vai_mae']
        ax1.text(0.95, 0.05, f'MAE = {vai_mae:.2f}', transform=ax1.transAxes,
                fontsize=12, fontweight='bold', ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # MAGNIFI scatter
        ax2 = axes[1]
        ax2.scatter(magnifi_expected, magnifi_extracted, c='#9b59b6', s=100, alpha=0.7, edgecolors='white', linewidth=1.5)
        ax2.plot([0, 25], [0, 25], 'k--', linewidth=2, label='Perfect Agreement')
        ax2.plot([0, 25], [2, 27], 'r--', linewidth=1, alpha=0.5, label='+/- 2 points')
        ax2.plot([0, 25], [-2, 23], 'r--', linewidth=1, alpha=0.5)
        ax2.fill_between([0, 25], [-2, 23], [2, 27], alpha=0.1, color='green')
        ax2.set_xlabel('Expected MAGNIFI-CD Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Extracted MAGNIFI-CD Score', fontsize=12, fontweight='bold')
        ax2.set_title('MAGNIFI-CD Score Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlim(-1, 26)
        ax2.set_ylim(-1, 26)
        ax2.legend(loc='upper left')
        ax2.set_aspect('equal')

        # Add MAE annotation
        magnifi_mae = self.data['summary']['magnifi_mae']
        ax2.text(0.95, 0.05, f'MAE = {magnifi_mae:.2f}', transform=ax2.transAxes,
                fontsize=12, fontweight='bold', ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle('MRI Report Parser - Score Extraction Accuracy', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: score_scatter.png")

    def plot_error_distribution(self):
        """Histogram of score errors"""
        results = self.data['results']

        vai_errors = [r['vai_error'] for r in results if r['vai_error'] is not None]
        magnifi_errors = [r['magnifi_error'] for r in results if r['magnifi_error'] is not None]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # VAI error distribution
        ax1 = axes[0]
        bins = np.arange(0, max(vai_errors) + 2, 1)
        ax1.hist(vai_errors, bins=bins, color='#3498db', edgecolor='white', linewidth=1.2, alpha=0.8)
        ax1.axvline(x=2, color='#e74c3c', linestyle='--', linewidth=2, label='Threshold (+/-2)')
        ax1.set_xlabel('Absolute Error (points)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('VAI Score Error Distribution', fontsize=14, fontweight='bold')
        ax1.legend()

        # Add statistics
        within_2 = self.data['summary']['vai_within_2'] * 100
        ax1.text(0.95, 0.95, f'Within +/-2: {within_2:.1f}%\nMAE: {self.data["summary"]["vai_mae"]:.2f}',
                transform=ax1.transAxes, fontsize=11, fontweight='bold', ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # MAGNIFI error distribution
        ax2 = axes[1]
        bins = np.arange(0, max(magnifi_errors) + 2, 1)
        ax2.hist(magnifi_errors, bins=bins, color='#9b59b6', edgecolor='white', linewidth=1.2, alpha=0.8)
        ax2.axvline(x=2, color='#e74c3c', linestyle='--', linewidth=2, label='Threshold (+/-2)')
        ax2.set_xlabel('Absolute Error (points)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('MAGNIFI-CD Score Error Distribution', fontsize=14, fontweight='bold')
        ax2.legend()

        # Add statistics
        within_2 = self.data['summary']['magnifi_within_2'] * 100
        ax2.text(0.95, 0.95, f'Within +/-2: {within_2:.1f}%\nMAE: {self.data["summary"]["magnifi_mae"]:.2f}',
                transform=ax2.transAxes, fontsize=11, fontweight='bold', ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.suptitle('MRI Report Parser - Score Error Distribution', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: error_distribution.png")

    def plot_confusion_matrices(self):
        """Confusion matrices for categorical features"""
        results = self.data['results']

        # Features to create confusion matrices for
        categorical_features = ['fistula_type', 't2_hyperintensity', 'extension', 'predominant_feature']

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, feature in enumerate(categorical_features):
            ax = axes[idx]

            # Collect all unique values
            expected_vals = []
            extracted_vals = []

            for r in results:
                exp = r['features_expected'].get(feature)
                ext = r['features_extracted'].get(feature)

                # Normalize
                exp = str(exp).lower() if exp else 'null'
                ext = str(ext).lower() if ext else 'null'

                expected_vals.append(exp)
                extracted_vals.append(ext)

            # Get unique categories
            all_categories = sorted(set(expected_vals) | set(extracted_vals))

            # Build confusion matrix
            n = len(all_categories)
            cat_to_idx = {cat: i for i, cat in enumerate(all_categories)}
            matrix = np.zeros((n, n), dtype=int)

            for exp, ext in zip(expected_vals, extracted_vals):
                matrix[cat_to_idx[exp], cat_to_idx[ext]] += 1

            # Plot heatmap
            im = ax.imshow(matrix, cmap='Blues', aspect='auto')

            # Add text annotations
            for i in range(n):
                for j in range(n):
                    color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
                    ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                           fontsize=10, fontweight='bold', color=color)

            # Labels
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))

            # Truncate long labels
            labels = [c[:10] + '..' if len(c) > 12 else c for c in all_categories]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(labels, fontsize=9)

            ax.set_xlabel('Extracted', fontsize=11, fontweight='bold')
            ax.set_ylabel('Expected', fontsize=11, fontweight='bold')
            ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')

        plt.suptitle('MRI Report Parser - Feature Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: confusion_matrices.png")

    def plot_difficulty_breakdown(self):
        """Bar chart of accuracy by difficulty level"""
        acc_by_diff = self.data['summary']['accuracy_by_difficulty']

        difficulties = list(acc_by_diff.keys())
        accuracies = [acc_by_diff[d] * 100 for d in difficulties]

        # Sort by intended order
        order = ['standard', 'edge_case', 'complex', 'ambiguous']
        sorted_pairs = []
        for d in order:
            if d in difficulties:
                sorted_pairs.append((d, acc_by_diff[d] * 100))
        for d in difficulties:
            if d not in order:
                sorted_pairs.append((d, acc_by_diff[d] * 100))

        difficulties, accuracies = zip(*sorted_pairs) if sorted_pairs else ([], [])

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'][:len(difficulties)]
        bars = ax.bar(difficulties, accuracies, color=colors, edgecolor='white', linewidth=2)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 2,
                   f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')

        ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('MRI Report Parser - Accuracy by Test Case Difficulty', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.axhline(y=80, color='#27ae60', linestyle='--', linewidth=2, label='Target (80%)')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'difficulty_breakdown.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: difficulty_breakdown.png")

    def plot_traceability_analysis(self):
        """Bar chart of traceability scores by case"""
        results = self.data['results']

        case_ids = [r['case_id'] for r in results]
        trace_scores = [r['traceability_score'] * 100 for r in results]
        difficulties = [r['difficulty'] for r in results]

        fig, ax = plt.subplots(figsize=(14, 6))

        # Color by difficulty
        diff_colors = {
            'standard': '#2ecc71',
            'edge_case': '#f39c12',
            'complex': '#e74c3c',
            'ambiguous': '#9b59b6'
        }
        colors = [diff_colors.get(d, '#3498db') for d in difficulties]

        bars = ax.bar(range(len(case_ids)), trace_scores, color=colors, edgecolor='white', linewidth=1)

        ax.set_xlabel('Test Case ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Traceability Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('MRI Report Parser - Traceability Score by Test Case', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(case_ids)))
        ax.set_xticklabels(case_ids)
        ax.set_ylim(0, 110)
        ax.axhline(y=self.data['summary']['avg_traceability'] * 100, color='#3498db',
                  linestyle='--', linewidth=2, label=f'Average ({self.data["summary"]["avg_traceability"]*100:.1f}%)')

        # Legend
        legend_elements = [mpatches.Patch(color=c, label=d.replace('_', ' ').title())
                         for d, c in diff_colors.items()]
        legend_elements.append(plt.Line2D([0], [0], color='#3498db', linestyle='--', linewidth=2, label='Average'))
        ax.legend(handles=legend_elements, loc='upper right', ncol=2)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'traceability_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: traceability_analysis.png")

    def plot_summary_dashboard(self):
        """Single summary dashboard with key metrics"""
        s = self.data['summary']

        fig = plt.figure(figsize=(16, 10))

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('MRI Report Parser Validation Dashboard', fontsize=18, fontweight='bold', y=0.98)

        # 1. Overall metrics (top left - spans 2 cols)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('off')

        metrics_text = f"""
OVERALL METRICS

Pass Rate: {s['passed_cases']}/{s['total_cases']} ({s['pass_rate']*100:.1f}%)
Feature Accuracy: {s['overall_feature_accuracy']*100:.1f}%
Traceability: {s['avg_traceability']*100:.1f}%

VAI MAE: {s['vai_mae']:.2f} points
VAI Within +/-2: {s['vai_within_2']*100:.1f}%

MAGNIFI MAE: {s['magnifi_mae']:.2f} points
MAGNIFI Within +/-2: {s['magnifi_within_2']*100:.1f}%

Avg Extraction Time: {s['avg_extraction_time']:.2f}s
"""
        ax1.text(0.1, 0.9, metrics_text, transform=ax1.transAxes, fontsize=13,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

        # 2. Pass/Fail pie chart (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        passed = s['passed_cases']
        failed = s['total_cases'] - passed
        ax2.pie([passed, failed], labels=['Passed', 'Failed'],
               colors=['#2ecc71', '#e74c3c'], autopct='%1.0f%%',
               explode=(0.05, 0), shadow=True, startangle=90,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax2.set_title('Test Results', fontsize=12, fontweight='bold')

        # 3. Feature accuracy bar (middle, spans all cols)
        ax3 = fig.add_subplot(gs[1, :])
        features = list(s['feature_accuracy'].keys())
        accuracies = [s['feature_accuracy'][f] * 100 for f in features]
        sorted_pairs = sorted(zip(features, accuracies), key=lambda x: -x[1])
        features, accuracies = zip(*sorted_pairs)

        colors = ['#2ecc71' if a >= 80 else '#f39c12' if a >= 60 else '#e74c3c' for a in accuracies]
        ax3.barh(features, accuracies, color=colors, edgecolor='white')
        ax3.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Per-Feature Accuracy', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 105)
        ax3.axvline(x=80, color='#27ae60', linestyle='--', linewidth=2)
        for i, (f, a) in enumerate(zip(features, accuracies)):
            ax3.text(a + 1, i, f'{a:.0f}%', va='center', fontsize=9, fontweight='bold')

        # 4. Difficulty breakdown (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        diff_acc = s['accuracy_by_difficulty']
        diffs = list(diff_acc.keys())
        accs = [diff_acc[d] * 100 for d in diffs]
        diff_colors = {'standard': '#2ecc71', 'edge_case': '#f39c12', 'complex': '#e74c3c', 'ambiguous': '#9b59b6'}
        colors = [diff_colors.get(d, '#3498db') for d in diffs]
        ax4.bar(diffs, accs, color=colors, edgecolor='white')
        ax4.set_ylabel('Pass Rate (%)', fontsize=10, fontweight='bold')
        ax4.set_title('By Difficulty', fontsize=11, fontweight='bold')
        ax4.set_ylim(0, 110)
        ax4.tick_params(axis='x', rotation=45)

        # 5. Score errors (bottom middle)
        ax5 = fig.add_subplot(gs[2, 1])
        results = self.data['results']
        vai_errors = [r['vai_error'] for r in results if r['vai_error'] is not None]
        magnifi_errors = [r['magnifi_error'] for r in results if r['magnifi_error'] is not None]
        ax5.boxplot([vai_errors, magnifi_errors], labels=['VAI', 'MAGNIFI-CD'])
        ax5.set_ylabel('Absolute Error', fontsize=10, fontweight='bold')
        ax5.set_title('Score Error Distribution', fontsize=11, fontweight='bold')
        ax5.axhline(y=2, color='#e74c3c', linestyle='--', linewidth=1.5, label='+/-2 threshold')

        # 6. Timing (bottom right)
        ax6 = fig.add_subplot(gs[2, 2])
        extraction_times = [r['extraction_time'] for r in results]
        ax6.hist(extraction_times, bins=10, color='#3498db', edgecolor='white', alpha=0.8)
        ax6.axvline(x=s['avg_extraction_time'], color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'Avg: {s["avg_extraction_time"]:.1f}s')
        ax6.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax6.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax6.set_title('Extraction Time', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=9)

        # Footer
        fig.text(0.5, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} | MRI-Crohn Atlas ISEF 2026',
                ha='center', fontsize=10, style='italic', color='#7f8c8d')

        plt.savefig(self.output_dir / 'validation_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: validation_dashboard.png")

    def generate_all_plots(self):
        """Generate all analysis plots"""
        print("\nGenerating error analysis plots...")
        print("-" * 40)

        self.plot_feature_accuracy_bar()
        self.plot_score_scatter()
        self.plot_error_distribution()
        self.plot_confusion_matrices()
        self.plot_difficulty_breakdown()
        self.plot_traceability_analysis()
        self.plot_summary_dashboard()

        print("-" * 40)
        print(f"All plots saved to: {self.output_dir}")


def main():
    """Main entry point"""
    project_root = Path(__file__).parent.parent.parent
    results_path = project_root / "data" / "parser_tests" / "validation_results.json"
    output_dir = project_root / "data" / "parser_tests" / "plots"

    if not results_path.exists():
        print(f"Error: Validation results not found at {results_path}")
        print("Run validate_parser.py first to generate results.")
        sys.exit(1)

    analyzer = ParserErrorAnalysis(results_path)
    analyzer.load_results()
    analyzer.setup_output_dir(output_dir)
    analyzer.generate_all_plots()


if __name__ == "__main__":
    main()
