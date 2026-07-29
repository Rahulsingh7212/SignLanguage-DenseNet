# ============================================================
# src/models/evaluator.py
# Model Evaluation: Confusion Matrix, F1, Per-Class Metrics
# ============================================================

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

import tensorflow as tf
from tensorflow import keras

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


class ModelEvaluator:
    """
    Comprehensive model evaluation on test set.
    Generates:
    - Confusion matrix (full + normalized)
    - Per-class F1, Precision, Recall
    - Most confused class pairs
    - Sample wrong predictions
    """

    def __init__(self, model: keras.Model):
        self.model      = model
        self.classes    = config.CLASSES
        self.num_classes = config.NUM_CLASSES

    # ── PREDICTION ───────────────────────────────────────────
    def predict(
        self,
        test_generator,
        verbose: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run predictions on test generator.
        Returns (y_true, y_pred) as class indices.
        """
        print("  🔮 Running predictions on test set...")

        # Get predictions
        predictions = self.model.predict(
            test_generator,
            verbose=verbose
        )

        # Predicted class indices
        y_pred = np.argmax(predictions, axis=1)

        # True class indices (from generator)
        y_true = test_generator.classes

        print(f"  ✅ Predictions complete: {len(y_pred)} images")

        return y_true, y_pred, predictions

    # ── METRICS ──────────────────────────────────────────────
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Compute overall and per-class metrics."""

        metrics = {
            'accuracy'          : accuracy_score(y_true, y_pred),
            'macro_f1'          : f1_score(y_true, y_pred, average='macro'),
            'weighted_f1'       : f1_score(y_true, y_pred, average='weighted'),
            'macro_precision'   : precision_score(y_true, y_pred, average='macro'),
            'macro_recall'      : recall_score(y_true, y_pred, average='macro'),
        }

        # Per-class F1
        per_class_f1 = f1_score(y_true, y_pred, average=None)
        metrics['per_class_f1'] = {
            self.classes[i]: per_class_f1[i]
            for i in range(self.num_classes)
        }

        return metrics

    def print_metrics(self, metrics: Dict) -> None:
        """Print formatted metrics report."""
        print("\n" + "="*60)
        print("  📊 EVALUATION RESULTS")
        print("="*60)
        print(f"  Overall Accuracy:     {metrics['accuracy']*100:.2f}%")
        print(f"  Macro F1 Score:       {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1 Score:    {metrics['weighted_f1']:.4f}")
        print(f"  Macro Precision:      {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:         {metrics['macro_recall']:.4f}")
        print("="*60)

        # Per-class F1 sorted by worst
        print("\n  📉 Per-Class F1 Score (worst → best):")
        sorted_f1 = sorted(
            metrics['per_class_f1'].items(),
            key=lambda x: x[1]
        )
        for cls, f1 in sorted_f1:
            bar = '█' * int(f1 * 20)
            status = "⚠️" if f1 < 0.9 else "✅"
            print(f"     {status} {cls:<10} {f1:.4f}  {bar}")

    def save_classification_report(
        self,
        y_true  : np.ndarray,
        y_pred  : np.ndarray,
        save_path: Path = config.CLASSIFICATION_REPORT_PATH
    ) -> None:
        """Save full sklearn classification report to file."""
        report = classification_report(
            y_true,
            y_pred,
            target_names = self.classes,
            digits       = 4
        )
        save_path.write_text(report)
        print(f"  ✅ Classification report saved: {save_path.name}")
        print("\n" + report)

    # ── CONFUSION MATRIX ─────────────────────────────────────
    def plot_confusion_matrix(
        self,
        y_true      : np.ndarray,
        y_pred      : np.ndarray,
        normalize   : bool  = True,
        save_path   : Path  = config.CONFUSION_MATRIX_PATH
    ) -> None:
        """
        Plot confusion matrix — full 29×29 grid.
        Normalized version shows percentages per true class.
        """

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt     = '.2f'
            title   = 'Normalized Confusion Matrix (% of True Class)'
            vmax    = 1.0
        else:
            cm_plot = cm
            fmt     = 'd'
            title   = 'Confusion Matrix (Raw Counts)'
            vmax    = cm.max()

        # ── Plot ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(22, 20))

        sns.heatmap(
            cm_plot,
            annot        = True,
            fmt          = fmt,
            cmap         = 'Blues',
            xticklabels  = self.classes,
            yticklabels  = self.classes,
            ax           = ax,
            vmin         = 0,
            vmax         = vmax,
            linewidths   = 0.5,
            linecolor    = 'gray',
            annot_kws    = {'size': 8}
        )

        ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold', labelpad=15)
        ax.set_ylabel('True Class',      fontsize=14, fontweight='bold', labelpad=15)
        ax.set_title(title,              fontsize=16, fontweight='bold', pad=20)

        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0,  fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✅ Confusion matrix saved: {save_path.name}")

        plt.show()

    def plot_per_class_f1(
        self,
        metrics     : Dict,
        save_path   : Path = config.PER_CLASS_F1_PATH
    ) -> None:
        """
        Bar chart of per-class F1 scores.
        Color-coded: green (good) → red (needs improvement).
        """

        f1_data = metrics['per_class_f1']
        classes = list(f1_data.keys())
        f1_vals = list(f1_data.values())

        # Color by score
        colors = ['#2ecc71' if v >= 0.95
                  else '#f39c12' if v >= 0.85
                  else '#e74c3c'
                  for v in f1_vals]

        fig, ax = plt.subplots(figsize=(18, 8))

        bars = ax.bar(classes, f1_vals, color=colors, edgecolor='black',
                      linewidth=0.7, width=0.7)

        # Value labels on bars
        for bar, val in zip(bars, f1_vals):
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.005,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold'
            )

        # Reference lines
        ax.axhline(y=0.95, color='green',  linestyle='--',
                   alpha=0.7, linewidth=1.5, label='95% target')
        ax.axhline(y=0.85, color='orange', linestyle='--',
                   alpha=0.7, linewidth=1.5, label='85% warning')
        ax.axhline(y=metrics['accuracy'], color='blue', linestyle='-.',
                   alpha=0.7, linewidth=2,
                   label=f"Overall Acc: {metrics['accuracy']*100:.2f}%")

        ax.set_xlabel('Sign Language Class', fontsize=13, fontweight='bold')
        ax.set_ylabel('F1 Score',            fontsize=13, fontweight='bold')
        ax.set_title('Per-Class F1 Score — DenseNet121 ASL Recognition',
                     fontsize=15, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='F1 ≥ 0.95 (Excellent)'),
            Patch(facecolor='#f39c12', label='F1 ≥ 0.85 (Good)'),
            Patch(facecolor='#e74c3c', label='F1 < 0.85 (Needs work)'),
        ]
        ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0],
                  loc='lower right', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✅ Per-class F1 plot saved: {save_path.name}")

        plt.show()

    def find_confused_pairs(
        self,
        y_true  : np.ndarray,
        y_pred  : np.ndarray,
        top_n   : int = 10
    ) -> pd.DataFrame:
        """
        Find the most commonly confused class pairs.
        Critical for understanding model weaknesses.
        """
        cm = confusion_matrix(y_true, y_pred)
        np.fill_diagonal(cm, 0)     # Remove correct predictions

        # Find top confused pairs
        confused_pairs = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'True Class'     : self.classes[i],
                        'Predicted As'   : self.classes[j],
                        'Count'          : cm[i, j],
                        'Confusion Rate' : cm[i, j] / (y_true == i).sum()
                    })

        df = pd.DataFrame(confused_pairs)
        df = df.sort_values('Count', ascending=False).head(top_n)
        df = df.reset_index(drop=True)

        print(f"\n  🔍 TOP {top_n} CONFUSED CLASS PAIRS:")
        print(f"  {'Rank':<6} {'True':<12} {'→ Predicted As':<16} "
              f"{'Count':<8} {'Rate'}")
        print(f"  {'-'*55}")

        for idx, row in df.iterrows():
            print(f"  {idx+1:<6} {row['True Class']:<12} "
                  f"{'→ '+row['Predicted As']:<16} "
                  f"{row['Count']:<8} "
                  f"{row['Confusion Rate']*100:.1f}%")

        return df

    def plot_wrong_predictions(
        self,
        test_generator,
        y_true      : np.ndarray,
        y_pred      : np.ndarray,
        predictions : np.ndarray,
        n_samples   : int = 24,
        save_path   : Path = None
    ) -> None:
        """
        Visualize wrong predictions — shows what confused the model.
        """
        import cv2

        wrong_indices = np.where(y_true != y_pred)[0]
        if len(wrong_indices) == 0:
            print("  🎉 No wrong predictions!")
            return

        # Sample wrong predictions
        sample_idx = np.random.choice(
            wrong_indices,
            size=min(n_samples, len(wrong_indices)),
            replace=False
        )

        # Get file paths from generator
        filepaths = test_generator.filepaths

        cols = 6
        rows = (len(sample_idx) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, 3 * rows))
        fig.suptitle(
            f'❌ Wrong Predictions (showing {len(sample_idx)} of '
            f'{len(wrong_indices)} errors)',
            fontsize=15, fontweight='bold', color='red'
        )

        for plot_idx, img_idx in enumerate(sample_idx):
            row = plot_idx // cols
            col = plot_idx % cols

            ax = axes[row][col] if rows > 1 else axes[col]

            # Load image
            img = cv2.imread(filepaths[img_idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

            ax.imshow(img)

            true_cls  = self.classes[y_true[img_idx]]
            pred_cls  = self.classes[y_pred[img_idx]]
            confidence = predictions[img_idx][y_pred[img_idx]] * 100

            ax.set_title(
                f'True: {true_cls}\nPred: {pred_cls} ({confidence:.0f}%)',
                fontsize=8,
                color='red',
                fontweight='bold'
            )
            ax.axis('off')

        # Hide empty plots
        for i in range(len(sample_idx), rows * cols):
            r, c = i // cols, i % cols
            ax = axes[r][c] if rows > 1 else axes[c]
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✅ Wrong predictions plot saved: {save_path.name}")

        plt.show()