# evaluate.py - Complete evaluation with ground truth

import pandas as pd
import numpy as np
import json
import logging
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    cohen_kappa_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Label mapping
LABEL2ID = {
    "cultural agnostic": 0,
    "cultural representative": 1,
    "cultural exclusive": 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

class ModelEvaluator:
    """Complete model evaluation with ground truth"""
    
    def __init__(self, test_csv_path, predictions_csv_path):
        logger.info("="*80)
        logger.info(" MODEL EVALUATION WITH GROUND TRUTH")
        logger.info("="*80)
        
        # Load test data with ground truth
        logger.info(f"\n Loading test data from: {test_csv_path}")
        self.test_df = pd.read_csv(test_csv_path)
        logger.info(f" Loaded {len(self.test_df)} test samples with ground truth")
        
        # Load predictions
        logger.info(f" Loading predictions from: {predictions_csv_path}")
        self.pred_df = pd.read_csv(predictions_csv_path)
        logger.info(f" Loaded {len(self.pred_df)} predictions")
        
        # Merge
        self.eval_df = self.test_df[['item', 'name', 'label']].copy()
        self.eval_df = self.eval_df.merge(
            self.pred_df[['item', 'predicted_label', 'confidence']], 
            on='item', 
            how='inner'
        )
        
        logger.info(f" Merged {len(self.eval_df)} samples for evaluation")
        
        # Get true and predicted labels
        self.y_true = self.eval_df['label'].values
        self.y_pred = self.eval_df['predicted_label'].values
        self.confidences = self.eval_df['confidence'].values
        
        # Label distribution
        logger.info("\n Ground Truth Distribution:")
        for label, count in self.test_df['label'].value_counts().items():
            logger.info(f"  {label:30s}: {count:4d} ({count/len(self.test_df)*100:.1f}%)")
    
    def compute_metrics(self):
        """Compute all evaluation metrics"""
        logger.info("\n" + "="*80)
        logger.info(" EVALUATION METRICS")
        logger.info("="*80)
        
        metrics = {}
        
        # Overall metrics
        accuracy = accuracy_score(self.y_true, self.y_pred)
        
        # Macro and weighted F1
        f1_macro = f1_score(self.y_true, self.y_pred, average='macro')
        f1_weighted = f1_score(self.y_true, self.y_pred, average='weighted')
        
        # Precision and Recall
        precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, _, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='weighted', zero_division=0
        )
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(self.y_true, self.y_pred)
        
        metrics['accuracy'] = accuracy
        metrics['f1_macro'] = f1_macro
        metrics['f1_weighted'] = f1_weighted
        metrics['precision_macro'] = precision_macro
        metrics['precision_weighted'] = precision_weighted
        metrics['recall_macro'] = recall_macro
        metrics['recall_weighted'] = recall_weighted
        metrics['cohen_kappa'] = kappa
        
        logger.info("\n Overall Metrics:")
        logger.info("-" * 80)
        logger.info(f"{'Accuracy':25s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"{'F1 Score (Macro)':25s}: {f1_macro:.4f}")
        logger.info(f"{'F1 Score (Weighted)':25s}: {f1_weighted:.4f}")
        logger.info(f"{'Precision (Macro)':25s}: {precision_macro:.4f}")
        logger.info(f"{'Precision (Weighted)':25s}: {precision_weighted:.4f}")
        logger.info(f"{'Recall (Macro)':25s}: {recall_macro:.4f}")
        logger.info(f"{'Recall (Weighted)':25s}: {recall_weighted:.4f}")
        logger.info(f"{'Cohen\'s Kappa':25s}: {kappa:.4f}")
        
        return metrics
    
    def per_class_metrics(self):
        """Compute per-class metrics"""
        logger.info("\n Per-Class Metrics:")
        logger.info("-" * 80)
        
        # Get per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None, zero_division=0
        )
        
        per_class = {}
        
        # Get unique labels in order
        labels = sorted(list(set(self.y_true) | set(self.y_pred)))
        
        logger.info(f"\n{'Class':30s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
        logger.info("-" * 80)
        
        for i, label in enumerate(labels):
            per_class[label] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
            
            logger.info(
                f"{label:30s} {precision[i]:10.4f} {recall[i]:10.4f} "
                f"{f1[i]:10.4f} {int(support[i]):10d}"
            )
        
        return per_class
    
    def confusion_matrix_analysis(self, save_plot=True):
        """Generate and analyze confusion matrix"""
        logger.info("\n Confusion Matrix:")
        logger.info("-" * 80)
        
        # Compute confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Get unique labels
        labels = sorted(list(set(self.y_true) | set(self.y_pred)))
        
        # Print confusion matrix
        logger.info(f"\n{'':30s}" + "".join([f"{label[:10]:>12s}" for label in labels]))
        logger.info("-" * (30 + 12 * len(labels)))
        
        for i, true_label in enumerate(labels):
            row_str = f"{true_label:30s}"
            for j, pred_label in enumerate(labels):
                row_str += f"{cm[i][j]:12d}"
            logger.info(row_str)
        
        # Normalized confusion matrix (percentages)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        logger.info("\n Confusion Matrix (Normalized - Row Percentages):")
        logger.info("-" * 80)
        logger.info(f"\n{'':30s}" + "".join([f"{label[:10]:>12s}" for label in labels]))
        logger.info("-" * (30 + 12 * len(labels)))
        
        for i, true_label in enumerate(labels):
            row_str = f"{true_label:30s}"
            for j, pred_label in enumerate(labels):
                row_str += f"{cm_normalized[i][j]:11.1%} "
            logger.info(row_str)
        
        # Plot confusion matrix
        if save_plot:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Absolute counts
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot 1: Absolute counts
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=[l.replace(' ', '\n') for l in labels],
                           yticklabels=[l.replace(' ', '\n') for l in labels],
                           ax=ax1, cbar_kws={'label': 'Count'})
                ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
                ax1.set_ylabel('True Label', fontsize=12)
                ax1.set_xlabel('Predicted Label', fontsize=12)
                
                # Plot 2: Normalized
                sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                           xticklabels=[l.replace(' ', '\n') for l in labels],
                           yticklabels=[l.replace(' ', '\n') for l in labels],
                           ax=ax2, cbar_kws={'label': 'Percentage'})
                ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
                ax2.set_ylabel('True Label', fontsize=12)
                ax2.set_xlabel('Predicted Label', fontsize=12)
                
                plt.tight_layout()
                plt.savefig('./predictions/confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info("\n Confusion matrix plot saved to: ./predictions/confusion_matrix.png")
                
            except Exception as e:
                logger.warning(f"  Could not generate confusion matrix plot: {e}")
        
        return cm, cm_normalized
    
    def classification_report_detailed(self):
        """Generate detailed classification report"""
        logger.info("\n Detailed Classification Report:")
        logger.info("-" * 80)
        
        report = classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=sorted(list(set(self.y_true) | set(self.y_pred))),
            digits=4
        )
        
        logger.info(f"\n{report}")
        
        return report
    
    def error_analysis(self):
        """Analyze misclassified samples"""
        logger.info("\n Error Analysis:")
        logger.info("-" * 80)
        
        # Find misclassified samples
        errors = self.eval_df[self.eval_df['label'] != self.eval_df['predicted_label']].copy()
        
        logger.info(f"\n Total Errors: {len(errors)} / {len(self.eval_df)} ({len(errors)/len(self.eval_df)*100:.2f}%)")
        
        if len(errors) == 0:
            logger.info(" Perfect predictions! No errors found.")
            return {}
        
        # Error breakdown by true class
        logger.info("\n Errors by True Class:")
        logger.info("-" * 80)
        
        error_analysis = {}
        
        for true_label in sorted(errors['label'].unique()):
            class_errors = errors[errors['label'] == true_label]
            total_class = len(self.eval_df[self.eval_df['label'] == true_label])
            error_rate = len(class_errors) / total_class * 100
            
            logger.info(f"\n{true_label}:")
            logger.info(f"  Errors: {len(class_errors)} / {total_class} ({error_rate:.1f}%)")
            
            # Where are they being predicted?
            pred_dist = class_errors['predicted_label'].value_counts()
            logger.info(f"  Misclassified as:")
            for pred_label, count in pred_dist.items():
                logger.info(f"    → {pred_label}: {count} ({count/len(class_errors)*100:.1f}%)")
            
            error_analysis[true_label] = {
                'total_errors': len(class_errors),
                'total_samples': total_class,
                'error_rate': error_rate,
                'misclassified_as': pred_dist.to_dict()
            }
        
        # Show examples of errors with lowest confidence
        logger.info("\n Top 10 Errors (Lowest Confidence):")
        logger.info("-" * 80)
        
        errors_sorted = errors.nsmallest(10, 'confidence')
        for idx, row in errors_sorted.iterrows():
            logger.info(
                f"\n{row['name'][:50]}\n"
                f"  True: {row['label']:30s} | Predicted: {row['predicted_label']:30s}\n"
                f"  Confidence: {row['confidence']:.4f}"
            )
        
        # Save all errors to CSV
        errors_file = './predictions/errors_analysis.csv'
        errors.to_csv(errors_file, index=False)
        logger.info(f"\n All errors saved to: {errors_file}")
        
        return error_analysis
    
    def confidence_analysis(self):
        """Analyze confidence scores"""
        logger.info("\n Confidence Analysis:")
        logger.info("-" * 80)
        
        # Overall confidence
        logger.info(f"\nOverall Confidence Statistics:")
        logger.info(f"  Mean: {self.confidences.mean():.4f}")
        logger.info(f"  Std:  {self.confidences.std():.4f}")
        logger.info(f"  Min:  {self.confidences.min():.4f}")
        logger.info(f"  Max:  {self.confidences.max():.4f}")
        
        # Confidence for correct vs incorrect predictions
        correct_mask = self.eval_df['label'] == self.eval_df['predicted_label']
        correct_conf = self.eval_df[correct_mask]['confidence']
        incorrect_conf = self.eval_df[~correct_mask]['confidence']
        
        logger.info(f"\n Correct Predictions (n={len(correct_conf)}):")
        logger.info(f"  Mean Confidence: {correct_conf.mean():.4f} ± {correct_conf.std():.4f}")
        
        logger.info(f"\n Incorrect Predictions (n={len(incorrect_conf)}):")
        if len(incorrect_conf) > 0:
            logger.info(f"  Mean Confidence: {incorrect_conf.mean():.4f} ± {incorrect_conf.std():.4f}")
        else:
            logger.info(f"  No incorrect predictions!")
        
        # Confidence by class (for correct predictions)
        logger.info(f"\n Confidence by Class (Correct Predictions Only):")
        logger.info("-" * 80)
        
        for label in sorted(self.eval_df['label'].unique()):
            class_correct = self.eval_df[
                (self.eval_df['label'] == label) & 
                (self.eval_df['label'] == self.eval_df['predicted_label'])
            ]
            if len(class_correct) > 0:
                logger.info(
                    f"  {label:30s}: {class_correct['confidence'].mean():.4f} ± "
                    f"{class_correct['confidence'].std():.4f} (n={len(class_correct)})"
                )
    
    def generate_complete_report(self, output_file='./predictions/evaluation_report.json'):
        """Generate complete evaluation report"""
        logger.info("\n" + "="*80)
        logger.info(" GENERATING COMPLETE REPORT")
        logger.info("="*80)
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_size': len(self.eval_df),
            'overall_metrics': self.compute_metrics(),
            'per_class_metrics': self.per_class_metrics(),
            'error_analysis': self.error_analysis()
        }
        
        # Confusion matrix
        cm, cm_norm = self.confusion_matrix_analysis()
        report['confusion_matrix'] = cm.tolist()
        report['confusion_matrix_normalized'] = cm_norm.tolist()
        
        # Classification report
        report['classification_report'] = self.classification_report_detailed()
        
        # Confidence analysis
        self.confidence_analysis()
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n Complete report saved to: {output_file}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info(" EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"\n Accuracy:        {report['overall_metrics']['accuracy']:.4f} ({report['overall_metrics']['accuracy']*100:.2f}%)")
        logger.info(f" F1 Macro:        {report['overall_metrics']['f1_macro']:.4f}")
        logger.info(f" F1 Weighted:     {report['overall_metrics']['f1_weighted']:.4f}")
        logger.info(f" Cohen's Kappa:   {report['overall_metrics']['cohen_kappa']:.4f}")
        
        errors = len(self.eval_df) - (self.eval_df['label'] == self.eval_df['predicted_label']).sum()
        logger.info(f"\n Total Errors:    {errors} / {len(self.eval_df)} ({errors/len(self.eval_df)*100:.2f}%)")
        
        logger.info("\n" + "="*80)
        
        return report

def main():
    """Main evaluation function"""
    
    # Find latest prediction file
    import glob
    pred_files = glob.glob('./predictions/predictions_*.csv')
    
    if not pred_files:
        logger.error(" No prediction files found!")
        logger.info("Please run inference.py first")
        return
    
    latest_pred = max(pred_files)
    logger.info(f"Using predictions from: {latest_pred}")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        test_csv_path='test.csv',
        predictions_csv_path=latest_pred
    )
    
    # Generate complete report
    report = evaluator.generate_complete_report()
    
    return report

if __name__ == "__main__":
    report = main()