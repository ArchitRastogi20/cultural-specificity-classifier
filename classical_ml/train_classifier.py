"""
Non-LM Cultural Classifier with GPU-accelerated XGBoost
Optimized for RTX 3090 with best practices
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import xgboost as xgb
from xgboost.callback import TrainingCallback
import json
import pickle
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TqdmCallback(TrainingCallback):
    """Custom callback for progress bar"""
    def __init__(self, pbar):
        self.pbar = pbar
        
    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        if epoch % 50 == 0:
            if evals_log:
                train_metric = list(evals_log['train'].values())[0][-1.
                msg = f"Round {epoch}: train-mlogloss={train_metric:.4f}"
                if 'validation' in evals_log:
                    val_metric = list(evals_log['validation'].values())[0][-1.
                    msg += f", val-mlogloss={val_metric:.4f}"
                self.pbar.set_postfix_str(msg)
        return False


class CulturalClassifier:
    """GPU-accelerated XGBoost classifier for cultural specificity"""
    
    def __init__(self, use_gpu=True):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None
        self.use_gpu = use_gpu
        
    def engineer_features(self, df: pd.DataFrame) -> tuple:
        """Advanced feature engineering"""
        logger.info(" Engineering features...")
        
        feature_df = df.copy()
        
        # Log transforms for skewed distributions
        logger.info("  → Log transformations...")
        feature_df['log_num_languages'] = np.log1p(feature_df['num_languages'])
        feature_df['log_en_page_length'] = np.log1p(feature_df['en_page_length'])
        feature_df['log_num_statements'] = np.log1p(feature_df['num_statements'])
        feature_df['log_num_categories'] = np.log1p(feature_df['num_categories'])
        feature_df['log_num_external_links'] = np.log1p(feature_df['num_external_links'])
        feature_df['log_num_identifiers'] = np.log1p(feature_df['num_identifiers'])
        feature_df['log_statement_diversity'] = np.log1p(feature_df['statement_diversity'])
        
        # Ratio features (normalized)
        logger.info("  → Ratio features...")
        feature_df['cultural_ratio'] = feature_df['num_cultural_properties'] / (feature_df['statement_diversity'] + 1)
        feature_df['geographic_ratio'] = feature_df['num_geographic_properties'] / (feature_df['statement_diversity'] + 1)
        feature_df['identifier_ratio'] = feature_df['num_identifiers'] / (feature_df['num_statements'] + 1)
        feature_df['categories_per_page'] = feature_df['num_categories'] / (feature_df['en_page_length'] + 1)
        feature_df['external_links_per_page'] = feature_df['num_external_links'] / (feature_df['en_page_length'] + 1)
        
        # Interaction features
        logger.info("  → Interaction features...")
        feature_df['languages_x_statements'] = feature_df['log_num_languages'] * feature_df['log_num_statements']
        feature_df['languages_x_page_length'] = feature_df['log_num_languages'] * feature_df['log_en_page_length']
        feature_df['has_country_x_languages'] = feature_df['has_country'].astype(int) * feature_df['log_num_languages']
        feature_df['has_country_x_statements'] = feature_df['has_country'].astype(int) * feature_df['log_num_statements']
        feature_df['cultural_x_geographic'] = feature_df['num_cultural_properties'] * feature_df['num_geographic_properties']
        
        # Composite scores
        logger.info("  → Composite scores...")
        
        # Global reach score (how widely known)
        feature_df['global_reach_score'] = (
            feature_df['log_num_languages'] * 0.5 +
            feature_df['log_en_page_length'] * 0.3 +
            feature_df['log_num_external_links'] * 0.2
        )
        
        # Cultural specificity score
        feature_df['cultural_specificity_score'] = (
            feature_df['num_cultural_properties'] * 2.0 +
            feature_df['num_geographic_properties'] * 1.5 +
            feature_df['has_country'].astype(int) * 1.0 +
            feature_df['has_origin_country'].astype(int) * 1.0 +
            feature_df['has_culture_property'].astype(int) * 2.0
        )
        
        # Information richness score
        feature_df['info_richness_score'] = (
            feature_df['log_num_statements'] * 0.4 +
            feature_df['log_statement_diversity'] * 0.4 +
            feature_df['log_num_identifiers'] * 0.2
        )
        
        # Page quality score
        feature_df['page_quality_score'] = (
            feature_df['log_en_page_length'] * 0.4 +
            feature_df['log_num_categories'] * 0.3 +
            feature_df['log_num_external_links'] * 0.3
        )
        
        # Binary threshold features
        logger.info("  → Binary threshold features...")
        feature_df['is_highly_global'] = (feature_df['num_languages'] > 20).astype(int)
        feature_df['is_niche'] = (feature_df['num_languages'] < 10).astype(int)
        feature_df['has_long_page'] = (feature_df['en_page_length'] > 10000).astype(int)
        feature_df['has_many_statements'] = (feature_df['num_statements'] > 30).astype(int)
        
        # Polynomial features for key indicators
        logger.info("  → Polynomial features...")
        feature_df['num_languages_squared'] = feature_df['log_num_languages'] ** 2
        feature_df['cultural_specificity_squared'] = feature_df['cultural_specificity_score'] ** 2
        
        # Select feature columns
        feature_cols = [
            # Original features (log-transformed)
            'log_num_languages', 'log_en_page_length', 'log_num_categories',
            'log_num_external_links', 'log_num_statements', 'log_num_identifiers',
            'log_statement_diversity',
            
            # Raw counts for cultural properties
            'num_cultural_properties', 'num_geographic_properties',
            
            # Boolean features
            'has_coordinates', 'has_country', 'has_culture_property', 'has_origin_country',
            
            # Ratio features
            'cultural_ratio', 'geographic_ratio', 'identifier_ratio',
            'categories_per_page', 'external_links_per_page',
            
            # Interaction features
            'languages_x_statements', 'languages_x_page_length',
            'has_country_x_languages', 'has_country_x_statements', 'cultural_x_geographic',
            
            # Composite scores
            'global_reach_score', 'cultural_specificity_score',
            'info_richness_score', 'page_quality_score',
            
            # Binary thresholds
            'is_highly_global', 'is_niche', 'has_long_page', 'has_many_statements',
            
            # Polynomial
            'num_languages_squared', 'cultural_specificity_squared'
        ]
        
        X = feature_df[feature_cols].fillna(0)
        self.feature_names = feature_cols
        
        logger.info(f" Created {len(feature_cols)} features")
        
        return X, feature_df
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model with GPU acceleration"""
        logger.info("\n" + "="*80)
        logger.info(" Training XGBoost Model")
        logger.info("="*80)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        if y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
        
        # Scale features
        logger.info(" Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # XGBoost parameters optimized for GPU
        params = {
            # GPU settings
            'tree_method': 'hist',
            'device': 'cuda' if self.use_gpu else 'cpu',
            
            # Model parameters
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            
            # Regularization (prevent overfitting)
            'max_depth': 8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            
            # Learning rate
            'learning_rate': 0.05,
            
            # Other
            'random_state': 42,
            'verbosity': 0
        }
        
        logger.info(f" Using device: {'GPU (CUDA)' if self.use_gpu else 'CPU'}")
        logger.info(f" Model parameters:")
        for key, value in params.items():
            if key not in ['objective', 'eval_metric', 'device', 'tree_method', 'verbosity', 'random_state']:
                logger.info(f"   {key}: {value}")
        
        # Create DMatrix for faster training
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train_encoded, feature_names=self.feature_names)
        
        eval_list = [(dtrain, 'train')]
        if X_val is not None:
            dval = xgb.DMatrix(X_val_scaled, label=y_val_encoded, feature_names=self.feature_names)
            eval_list.append((dval, 'validation'))
        
        # Progress bar
        num_rounds = 1000
        pbar = tqdm(total=num_rounds, desc="Training", unit="rounds")
        
        # Train model
        logger.info("\n Training in progress...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_rounds,
            evals=eval_list,
            early_stopping_rounds=50,
            verbose_eval=False,
            callbacks=[TqdmCallback(pbar)]
        )
        
        pbar.close()
        
        logger.info(f"\n Training completed!")
        logger.info(f" Best iteration: {self.model.best_iteration}")
        logger.info(f" Best score: {self.model.best_score:.4f}")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        dtest = xgb.DMatrix(X_scaled, feature_names=self.feature_names)
        y_pred_proba = self.model.predict(dtest)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return self.label_encoder.inverse_transform(y_pred), y_pred_proba
    
    def evaluate(self, X, y_true):
        """Evaluate model performance"""
        y_pred, y_pred_proba = self.predict(X)
        
        # Calculate metrics
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        
        return {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def get_feature_importance(self, importance_type='gain', top_n=20):
        """Get feature importance"""
        importance = self.model.get_score(importance_type=importance_type)
        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath='cultural_classifier.pkl'):
        """Save model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'use_gpu': self.use_gpu
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f" Model saved to {filepath}")
    
    def load_model(self, filepath='cultural_classifier.pkl'):
        """Load model and preprocessors"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.use_gpu = model_data['use_gpu']
        logger.info(f" Model loaded from {filepath}")


def main():
    """Main training pipeline"""
    logger.info("="*80)
    logger.info(" Cultural Classification Training Pipeline")
    logger.info("="*80)
    
    # Load data
    logger.info("\n Loading datasets...")
    try:
        df_train = pd.read_csv('train_enriched.csv')
        logger.info(f" Loaded training set: {len(df_train)} samples")
        
        # Use 80/20 split for train/validation
        from sklearn.model_selection import train_test_split
        df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['label'])
        logger.info(f" Split: train={len(df_train)}, validation={len(df_val)}")
        
    except Exception as e:
        logger.error(f" Error loading data: {e}")
        return
    
    # Display label distribution
    logger.info("\n Label Distribution:")
    logger.info("Training set:")
    for label, count in df_train['label'].value_counts().items():
        logger.info(f"  {label}: {count} ({count/len(df_train)*100:.1f}%)")
    logger.info("Validation set:")
    for label, count in df_val['label'].value_counts().items():
        logger.info(f"  {label}: {count} ({count/len(df_val)*100:.1f}%)")
    
    # Initialize classifier
    classifier = CulturalClassifier(use_gpu=True)
    
    # Engineer features
    X_train, _ = classifier.engineer_features(df_train)
    X_val, _ = classifier.engineer_features(df_val)
    
    y_train = df_train['label']
    y_val = df_val['label']
    
    # Train model
    classifier.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on training set
    logger.info("\n" + "="*80)
    logger.info(" Training Set Performance")
    logger.info("="*80)
    train_results = classifier.evaluate(X_train, y_train)
    logger.info(f"F1 Score (Macro):    {train_results['f1_macro']:.4f}")
    logger.info(f"F1 Score (Weighted): {train_results['f1_weighted']:.4f}")
    logger.info(f"Precision (Macro):   {train_results['precision_macro']:.4f}")
    logger.info(f"Recall (Macro):      {train_results['recall_macro']:.4f}")
    
    # Evaluate on validation set
    logger.info("\n" + "="*80)
    logger.info(" Validation Set Performance")
    logger.info("="*80)
    val_results = classifier.evaluate(X_val, y_val)
    logger.info(f"F1 Score (Macro):    {val_results['f1_macro']:.4f}")
    logger.info(f"F1 Score (Weighted): {val_results['f1_weighted']:.4f}")
    logger.info(f"Precision (Macro):   {val_results['precision_macro']:.4f}")
    logger.info(f"Recall (Macro):      {val_results['recall_macro']:.4f}")
    
    # Per-class metrics
    logger.info("\n Per-Class Metrics (Validation):")
    report = val_results['classification_report']
    for label in sorted(report.keys()):
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics = report[label]
            logger.info(f"\n  {label.upper()}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall:    {metrics['recall']:.4f}")
            logger.info(f"    F1-Score:  {metrics['f1-score']:.4f}")
            logger.info(f"    Support:   {int(metrics['support'])}")
    
    # Feature importance
    logger.info("\n" + "="*80)
    logger.info(" Top 20 Most Important Features")
    logger.info("="*80)
    importance_df = classifier.get_feature_importance(importance_type='gain', top_n=20)
    for idx, row in importance_df.iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:>10.2f}")
    
    # Save model
    logger.info("\n" + "="*80)
    logger.info(" Saving Model")
    logger.info("="*80)
    classifier.save_model('cultural_classifier.pkl')
    
    # Save predictions
    logger.info("\n Saving predictions...")
    val_predictions_df = df_val[['item', 'name', 'label']].copy()
    val_predictions_df['predicted_label'] = val_results['predictions']
    val_predictions_df['correct'] = val_predictions_df['label'] == val_predictions_df['predicted_label']
    
    # Add probabilities
    for i, label in enumerate(classifier.label_encoder.classes_):
        val_predictions_df[f'prob_{label}'] = val_results['probabilities'][:, i]
    
    val_predictions_df.to_csv('validation_predictions.csv', index=False)
    logger.info(" Saved validation_predictions.csv")
    
    # Save metrics
    metrics_dict = {
        'training': {
            'f1_macro': float(train_results['f1_macro']),
            'f1_weighted': float(train_results['f1_weighted']),
            'precision_macro': float(train_results['precision_macro']),
            'recall_macro': float(train_results['recall_macro'])
        },
        'validation': {
            'f1_macro': float(val_results['f1_macro']),
            'f1_weighted': float(val_results['f1_weighted']),
            'precision_macro': float(val_results['precision_macro']),
            'recall_macro': float(val_results['recall_macro'])
        },
        'per_class_validation': {
            label: {
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1-score': float(metrics['f1-score']),
                'support': int(metrics['support'])
            }
            for label, metrics in report.items()
            if label not in ['accuracy', 'macro avg', 'weighted avg']
        }
    }
    
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(" Saved training_metrics.json")
    
    logger.info("\n" + "="*80)
    logger.info(" Training Pipeline Completed Successfully!")
    logger.info(" Output files:")
    logger.info("   • cultural_classifier.pkl (trained model)")
    logger.info("   • validation_predictions.csv (predictions with probabilities)")
    logger.info("   • training_metrics.json (detailed metrics)")
    logger.info("   • training.log (detailed logs)")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()