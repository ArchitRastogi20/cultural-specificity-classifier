"""
Hyperparameter Tuning for Cultural Classifier
Uses Optuna for efficient hyperparameter optimization with GPU
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
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
        logging.FileHandler('hyperparameter_tuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress Optuna's default logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class CulturalClassifierTuner:
    """Hyperparameter tuning for Cultural Classifier"""
    
    def __init__(self, X_train, y_train, X_val, y_val, use_gpu=True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.use_gpu = use_gpu
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.y_val_encoded = self.label_encoder.transform(y_val)
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_val_scaled = self.scaler.transform(X_val)
        
        self.best_params = None
        self.best_score = 0
        
    def objective(self, trial):
        """Objective function for Optuna"""
        
        # Suggest hyperparameters
        params = {
            'device': 'cuda' if self.use_gpu else 'cpu',
            'tree_method': 'hist',
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'verbosity': 0,
            'random_state': 42,
            
            # Hyperparameters to tune
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
        }
        
        # Create DMatrix
        dtrain = xgb.DMatrix(self.X_train_scaled, label=self.y_train_encoded)
        dval = xgb.DMatrix(self.X_val_scaled, label=self.y_val_encoded)
        
        # Train with early stopping
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Predict on validation set
        y_pred_proba = model.predict(dval)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate F1 macro score
        f1 = f1_score(self.y_val_encoded, y_pred, average='macro')
        
        return f1
    
    def tune(self, n_trials=100):
        """Run hyperparameter tuning"""
        logger.info("="*80)
        logger.info(" Starting Hyperparameter Tuning")
        logger.info("="*80)
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Using device: {'GPU (CUDA)' if self.use_gpu else 'CPU'}")
        logger.info("")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize with progress bar
        with tqdm(total=n_trials, desc="Tuning", unit="trial") as pbar:
            def callback(study, trial):
                pbar.update(1)
                pbar.set_postfix({
                    'best_f1': f'{study.best_value:.4f}',
                    'trial': trial.number
                })
            
            study.optimize(self.objective, n_trials=n_trials, callbacks=[callback])
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info("\n" + "="*80)
        logger.info(" Tuning Complete!")
        logger.info("="*80)
        logger.info(f"Best F1 Score (Macro): {self.best_score:.4f}")
        logger.info("\nBest Hyperparameters:")
        for key, value in self.best_params.items():
            logger.info(f"  {key}: {value}")
        
        return study


def engineer_features(df: pd.DataFrame, feature_names=None):
    """Engineer features (same as training)"""
    feature_df = df.copy()
    
    # Log transforms
    feature_df['log_num_languages'] = np.log1p(feature_df['num_languages'])
    feature_df['log_en_page_length'] = np.log1p(feature_df['en_page_length'])
    feature_df['log_num_statements'] = np.log1p(feature_df['num_statements'])
    feature_df['log_num_categories'] = np.log1p(feature_df['num_categories'])
    feature_df['log_num_external_links'] = np.log1p(feature_df['num_external_links'])
    feature_df['log_num_identifiers'] = np.log1p(feature_df['num_identifiers'])
    feature_df['log_statement_diversity'] = np.log1p(feature_df['statement_diversity'])
    
    # Ratio features
    feature_df['cultural_ratio'] = feature_df['num_cultural_properties'] / (feature_df['statement_diversity'] + 1)
    feature_df['geographic_ratio'] = feature_df['num_geographic_properties'] / (feature_df['statement_diversity'] + 1)
    feature_df['identifier_ratio'] = feature_df['num_identifiers'] / (feature_df['num_statements'] + 1)
    feature_df['categories_per_page'] = feature_df['num_categories'] / (feature_df['en_page_length'] + 1)
    feature_df['external_links_per_page'] = feature_df['num_external_links'] / (feature_df['en_page_length'] + 1)
    
    # Interaction features
    feature_df['languages_x_statements'] = feature_df['log_num_languages'] * feature_df['log_num_statements']
    feature_df['languages_x_page_length'] = feature_df['log_num_languages'] * feature_df['log_en_page_length']
    feature_df['has_country_x_languages'] = feature_df['has_country'].astype(int) * feature_df['log_num_languages']
    feature_df['has_country_x_statements'] = feature_df['has_country'].astype(int) * feature_df['log_num_statements']
    feature_df['cultural_x_geographic'] = feature_df['num_cultural_properties'] * feature_df['num_geographic_properties']
    
    # Composite scores
    feature_df['global_reach_score'] = (
        feature_df['log_num_languages'] * 0.5 +
        feature_df['log_en_page_length'] * 0.3 +
        feature_df['log_num_external_links'] * 0.2
    )
    
    feature_df['cultural_specificity_score'] = (
        feature_df['num_cultural_properties'] * 2.0 +
        feature_df['num_geographic_properties'] * 1.5 +
        feature_df['has_country'].astype(int) * 1.0 +
        feature_df['has_origin_country'].astype(int) * 1.0 +
        feature_df['has_culture_property'].astype(int) * 2.0
    )
    
    feature_df['info_richness_score'] = (
        feature_df['log_num_statements'] * 0.4 +
        feature_df['log_statement_diversity'] * 0.4 +
        feature_df['log_num_identifiers'] * 0.2
    )
    
    feature_df['page_quality_score'] = (
        feature_df['log_en_page_length'] * 0.4 +
        feature_df['log_num_categories'] * 0.3 +
        feature_df['log_num_external_links'] * 0.3
    )
    
    # Binary threshold features
    feature_df['is_highly_global'] = (feature_df['num_languages'] > 20).astype(int)
    feature_df['is_niche'] = (feature_df['num_languages'] < 10).astype(int)
    feature_df['has_long_page'] = (feature_df['en_page_length'] > 10000).astype(int)
    feature_df['has_many_statements'] = (feature_df['num_statements'] > 30).astype(int)
    
    # Polynomial features
    feature_df['num_languages_squared'] = feature_df['log_num_languages'] ** 2
    feature_df['cultural_specificity_squared'] = feature_df['cultural_specificity_score'] ** 2
    
    # Select feature columns
    feature_cols = [
        'log_num_languages', 'log_en_page_length', 'log_num_categories',
        'log_num_external_links', 'log_num_statements', 'log_num_identifiers',
        'log_statement_diversity', 'num_cultural_properties', 'num_geographic_properties',
        'has_coordinates', 'has_country', 'has_culture_property', 'has_origin_country',
        'cultural_ratio', 'geographic_ratio', 'identifier_ratio',
        'categories_per_page', 'external_links_per_page',
        'languages_x_statements', 'languages_x_page_length',
        'has_country_x_languages', 'has_country_x_statements', 'cultural_x_geographic',
        'global_reach_score', 'cultural_specificity_score',
        'info_richness_score', 'page_quality_score',
        'is_highly_global', 'is_niche', 'has_long_page', 'has_many_statements',
        'num_languages_squared', 'cultural_specificity_squared'
    ]
    
    X = feature_df[feature_cols].fillna(0)
    return X, feature_cols


def train_best_model(X_train, y_train, X_val, y_val, best_params, feature_names, use_gpu=True):
    """Train final model with best hyperparameters"""
    logger.info("\n" + "="*80)
    logger.info(" Training Final Model with Best Hyperparameters")
    logger.info("="*80)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Prepare params
    params = {
        'device': 'cuda' if use_gpu else 'cpu',
        'tree_method': 'hist',
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'random_state': 42,
        **best_params
    }
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train_encoded, feature_names=feature_names)
    dval = xgb.DMatrix(X_val_scaled, label=y_val_encoded, feature_names=feature_names)
    
    # Train model with progress bar
    logger.info("Training in progress...")
    pbar = tqdm(total=1000, desc="Training", unit="rounds")
    
    def callback(env):
        pbar.update(1)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    pbar.close()
    
    logger.info(f" Training completed!")
    logger.info(f" Best iteration: {model.best_iteration}")
    logger.info(f" Best score: {model.best_score:.4f}")
    
    # Evaluate
    from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
    
    # Training set
    y_train_pred_proba = model.predict(dtrain)
    y_train_pred = np.argmax(y_train_pred_proba, axis=1)
    train_f1 = f1_score(y_train_encoded, y_train_pred, average='macro')
    train_precision = precision_score(y_train_encoded, y_train_pred, average='macro')
    train_recall = recall_score(y_train_encoded, y_train_pred, average='macro')
    
    # Validation set
    y_val_pred_proba = model.predict(dval)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    val_f1 = f1_score(y_val_encoded, y_val_pred, average='macro')
    val_precision = precision_score(y_val_encoded, y_val_pred, average='macro')
    val_recall = recall_score(y_val_encoded, y_val_pred, average='macro')
    
    logger.info("\n" + "="*80)
    logger.info(" Final Model Performance")
    logger.info("="*80)
    logger.info("\nTraining Set:")
    logger.info(f"  F1 Score (Macro):    {train_f1:.4f}")
    logger.info(f"  Precision (Macro):   {train_precision:.4f}")
    logger.info(f"  Recall (Macro):      {train_recall:.4f}")
    logger.info("\nValidation Set:")
    logger.info(f"  F1 Score (Macro):    {val_f1:.4f}")
    logger.info(f"  Precision (Macro):   {val_precision:.4f}")
    logger.info(f"  Recall (Macro):      {val_recall:.4f}")
    
    # Per-class metrics
    y_val_true_labels = label_encoder.inverse_transform(y_val_encoded)
    y_val_pred_labels = label_encoder.inverse_transform(y_val_pred)
    report = classification_report(y_val_true_labels, y_val_pred_labels, output_dict=True)
    
    logger.info("\n Per-Class Metrics (Validation):")
    for label in sorted(report.keys()):
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics = report[label]
            logger.info(f"\n  {label.upper()}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall:    {metrics['recall']:.4f}")
            logger.info(f"    F1-Score:  {metrics['f1-score']:.4f}")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'use_gpu': use_gpu,
        'best_params': best_params
    }
    
    with open('cultural_classifier_tuned.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info("\n Saved tuned model to cultural_classifier_tuned.pkl")
    
    return model, scaler, label_encoder


def main():
    """Main tuning pipeline"""
    logger.info("="*80)
    logger.info(" Hyperparameter Tuning Pipeline")
    logger.info("="*80)
    
    # Load data
    logger.info("\n Loading training data...")
    df_train = pd.read_csv('train_enriched.csv')
    logger.info(f" Loaded {len(df_train)} samples")
    
    # Split for tuning (use smaller validation set for faster tuning)
    df_train, df_val = train_test_split(
        df_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_train['label']
    )
    logger.info(f" Split: train={len(df_train)}, validation={len(df_val)}")
    
    # Engineer features
    logger.info("\n Engineering features...")
    X_train, feature_names = engineer_features(df_train)
    X_val, _ = engineer_features(df_val)
    logger.info(f" Created {len(feature_names)} features")
    
    y_train = df_train['label']
    y_val = df_val['label']
    
    # Initialize tuner
    tuner = CulturalClassifierTuner(X_train, y_train, X_val, y_val, use_gpu=True)
    
    # Run tuning (adjust n_trials based on time available)
    # 50 trials ~ 5-10 minutes on GPU
    # 100 trials ~ 10-20 minutes on GPU
    study = tuner.tune(n_trials=50)
    
    # Save tuning results
    import json
    tuning_results = {
        'best_f1_score': tuner.best_score,
        'best_params': tuner.best_params,
        'n_trials': len(study.trials),
        'all_trials': [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params
            }
            for trial in study.trials
        ]
    }
    
    with open('tuning_results.json', 'w') as f:
        json.dump(tuning_results, f, indent=2)
    logger.info("\n Saved tuning_results.json")
    
    # Train final model with best hyperparameters
    final_model, scaler, label_encoder = train_best_model(
        X_train, y_train, X_val, y_val,
        tuner.best_params, feature_names, use_gpu=True
    )
    
    logger.info("\n" + "="*80)
    logger.info(" Hyperparameter Tuning Complete!")
    logger.info(" Output files:")
    logger.info("   • cultural_classifier_tuned.pkl (tuned model)")
    logger.info("   • tuning_results.json (all trial results)")
    logger.info("   • hyperparameter_tuning.log (detailed logs)")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()