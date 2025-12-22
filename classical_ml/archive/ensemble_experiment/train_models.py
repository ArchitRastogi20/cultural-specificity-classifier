"""
Cultural Classification Model Training
Trains 4 models:
1. XGBoost without feature engineering
2. XGBoost with feature engineering
3. Ensemble (XGB+CatBoost+LightGBM) without feature engineering
4. Ensemble (XGB+CatBoost+LightGBM) with feature engineering

Hardware: RTX 3080, 8 vCPU, 30GB RAM
"""

import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Limit CPU usage
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

# Create results directory
os.makedirs('results', exist_ok=True)

# Setup logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'results/training_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_basic_features(df):
    """Create basic numerical features from available columns"""
    logger.info("Creating basic features from available data...")
    
    features = pd.DataFrame()
    
    # Text length features
    features['name_length'] = df['name'].fillna('').astype(str).str.len()
    features['desc_length'] = df['description'].fillna('').astype(str).str.len()
    features['name_word_count'] = df['name'].fillna('').astype(str).str.split().str.len()
    features['desc_word_count'] = df['description'].fillna('').astype(str).str.split().str.len()
    
    # Categorical encoding
    features['type_entity'] = (df['type'] == 'entity').astype(int)
    features['type_concept'] = (df['type'] == 'concept').astype(int)
    
    # Category features (one-hot top categories)
    top_categories = ['music', 'films', 'sports', 'literature', 'visual arts', 
                      'architecture', 'media', 'history', 'politics', 'food']
    for cat in top_categories:
        features[f'cat_{cat.replace(" ", "_")}'] = (df['category'] == cat).astype(int)
    
    # Default values for enriched features (will be 0 if not available)
    enriched_cols = ['num_languages', 'en_page_length', 'num_statements', 'num_categories',
                     'num_external_links', 'num_identifiers', 'statement_diversity',
                     'num_cultural_properties', 'num_geographic_properties',
                     'has_coordinates', 'has_country', 'has_culture_property', 'has_origin_country']
    
    for col in enriched_cols:
        if col in df.columns:
            features[col] = df[col].fillna(0)
        else:
            # Default values for missing enriched features
            if col.startswith('has_'):
                features[col] = 0  # Boolean features default to False
            elif col.startswith('num_'):
                features[col] = 1  # Count features default to 1 to avoid log(0)
            else:
                features[col] = 0
    
    # Fill NaN with 0
    features = features.fillna(0)
    
    logger.info(f"Created {len(features.columns)} basic features")
    return features


def engineer_features(features_df):
    """Advanced feature engineering"""
    logger.info("Applying feature engineering...")
    
    df = features_df.copy()
    
    # Log transformations
    df['log_num_languages'] = np.log1p(df['num_languages'])
    df['log_en_page_length'] = np.log1p(df['en_page_length'])
    df['log_num_statements'] = np.log1p(df['num_statements'])
    df['log_num_categories'] = np.log1p(df['num_categories'])
    df['log_num_external_links'] = np.log1p(df['num_external_links'])
    df['log_num_identifiers'] = np.log1p(df['num_identifiers'])
    df['log_statement_diversity'] = np.log1p(df['statement_diversity'])
    
    # Ratio features
    df['cultural_ratio'] = df['num_cultural_properties'] / (df['statement_diversity'] + 1)
    df['geographic_ratio'] = df['num_geographic_properties'] / (df['statement_diversity'] + 1)
    df['identifier_ratio'] = df['num_identifiers'] / (df['num_statements'] + 1)
    df['categories_per_page'] = df['num_categories'] / (df['en_page_length'] + 1)
    df['external_links_per_page'] = df['num_external_links'] / (df['en_page_length'] + 1)
    
    # Advanced ratios
    df['statements_per_language'] = df['num_statements'] / (df['num_languages'] + 1)
    df['identifiers_per_language'] = df['num_identifiers'] / (df['num_languages'] + 1)
    df['categories_per_language'] = df['num_categories'] / (df['num_languages'] + 1)
    df['cultural_density'] = df['num_cultural_properties'] / (df['num_statements'] + 1)
    df['geographic_density'] = df['num_geographic_properties'] / (df['num_statements'] + 1)
    
    # Interaction features
    df['languages_x_statements'] = df['log_num_languages'] * df['log_num_statements']
    df['languages_x_page_length'] = df['log_num_languages'] * df['log_en_page_length']
    df['has_country_x_languages'] = df['has_country'] * df['log_num_languages']
    df['has_country_x_statements'] = df['has_country'] * df['log_num_statements']
    df['cultural_x_geographic'] = df['num_cultural_properties'] * df['num_geographic_properties']
    
    # Three-way interactions
    df['lang_x_stmt_x_country'] = (
        df['log_num_languages'] * 
        df['log_num_statements'] * 
        df['has_country']
    )
    df['cultural_x_geo_x_lang'] = (
        df['num_cultural_properties'] * 
        df['num_geographic_properties'] * 
        df['log_num_languages']
    )
    
    # Composite scores
    df['global_reach_score'] = (
        df['log_num_languages'] * 0.5 +
        df['log_en_page_length'] * 0.3 +
        df['log_num_external_links'] * 0.2
    )
    
    df['cultural_specificity_score'] = (
        df['num_cultural_properties'] * 2.0 +
        df['num_geographic_properties'] * 1.5 +
        df['has_country'] * 1.0 +
        df['has_origin_country'] * 1.0 +
        df['has_culture_property'] * 2.0
    )
    
    df['info_richness_score'] = (
        df['log_num_statements'] * 0.4 +
        df['log_statement_diversity'] * 0.4 +
        df['log_num_identifiers'] * 0.2
    )
    
    df['page_quality_score'] = (
        df['log_en_page_length'] * 0.4 +
        df['log_num_categories'] * 0.3 +
        df['log_num_external_links'] * 0.3
    )
    
    # Exclusivity indicators
    df['exclusivity_score'] = (
        (df['num_languages'] < 15).astype(int) * 2 +
        df['has_country'] * 1.5 +
        df['has_origin_country'] * 1.5 +
        (df['num_cultural_properties'] > 2).astype(int) * 1
    )
    
    df['representativeness_score'] = (
        (df['num_languages'] >= 15).astype(int) * 
        (df['num_languages'] < 50).astype(int) * 2 +
        (df['num_cultural_properties'] > 0).astype(int) * 1.5
    )
    
    df['agnostic_score'] = (
        (df['num_languages'] > 50).astype(int) * 2 +
        (df['has_country'] == 0).astype(int) * 1 +
        (df['num_cultural_properties'] == 0).astype(int) * 1
    )
    
    # Binary thresholds
    df['is_highly_global'] = (df['num_languages'] > 20).astype(int)
    df['is_niche'] = (df['num_languages'] < 10).astype(int)
    df['has_long_page'] = (df['en_page_length'] > 10000).astype(int)
    df['has_many_statements'] = (df['num_statements'] > 30).astype(int)
    df['is_very_global'] = (df['num_languages'] > 50).astype(int)
    df['is_moderately_global'] = ((df['num_languages'] >= 15) & (df['num_languages'] <= 50)).astype(int)
    
    # Polynomial features
    df['num_languages_squared'] = df['log_num_languages'] ** 2
    df['num_languages_cubed'] = df['log_num_languages'] ** 3
    df['cultural_specificity_squared'] = df['cultural_specificity_score'] ** 2
    df['global_reach_squared'] = df['global_reach_score'] ** 2
    
    # Statistical bins
    df['lang_bin_low'] = (df['num_languages'] < 10).astype(int)
    df['lang_bin_mid'] = ((df['num_languages'] >= 10) & (df['num_languages'] < 30)).astype(int)
    df['lang_bin_high'] = (df['num_languages'] >= 30).astype(int)
    
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    logger.info(f"Engineered features: {len(df.columns)} total features")
    return df


def evaluate_model(y_true, y_pred, model_name):
    """Evaluate and log model performance"""
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    logger.info(f"\n{model_name} Performance:")
    logger.info(f"  Accuracy:         {accuracy:.4f}")
    logger.info(f"  F1 (Macro):       {f1_macro:.4f}")
    logger.info(f"  F1 (Weighted):    {f1_weighted:.4f}")
    logger.info(f"  Precision:        {precision:.4f}")
    logger.info(f"  Recall:           {recall:.4f}")
    
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'classification_report': report
    }


def train_xgboost_simple(X_train, y_train, X_val, y_val, label_encoder):
    """Train XGBoost without feature engineering"""
    logger.info("\n" + "="*80)
    logger.info("Training XGBoost (No Feature Engineering)")
    logger.info("="*80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train_encoded, feature_names=[f'f{i}' for i in range(X_train.shape[1])])
    dval = xgb.DMatrix(X_val_scaled, label=y_val_encoded, feature_names=[f'f{i}' for i in range(X_val.shape[1])])
    
    params = {
        'objective': 'multi:softprob',
        'num_class': len(label_encoder.classes_),
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Predictions
    train_pred_proba = model.predict(dtrain)
    val_pred_proba = model.predict(dval)
    
    train_pred = label_encoder.inverse_transform(np.argmax(train_pred_proba, axis=1))
    val_pred = label_encoder.inverse_transform(np.argmax(val_pred_proba, axis=1))
    
    # Evaluate
    train_metrics = evaluate_model(y_train, train_pred, "XGBoost-Simple Train")
    val_metrics = evaluate_model(y_val, val_pred, "XGBoost-Simple Val")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': [f'f{i}' for i in range(X_train.shape[1])]
    }
    
    with open('results/xgboost_simple.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info("Model saved: results/xgboost_simple.pkl")
    
    return val_metrics


def train_xgboost_engineered(X_train, y_train, X_val, y_val, label_encoder):
    """Train XGBoost with feature engineering"""
    logger.info("\n" + "="*80)
    logger.info("Training XGBoost (With Feature Engineering)")
    logger.info("="*80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    feature_names = [f'f{i}' for i in range(X_train.shape[1])]
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train_encoded, feature_names=feature_names)
    dval = xgb.DMatrix(X_val_scaled, label=y_val_encoded, feature_names=feature_names)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': len(label_encoder.classes_),
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    # Predictions
    train_pred_proba = model.predict(dtrain)
    val_pred_proba = model.predict(dval)
    
    train_pred = label_encoder.inverse_transform(np.argmax(train_pred_proba, axis=1))
    val_pred = label_encoder.inverse_transform(np.argmax(val_pred_proba, axis=1))
    
    # Evaluate
    train_metrics = evaluate_model(y_train, train_pred, "XGBoost-Engineered Train")
    val_metrics = evaluate_model(y_val, val_pred, "XGBoost-Engineered Val")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names
    }
    
    with open('results/xgboost_engineered.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info("Model saved: results/xgboost_engineered.pkl")
    
    return val_metrics


def train_ensemble_simple(X_train, y_train, X_val, y_val, label_encoder):
    """Train Ensemble (XGB+Cat+LGB) without feature engineering"""
    logger.info("\n" + "="*80)
    logger.info("Training Ensemble (No Feature Engineering)")
    logger.info("="*80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # XGBoost
    logger.info("\n1. Training XGBoost...")
    feature_names = [f'f{i}' for i in range(X_train.shape[1])]
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train_encoded, feature_names=feature_names)
    dval = xgb.DMatrix(X_val_scaled, label=y_val_encoded, feature_names=feature_names)
    
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': len(label_encoder.classes_),
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    xgb_val_pred = np.argmax(xgb_model.predict(dval), axis=1)
    xgb_f1 = f1_score(y_val_encoded, xgb_val_pred, average='macro')
    logger.info(f"  XGBoost F1: {xgb_f1:.4f}")
    
    # LightGBM
    logger.info("\n2. Training LightGBM...")
    lgb_train = lgb.Dataset(X_train_scaled, label=y_train_encoded)
    lgb_val = lgb.Dataset(X_val_scaled, label=y_val_encoded, reference=lgb_train)
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': len(label_encoder.classes_),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_threads': 8,
        'verbose': -1,
        'seed': 42
    }
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    lgb_val_pred = np.argmax(lgb_model.predict(X_val_scaled), axis=1)
    lgb_f1 = f1_score(y_val_encoded, lgb_val_pred, average='macro')
    logger.info(f"  LightGBM F1: {lgb_f1:.4f}")
    
    # CatBoost
    logger.info("\n3. Training CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        task_type='GPU',
        devices='0',
        verbose=False,
        random_seed=42,
        early_stopping_rounds=50
    )
    
    cat_model.fit(X_train_scaled, y_train_encoded, eval_set=(X_val_scaled, y_val_encoded))
    
    cat_val_pred = cat_model.predict(X_val_scaled).flatten()
    cat_f1 = f1_score(y_val_encoded, cat_val_pred, average='macro')
    logger.info(f"  CatBoost F1: {cat_f1:.4f}")
    
    # Weighted Ensemble
    logger.info("\n4. Creating Weighted Ensemble...")
    total_f1 = xgb_f1 + lgb_f1 + cat_f1
    w_xgb = xgb_f1 / total_f1
    w_lgb = lgb_f1 / total_f1
    w_cat = cat_f1 / total_f1
    
    xgb_val_proba = xgb_model.predict(dval)
    lgb_val_proba = lgb_model.predict(X_val_scaled)
    cat_val_proba = cat_model.predict_proba(X_val_scaled)
    
    ensemble_proba = w_xgb * xgb_val_proba + w_lgb * lgb_val_proba + w_cat * cat_val_proba
    ensemble_pred_encoded = np.argmax(ensemble_proba, axis=1)
    ensemble_pred = label_encoder.inverse_transform(ensemble_pred_encoded)
    
    ensemble_f1 = f1_score(y_val_encoded, ensemble_pred_encoded, average='macro')
    logger.info(f"\n  Ensemble F1: {ensemble_f1:.4f}")
    logger.info(f"  Weights: XGB={w_xgb:.3f}, LGB={w_lgb:.3f}, CAT={w_cat:.3f}")
    
    # Evaluate
    val_metrics = evaluate_model(y_val, ensemble_pred, "Ensemble-Simple Val")
    
    # Save model
    model_data = {
        'xgb_model': xgb_model,
        'lgb_model': lgb_model,
        'catboost_model': cat_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'ensemble_weights': {'xgb': w_xgb, 'lgb': w_lgb, 'catboost': w_cat}
    }
    
    with open('results/ensemble_simple.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info("Model saved: results/ensemble_simple.pkl")
    
    return val_metrics


def train_ensemble_engineered(X_train, y_train, X_val, y_val, label_encoder):
    """Train Ensemble (XGB+Cat+LGB) with feature engineering"""
    logger.info("\n" + "="*80)
    logger.info("Training Ensemble (With Feature Engineering)")
    logger.info("="*80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # XGBoost
    logger.info("\n1. Training XGBoost...")
    feature_names = [f'f{i}' for i in range(X_train.shape[1])]
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train_encoded, feature_names=feature_names)
    dval = xgb.DMatrix(X_val_scaled, label=y_val_encoded, feature_names=feature_names)
    
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': len(label_encoder.classes_),
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'eval_metric': 'mlogloss',
        'seed': 42
    }
    
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    xgb_val_pred = np.argmax(xgb_model.predict(dval), axis=1)
    xgb_f1 = f1_score(y_val_encoded, xgb_val_pred, average='macro')
    logger.info(f"  XGBoost F1: {xgb_f1:.4f}")
    
    # LightGBM
    logger.info("\n2. Training LightGBM...")
    lgb_train = lgb.Dataset(X_train_scaled, label=y_train_encoded)
    lgb_val = lgb.Dataset(X_val_scaled, label=y_val_encoded, reference=lgb_train)
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': len(label_encoder.classes_),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'num_threads': 8,
        'verbose': -1,
        'seed': 42
    }
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    lgb_val_pred = np.argmax(lgb_model.predict(X_val_scaled), axis=1)
    lgb_f1 = f1_score(y_val_encoded, lgb_val_pred, average='macro')
    logger.info(f"  LightGBM F1: {lgb_f1:.4f}")
    
    # CatBoost
    logger.info("\n3. Training CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        loss_function='MultiClass',
        task_type='GPU',
        devices='0',
        l2_leaf_reg=3,
        verbose=False,
        random_seed=42,
        early_stopping_rounds=100
    )
    
    cat_model.fit(X_train_scaled, y_train_encoded, eval_set=(X_val_scaled, y_val_encoded))
    
    cat_val_pred = cat_model.predict(X_val_scaled).flatten()
    cat_f1 = f1_score(y_val_encoded, cat_val_pred, average='macro')
    logger.info(f"  CatBoost F1: {cat_f1:.4f}")
    
    # Weighted Ensemble
    logger.info("\n4. Creating Weighted Ensemble...")
    total_f1 = xgb_f1 + lgb_f1 + cat_f1
    w_xgb = xgb_f1 / total_f1
    w_lgb = lgb_f1 / total_f1
    w_cat = cat_f1 / total_f1
    
    xgb_val_proba = xgb_model.predict(dval)
    lgb_val_proba = lgb_model.predict(X_val_scaled)
    cat_val_proba = cat_model.predict_proba(X_val_scaled)
    
    ensemble_proba = w_xgb * xgb_val_proba + w_lgb * lgb_val_proba + w_cat * cat_val_proba
    ensemble_pred_encoded = np.argmax(ensemble_proba, axis=1)
    ensemble_pred = label_encoder.inverse_transform(ensemble_pred_encoded)
    
    ensemble_f1 = f1_score(y_val_encoded, ensemble_pred_encoded, average='macro')
    logger.info(f"\n  Ensemble F1: {ensemble_f1:.4f}")
    logger.info(f"  Weights: XGB={w_xgb:.3f}, LGB={w_lgb:.3f}, CAT={w_cat:.3f}")
    
    # Evaluate
    val_metrics = evaluate_model(y_val, ensemble_pred, "Ensemble-Engineered Val")
    
    # Save model
    model_data = {
        'xgb_model': xgb_model,
        'lgb_model': lgb_model,
        'catboost_model': cat_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'ensemble_weights': {'xgb': w_xgb, 'lgb': w_lgb, 'catboost': w_cat}
    }
    
    with open('results/ensemble_engineered.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info("Model saved: results/ensemble_engineered.pkl")
    
    return val_metrics


def test_model(model_path, model_name, X_test, y_test, label_encoder):
    """Test a single model on test set"""
    if not os.path.exists(model_path):
        logger.warning(f"Model not found: {model_path}")
        return None
    
    logger.info(f"\nTesting {model_name}...")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    scaler = model_data['scaler']
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions based on model type
    if 'ensemble' in model_path:
        # Ensemble prediction
        xgb_model = model_data['xgb_model']
        lgb_model = model_data['lgb_model']
        cat_model = model_data['catboost_model']
        weights = model_data['ensemble_weights']
        
        feature_names = model_data['feature_names']
        dtest = xgb.DMatrix(X_test_scaled, feature_names=feature_names)
        
        xgb_proba = xgb_model.predict(dtest)
        lgb_proba = lgb_model.predict(X_test_scaled)
        cat_proba = cat_model.predict_proba(X_test_scaled)
        
        ensemble_proba = (
            weights['xgb'] * xgb_proba +
            weights['lgb'] * lgb_proba +
            weights['catboost'] * cat_proba
        )
        
        y_pred_encoded = np.argmax(ensemble_proba, axis=1)
    else:
        # XGBoost prediction
        feature_names = model_data['feature_names']
        dtest = xgb.DMatrix(X_test_scaled, feature_names=feature_names)
        proba = model_data['model'].predict(dtest)
        y_pred_encoded = np.argmax(proba, axis=1)
    
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred, f"{model_name} Test")
    return metrics


def main():
    """Main training pipeline"""
    logger.info("="*80)
    logger.info("Cultural Classification Training Pipeline")
    logger.info("="*80)
    logger.info(f"Timestamp: {timestamp}")
    
    # Load data
    logger.info("\nLoading data...")
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    logger.info(f"Loaded {len(df_train)} training samples")
    logger.info(f"Loaded {len(df_test)} test samples")
    
    # Split train into train/val
    df_train, df_val = train_test_split(
        df_train, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_train['label']
    )
    logger.info(f"Split: train={len(df_train)}, validation={len(df_val)}")
    
    # Label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(df_train['label'])
    logger.info(f"Classes: {label_encoder.classes_}")
    
    # Create basic features
    X_train_basic = create_basic_features(df_train)
    X_val_basic = create_basic_features(df_val)
    X_test_basic = create_basic_features(df_test)
    
    y_train = df_train['label']
    y_val = df_val['label']
    y_test = df_test['label']
    
    # Engineer features
    X_train_eng = engineer_features(X_train_basic)
    X_val_eng = engineer_features(X_val_basic)
    X_test_eng = engineer_features(X_test_basic)
    
    # Store results
    all_results = {}
    
    # Train models
    # 1. XGBoost Simple
    all_results['xgboost_simple'] = train_xgboost_simple(
        X_train_basic, y_train, X_val_basic, y_val, label_encoder
    )
    
    # 2. XGBoost Engineered
    all_results['xgboost_engineered'] = train_xgboost_engineered(
        X_train_eng, y_train, X_val_eng, y_val, label_encoder
    )
    
    # 3. Ensemble Simple
    all_results['ensemble_simple'] = train_ensemble_simple(
        X_train_basic, y_train, X_val_basic, y_val, label_encoder
    )
    
    # 4. Ensemble Engineered
    all_results['ensemble_engineered'] = train_ensemble_engineered(
        X_train_eng, y_train, X_val_eng, y_val, label_encoder
    )
    
    # Test on test set
    logger.info("\n" + "="*80)
    logger.info("TESTING ON TEST SET")
    logger.info("="*80)
    
    test_results = {}
    
    # Test each model with appropriate feature set
    test_results['xgboost_simple'] = test_model(
        'results/xgboost_simple.pkl', 'XGBoost-Simple', 
        X_test_basic, y_test, label_encoder
    )
    
    test_results['xgboost_engineered'] = test_model(
        'results/xgboost_engineered.pkl', 'XGBoost-Engineered',
        X_test_eng, y_test, label_encoder
    )
    
    test_results['ensemble_simple'] = test_model(
        'results/ensemble_simple.pkl', 'Ensemble-Simple',
        X_test_basic, y_test, label_encoder
    )
    
    test_results['ensemble_engineered'] = test_model(
        'results/ensemble_engineered.pkl', 'Ensemble-Engineered',
        X_test_eng, y_test, label_encoder
    )
    
    # Save summary
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*80)
    
    summary_data = []
    simple_results_data = []
    
    # Model descriptions for simple results
    model_descriptions = {
        'xgboost_simple': 'XGBoost with basic features (no feature engineering)',
        'xgboost_engineered': 'XGBoost with 73 engineered features',
        'ensemble_simple': 'Ensemble (XGB+LGB+CAT) with basic features',
        'ensemble_engineered': 'Ensemble (XGB+LGB+CAT) with 73 engineered features'
    }
    
    # Validation results
    for model_name, metrics in all_results.items():
        summary_data.append({
            'Model': model_name,
            'Split': 'Validation',
            'F1_Macro': metrics['f1_macro'],
            'F1_Weighted': metrics['f1_weighted'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall']
        })
    
    # Test results
    for model_name, metrics in test_results.items():
        if metrics is not None:
            summary_data.append({
                'Model': model_name,
                'Split': 'Test',
                'F1_Macro': metrics['f1_macro'],
                'F1_Weighted': metrics['f1_weighted'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            })
            
            # Simple results (test set only)
            simple_results_data.append({
                'Model': model_name,
                'F1_Score_Test': f"{metrics['f1_macro']:.4f}",
                'Notes': model_descriptions.get(model_name, '')
            })
    
    # Save full summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/summary.csv', index=False)
    
    # Save simple results
    simple_results_df = pd.DataFrame(simple_results_data)
    simple_results_df.to_csv('results/model_results.csv', index=False)
    
    logger.info("\n" + summary_df.to_string(index=False))
    logger.info("\n\nSimple Results (Test Set):")
    logger.info("\n" + simple_results_df.to_string(index=False))
    logger.info("\nResults saved to results/")
    logger.info("  - results/summary.csv (detailed metrics)")
    logger.info("  - results/model_results.csv (simple results)")
    logger.info("  - results/xgboost_simple.pkl")
    logger.info("  - results/xgboost_engineered.pkl")
    logger.info("  - results/ensemble_simple.pkl")
    logger.info("  - results/ensemble_engineered.pkl")
    logger.info("\nTraining complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()