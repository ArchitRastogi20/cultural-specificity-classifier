"""
Testing Script - Compare Performance WITH and WITHOUT Wikipedia Features
Tests on valid.csv using:
1. Only basic features (name, description, type, category, subcategory)
2. With Wikipedia/Wikidata features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
import logging
import pickle
from tqdm import tqdm
import requests
import re
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comparison_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WikiDataExtractor:
    """Extract features from Wikipedia and Wikidata"""
    
    def __init__(self, max_retries=3, timeout=10):
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CulturalClassifier/1.0 (Educational Project)'
        })
        
    def extract_qid_from_url(self, url: str) -> Optional[str]:
        try:
            match = re.search(r'Q\d+', url)
            return match.group(0) if match else None
        except:
            return None
    
    def get_wikipedia_stats(self, qid: str) -> Dict:
        stats = {
            'num_languages': 0,
            'en_page_length': 0,
            'num_interwiki_links': 0,
            'num_categories': 0,
            'has_coordinates': False,
            'num_external_links': 0
        }
        
        try:
            url = f"https://www.wikidata.org/wiki/Special/EntityData/{qid}.json"
            response = self._make_request(url)
            if response:
                data = response.json()
                entity_data = data.get('entities', {}).get(qid, {})
                
                sitelinks = entity_data.get('sitelinks', {})
                stats['num_languages'] = len([s for s in sitelinks.keys() if s.endswith('wiki')])
                
                en_wiki = sitelinks.get('enwiki', {})
                en_title = en_wiki.get('title', '')
                
                claims = entity_data.get('claims', {})
                stats['has_coordinates'] = 'P625' in claims
                
                if en_title:
                    wiki_stats = self.get_en_wikipedia_page_stats(en_title)
                    stats.update(wiki_stats)
        except:
            pass
            
        return stats
    
    def get_en_wikipedia_page_stats(self, title: str) -> Dict:
        stats = {
            'en_page_length': 0,
            'num_interwiki_links': 0,
            'num_categories': 0,
            'num_external_links': 0
        }
        
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'titles': title,
                'prop': 'info|categories|extlinks',
                'inprop': 'length',
                'cllimit': 'max',
                'ellimit': 'max',
                'format': 'json'
            }
            
            response = self._make_request(url, params=params)
            if response:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})
                
                for page_id, page_data in pages.items():
                    if page_id != '-1':
                        stats['en_page_length'] = page_data.get('length', 0)
                        stats['num_categories'] = len(page_data.get('categories', []))
                        stats['num_external_links'] = len(page_data.get('extlinks', []))
        except:
            pass
            
        return stats
    
    def get_wikidata_features(self, qid: str) -> Dict:
        features = {
            'num_statements': 0,
            'num_cultural_properties': 0,
            'num_geographic_properties': 0,
            'has_country': False,
            'has_culture_property': False,
            'has_origin_country': False,
            'num_identifiers': 0,
            'statement_diversity': 0.0
        }
        
        cultural_properties = {
            'P17', 'P495', 'P2596', 'P2012', 'P1532', 'P27', 'P1376', 'P361', 'P279', 'P31',
        }
        
        geographic_properties = {
            'P17', 'P495', 'P131', 'P276', 'P27', 'P1376', 'P159', 'P625'
        }
        
        try:
            url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
            response = self._make_request(url)
            
            if response:
                data = response.json()
                entity_data = data.get('entities', {}).get(qid, {})
                claims = entity_data.get('claims', {})
                
                features['num_statements'] = sum(len(v) for v in claims.values())
                unique_properties = set(claims.keys())
                features['statement_diversity'] = len(unique_properties)
                features['num_cultural_properties'] = len(unique_properties & cultural_properties)
                features['num_geographic_properties'] = len(unique_properties & geographic_properties)
                features['has_country'] = 'P17' in claims
                features['has_culture_property'] = 'P2596' in claims
                features['has_origin_country'] = 'P495' in claims
                
                features['num_identifiers'] = len([
                    p for p in unique_properties if p.startswith('P') and 
                    any(c.get('mainsnak', {}).get('datatype') == 'external-id' 
                        for c in claims.get(p, []))
                ])
        except:
            pass
            
        return features
    
    def _make_request(self, url: str, params: Optional[Dict] = None):
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response
            except:
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(0.5 * (attempt + 1))
        return None
    
    def extract_all_features(self, item_url: str) -> Dict:
        qid = self.extract_qid_from_url(item_url)
        if not qid:
            return self._empty_features()
        wiki_stats = self.get_wikipedia_stats(qid)
        wikidata_features = self.get_wikidata_features(qid)
        return {**wiki_stats, **wikidata_features}
    
    def _empty_features(self) -> Dict:
        return {
            'num_languages': 0,
            'en_page_length': 0,
            'num_interwiki_links': 0,
            'num_categories': 0,
            'has_coordinates': False,
            'num_external_links': 0,
            'num_statements': 0,
            'num_cultural_properties': 0,
            'num_geographic_properties': 0,
            'has_country': False,
            'has_culture_property': False,
            'has_origin_country': False,
            'num_identifiers': 0,
            'statement_diversity': 0.0
        }


def process_single_item(args):
    idx, item_url, extractor = args
    try:
        features = extractor.extract_all_features(item_url)
        return idx, features
    except:
        return idx, extractor._empty_features()


def extract_features_parallel(df: pd.DataFrame, num_workers: int = 32) -> pd.DataFrame:
    logger.info(f" Extracting Wikipedia/Wikidata features...")
    
    extractor = WikiDataExtractor()
    args_list = [(idx, row['item'], extractor) for idx, row in df.iterrows()]
    results = {}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_item, args): args[0. for args in args_list}
        
        with tqdm(total=len(futures), desc="Extracting features", unit="items") as pbar:
            for future in as_completed(futures):
                idx, features = future.result()
                results[idx] = features
                pbar.update(1)
    
    features_df = pd.DataFrame.from_dict(results, orient='index').sort_index()
    result_df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    
    return result_df


def engineer_wiki_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features FROM Wikipedia data"""
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
    
    return feature_df


def engineer_basic_features(df: pd.DataFrame):
    """Engineer features from ONLY basic columns (no Wikipedia)"""
    logger.info(" Engineering features from basic columns only...")
    
    # Combine text fields
    df['text'] = df['name'].fillna('') + ' ' + df['description'].fillna('')
    
    # TF-IDF on text (name + description)
    tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    tfidf_features = tfidf.fit_transform(df['text']).toarray()
    tfidf_df = pd.DataFrame(
        tfidf_features,
        columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    )
    
    # Text length features
    df['name_length'] = df['name'].fillna('').str.len()
    df['desc_length'] = df['description'].fillna('').str.len()
    df['name_word_count'] = df['name'].fillna('').str.split().str.len()
    df['desc_word_count'] = df['description'].fillna('').str.split().str.len()
    
    # One-hot encode categorical features
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    category_dummies = pd.get_dummies(df['category'], prefix='category')
    
    # Combine all basic features
    basic_features = pd.concat([
        df[['name_length', 'desc_length', 'name_word_count', 'desc_word_count']],
        tfidf_df,
        type_dummies,
        category_dummies
    ], axis=1)
    
    logger.info(f" Created {len(basic_features.columns)} basic features")
    
    return basic_features, tfidf


def evaluate_model(y_true, y_pred, model_name):
    """Evaluate and log performance metrics"""
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    accuracy = (y_true == y_pred).mean()
    
    logger.info(f"\n{'='*80}")
    logger.info(f" {model_name} RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f" Accuracy:            {accuracy:.4f}")
    logger.info(f" F1 Score (Macro):    {f1_macro:.4f}")
    logger.info(f" F1 Score (Weighted): {f1_weighted:.4f}")
    logger.info(f" Precision (Macro):   {precision_macro:.4f}")
    logger.info(f" Recall (Macro):      {recall_macro:.4f}")
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    logger.info("\n Per-Class Metrics:")
    for label in sorted(report.keys()):
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics = report[label]
            logger.info(f"\n  {label.upper()}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall:    {metrics['recall']:.4f}")
            logger.info(f"    F1-Score:  {metrics['f1-score']:.4f}")
            logger.info(f"    Support:   {int(metrics['support'])}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'per_class': report
    }


def main():
    """Main comparison pipeline"""
    logger.info("="*80)
    logger.info(" COMPARISON: Basic Features vs Wikipedia Features")
    logger.info("="*80)
    
    # Load test data
    logger.info("\n Loading test dataset...")
    df_test = pd.read_csv('valid.csv')
    logger.info(f" Loaded {len(df_test)} test samples")
    
    y_true = df_test['label']
    logger.info(f"\n Label Distribution:")
    for label, count in y_true.value_counts().items():
        logger.info(f"  {label}: {count} ({count/len(y_true)*100:.1f}%)")
    
    # ============================================================
    # TEST 1: WITHOUT Wikipedia Features (Basic columns only)
    # ============================================================
    logger.info("\n" + "="*80)
    logger.info(" TEST 1: Using ONLY Basic Columns")
    logger.info("   (item, name, description, type, category, subcategory)")
    logger.info("="*80)
    
    # Train a simple baseline model on basic features
    logger.info("\n Training baseline model on basic features...")
    X_basic, tfidf = engineer_basic_features(df_test)
    
    # Use a simple XGBoost model
    label_encoder_basic = LabelEncoder()
    y_encoded = label_encoder_basic.fit_transform(y_true)
    
    # Quick train/test split for baseline
    from sklearn.model_selection import train_test_split
    X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(
        X_basic, y_encoded, test_size=0.5, random_state=42, stratify=y_encoded
    )
    
    dtrain_basic = xgb.DMatrix(X_train_basic, label=y_train_basic)
    dtest_basic = xgb.DMatrix(X_test_basic, label=y_test_basic)
    
    params_basic = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 4,
        'learning_rate': 0.05,
        'device': 'cuda',
        'verbosity': 0
    }
    
    logger.info("Training baseline model...")
    model_basic = xgb.train(
        params_basic,
        dtrain_basic,
        num_boost_round=200,
        evals=[(dtest_basic, 'test')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    # Predict
    y_pred_basic_encoded = np.argmax(model_basic.predict(dtest_basic), axis=1)
    y_pred_basic = label_encoder_basic.inverse_transform(y_pred_basic_encoded)
    y_test_basic_labels = label_encoder_basic.inverse_transform(y_test_basic)
    
    # Evaluate
    results_basic = evaluate_model(y_test_basic_labels, y_pred_basic, "BASELINE (Basic Features Only)")
    
    # ============================================================
    # TEST 2: WITH Wikipedia Features
    # ============================================================
    logger.info("\n\n" + "="*80)
    logger.info(" TEST 2: Using Wikipedia/Wikidata Features")
    logger.info("="*80)
    
    # Extract Wikipedia features
    start_time = time.time()
    df_test_enriched = extract_features_parallel(df_test, num_workers=32)
    extraction_time = time.time() - start_time
    logger.info(f" Feature extraction completed in {extraction_time:.2f}s")
    
    # Engineer features
    df_test_featured = engineer_wiki_features(df_test_enriched)
    
    # Load tuned model
    logger.info("\n Loading tuned model...")
    with open('cultural_classifier_tuned.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    
    # Prepare features
    X_test_wiki = df_test_featured[feature_names].fillna(0)
    X_test_wiki_scaled = scaler.transform(X_test_wiki)
    
    # Predict
    dtest_wiki = xgb.DMatrix(X_test_wiki_scaled, feature_names=feature_names)
    y_pred_wiki_proba = model.predict(dtest_wiki)
    y_pred_wiki_encoded = np.argmax(y_pred_wiki_proba, axis=1)
    y_pred_wiki = label_encoder.inverse_transform(y_pred_wiki_encoded)
    
    # Evaluate
    results_wiki = evaluate_model(y_true, y_pred_wiki, "WITH WIKIPEDIA FEATURES")
    
    # ============================================================
    # COMPARISON SUMMARY
    # ============================================================
    logger.info("\n\n" + "="*80)
    logger.info(" FINAL COMPARISON SUMMARY")
    logger.info("="*80)
    
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 (Macro)', 'F1 (Weighted)', 'Precision (Macro)', 'Recall (Macro)'],
        'Basic Features Only': [
            f"{results_basic['accuracy']:.4f}",
            f"{results_basic['f1_macro']:.4f}",
            f"{results_basic['f1_weighted']:.4f}",
            f"{results_basic['precision_macro']:.4f}",
            f"{results_basic['recall_macro']:.4f}"
        ],
        'With Wikipedia Features': [
            f"{results_wiki['accuracy']:.4f}",
            f"{results_wiki['f1_macro']:.4f}",
            f"{results_wiki['f1_weighted']:.4f}",
            f"{results_wiki['precision_macro']:.4f}",
            f"{results_wiki['recall_macro']:.4f}"
        ],
        'Improvement': [
            f"{(results_wiki['accuracy'] - results_basic['accuracy']):.4f}",
            f"{(results_wiki['f1_macro'] - results_basic['f1_macro']):.4f}",
            f"{(results_wiki['f1_weighted'] - results_basic['f1_weighted']):.4f}",
            f"{(results_wiki['precision_macro'] - results_basic['precision_macro']):.4f}",
            f"{(results_wiki['recall_macro'] - results_basic['recall_macro']):.4f}"
        ]
    })
    
    logger.info(f"\n{comparison.to_string(index=False)}")
    
    # Save comparison
    comparison.to_csv('feature_comparison.csv', index=False)
    logger.info("\n Saved feature_comparison.csv")
    
    logger.info("\n" + "="*80)
    logger.info(" Comparison Complete!")
    logger.info(f" Wikipedia features improved F1 (Macro) by: {(results_wiki['f1_macro'] - results_basic['f1_macro']):.4f}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()