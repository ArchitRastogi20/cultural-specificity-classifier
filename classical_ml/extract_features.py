"""
Wikipedia and Wikidata Feature Extractor for Cultural Classification
Extracts features from Wikipedia pages and Wikidata for each item in the dataset.
"""

import pandas as pd
import logging
import time
import requests
from typing import Dict, Optional, Tuple
import json
from collections import Counter
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_extraction.log'),
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
        """Extract Wikidata QID from URL"""
        try:
            # Extract QID from wikidata URL
            match = re.search(r'Q\d+', url)
            return match.group(0) if match else None
        except Exception as e:
            logger.error(f"Error extracting QID from {url}: {e}")
            return None
    
    def get_wikipedia_stats(self, qid: str) -> Dict:
        """
        Get Wikipedia statistics for an item:
        - Number of language editions
        - Page length (English)
        - Number of interwiki links
        - Categories
        """
        stats = {
            'num_languages': 0,
            'en_page_length': 0,
            'num_interwiki_links': 0,
            'num_categories': 0,
            'has_coordinates': False,
            'num_external_links': 0
        }
        
        try:
            # Get language editions from Wikidata
            url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
            response = self._make_request(url)
            if response:
                data = response.json()
                entity_data = data.get('entities', {}).get(qid, {})
                
                # Count language editions (sitelinks)
                sitelinks = entity_data.get('sitelinks', {})
                stats['num_languages'] = len([s for s in sitelinks.keys() if s.endswith('wiki')])
                
                # Get English Wikipedia title
                en_wiki = sitelinks.get('enwiki', {})
                en_title = en_wiki.get('title', '')
                
                # Check for coordinates
                claims = entity_data.get('claims', {})
                stats['has_coordinates'] = 'P625' in claims  # P625 is coordinate location
                
                # Get English Wikipedia page stats if available
                if en_title:
                    wiki_stats = self.get_en_wikipedia_page_stats(en_title)
                    stats.update(wiki_stats)
                    
        except Exception as e:
            logger.debug(f"Error getting Wikipedia stats for {qid}: {e}")
            
        return stats
    
    def get_en_wikipedia_page_stats(self, title: str) -> Dict:
        """Get English Wikipedia page statistics"""
        stats = {
            'en_page_length': 0,
            'num_interwiki_links': 0,
            'num_categories': 0,
            'num_external_links': 0
        }
        
        try:
            # Wikipedia API request
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
                    if page_id != '-1':  # Page exists
                        stats['en_page_length'] = page_data.get('length', 0)
                        stats['num_categories'] = len(page_data.get('categories', []))
                        stats['num_external_links'] = len(page_data.get('extlinks', []))
                        
        except Exception as e:
            logger.debug(f"Error getting English Wikipedia stats for {title}: {e}")
            
        return stats
    
    def get_wikidata_features(self, qid: str) -> Dict:
        """
        Get Wikidata features:
        - Number of statements
        - Number of cultural properties
        - Number of geographic properties
        - Specific cultural indicators
        """
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
        
        # Cultural property IDs in Wikidata
        cultural_properties = {
            'P17',   # country
            'P495',  # country of origin
            'P2596', # culture
            'P2012', # cuisine
            'P1532', # country for sport
            'P27',   # country of citizenship
            'P1376', # capital of
            'P361',  # part of (often cultural)
            'P279',  # subclass of
            'P31',   # instance of
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
                
                # Count statements
                features['num_statements'] = sum(len(v) for v in claims.values())
                
                # Count unique properties
                unique_properties = set(claims.keys())
                features['statement_diversity'] = len(unique_properties)
                
                # Count cultural properties
                features['num_cultural_properties'] = len(
                    unique_properties & cultural_properties
                )
                
                # Count geographic properties
                features['num_geographic_properties'] = len(
                    unique_properties & geographic_properties
                )
                
                # Specific cultural indicators
                features['has_country'] = 'P17' in claims
                features['has_culture_property'] = 'P2596' in claims
                features['has_origin_country'] = 'P495' in claims
                
                # Count external identifiers
                features['num_identifiers'] = len([
                    p for p in unique_properties if p.startswith('P') and 
                    any(c.get('mainsnak', {}).get('datatype') == 'external-id' 
                        for c in claims.get(p, []))
                ])
                
        except Exception as e:
            logger.debug(f"Error getting Wikidata features for {qid}: {e}")
            
        return features
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """Make HTTP request with retries"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        return None
    
    def extract_all_features(self, item_url: str) -> Dict:
        """Extract all features for a single item"""
        qid = self.extract_qid_from_url(item_url)
        
        if not qid:
            return self._empty_features()
        
        # Get Wikipedia stats
        wiki_stats = self.get_wikipedia_stats(qid)
        
        # Get Wikidata features
        wikidata_features = self.get_wikidata_features(qid)
        
        # Combine all features
        all_features = {**wiki_stats, **wikidata_features}
        
        return all_features
    
    def _empty_features(self) -> Dict:
        """Return empty features dictionary"""
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


def process_single_item(args) -> Tuple[int, Dict]:
    """Process a single item (for parallel processing)"""
    idx, item_url, extractor = args
    try:
        features = extractor.extract_all_features(item_url)
        return idx, features
    except Exception as e:
        logger.error(f"Error processing item {idx} ({item_url}): {e}")
        return idx, extractor._empty_features()


def extract_features_parallel(df: pd.DataFrame, num_workers: int = 32, batch_size: int = 100) -> pd.DataFrame:
    """Extract features for all items in parallel"""
    logger.info(f"Starting feature extraction for {len(df)} items with {num_workers} workers")
    
    extractor = WikiDataExtractor()
    
    # Prepare arguments for parallel processing
    args_list = [(idx, row['item'], extractor) for idx, row in df.iterrows()]
    
    # Store results
    results = {}
    
    # Process in parallel with progress bar
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_item, args): args[0. 
                  for args in args_list}
        
        with tqdm(total=len(futures), desc="Extracting features", unit="items") as pbar:
            for future in as_completed(futures):
                idx, features = future.result()
                results[idx] = features
                pbar.update(1)
                
                # Log progress periodically
                if len(results) % batch_size == 0:
                    logger.info(f" Processed {len(results)}/{len(df)} items ({len(results)/len(df)*100:.1f}%)")
    
    # Convert results to DataFrame
    features_df = pd.DataFrame.from_dict(results, orient='index')
    features_df = features_df.sort_index()
    
    # Combine with original dataframe
    result_df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    
    logger.info(" Feature extraction complete")
    return result_df


def main():
    """Main execution function"""
    logger.info("=" * 80)
    logger.info("Wikipedia/Wikidata Feature Extraction for Cultural Classification")
    logger.info("=" * 80)
    
    # Load dataset
    logger.info("\n Loading training dataset...")
    try:
        df_train = pd.read_csv('train.csv')
        logger.info(f" Successfully loaded {len(df_train)} training samples")
        logger.info(f" Columns: {df_train.columns.tolist()}")
        
        # Display sample
        logger.info("\n Sample data:")
        logger.info(f"\n{df_train.head(3).to_string()}")
        
        # Display label distribution
        logger.info("\n Label distribution:")
        label_counts = df_train['label'].value_counts()
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count} ({count/len(df_train)*100:.1f}%)")
            
    except Exception as e:
        logger.error(f" Error loading dataset: {e}")
        logger.error("\nPlease ensure 'train.csv' is in the current directory")
        logger.error("You can download it from: https://huggingface.co/datasets/sapienzanlp/nlp2025_hw1_cultural_dataset")
        return
    
    # Extract features
    logger.info("\n" + "=" * 80)
    logger.info(" Starting feature extraction...")
    logger.info("=" * 80)
    
    start_time = time.time()
    df_enriched = extract_features_parallel(df_train, num_workers=32, batch_size=100)
    elapsed_time = time.time() - start_time
    
    logger.info(f"\n Feature extraction completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    logger.info(f" Average time per item: {elapsed_time/len(df_train):.3f} seconds")
    logger.info(f" Throughput: {len(df_train)/elapsed_time:.2f} items/second")
    
    # Save enriched dataset
    logger.info("\n" + "=" * 80)
    logger.info(" Saving enriched dataset...")
    logger.info("=" * 80)
    
    output_file_csv = 'train_enriched.csv'
    output_file_tsv = 'train_enriched.tsv'
    
    df_enriched.to_csv(output_file_csv, index=False)
    logger.info(f" Saved to {output_file_csv}")
    
    df_enriched.to_csv(output_file_tsv, sep='\t', index=False)
    logger.info(f" Saved to {output_file_tsv}")
    
    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info(" Feature Statistics Summary")
    logger.info("=" * 80)
    
    feature_cols = [col for col in df_enriched.columns 
                   if col not in df_train.columns]
    
    logger.info(f"\nExtracted {len(feature_cols)} new features:")
    
    for col in feature_cols:
        if df_enriched[col].dtype in ['int64', 'float64']:
            non_zero = (df_enriched[col] != 0).sum()
            logger.info(f"\n   {col}:")
            logger.info(f"     Mean: {df_enriched[col].mean():.2f}")
            logger.info(f"     Median: {df_enriched[col].median():.2f}")
            logger.info(f"     Min: {df_enriched[col].min()}")
            logger.info(f"     Max: {df_enriched[col].max()}")
            logger.info(f"     Non-zero values: {non_zero} ({non_zero/len(df_enriched)*100:.1f}%)")
        elif df_enriched[col].dtype == 'bool':
            true_count = df_enriched[col].sum()
            logger.info(f"\n   {col}:")
            logger.info(f"     True: {true_count} ({true_count/len(df_enriched)*100:.1f}%)")
            logger.info(f"     False: {len(df_enriched) - true_count} ({(len(df_enriched) - true_count)/len(df_enriched)*100:.1f}%)")
    
    # Feature coverage analysis
    logger.info("\n" + "=" * 80)
    logger.info(" Feature Coverage Analysis")
    logger.info("=" * 80)
    
    # Check how many items have at least some data
    items_with_languages = (df_enriched['num_languages'] > 0).sum()
    items_with_statements = (df_enriched['num_statements'] > 0).sum()
    items_with_page_length = (df_enriched['en_page_length'] > 0).sum()
    
    logger.info(f"\nItems with language editions: {items_with_languages} ({items_with_languages/len(df_enriched)*100:.1f}%)")
    logger.info(f"Items with Wikidata statements: {items_with_statements} ({items_with_statements/len(df_enriched)*100:.1f}%)")
    logger.info(f"Items with English Wikipedia page: {items_with_page_length} ({items_with_page_length/len(df_enriched)*100:.1f}%)")
    
    # Analyze by label
    logger.info("\n" + "=" * 80)
    logger.info(" Feature Statistics by Label")
    logger.info("=" * 80)
    
    for label in df_enriched['label'].unique():
        label_data = df_enriched[df_enriched['label'] == label]
        logger.info(f"\n  {label.upper()} (n={len(label_data)}):")
        logger.info(f"   Avg languages: {label_data['num_languages'].mean():.2f}")
        logger.info(f"   Avg statements: {label_data['num_statements'].mean():.2f}")
        logger.info(f"   Avg page length: {label_data['en_page_length'].mean():.2f}")
        logger.info(f"   With country: {label_data['has_country'].sum()} ({label_data['has_country'].sum()/len(label_data)*100:.1f}%)")
    
    logger.info("\n" + "=" * 80)
    logger.info(" Process completed successfully!")
    logger.info(f" Output files:")
    logger.info(f"   • {output_file_csv}")
    logger.info(f"   • {output_file_tsv}")
    logger.info(f"   • feature_extraction.log")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()