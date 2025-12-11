#!/usr/bin/env python3
"""
Test Script for NLP HW1 - LM-based Classification with Data Augmentation
Loads model from HuggingFace and evaluates on augmented test data
"""

import asyncio
import aiohttp
from aiohttp import ClientSession
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AsyncWikipediaAugmenter:
    """Ultra-fast async Wikipedia augmentation - same as training"""
    
    def __init__(self, cache_file='wiki_cache_test.json', max_concurrent=50):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"Initialized augmenter with cache: {cache_file}")
    
    def load_cache(self):
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                logger.info(f"Loaded {len(cache)} cached entries")
                return cache
        logger.info("No cache file found, starting fresh")
        return {}
    
    def save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.cache)} entries to cache")
    
    async def fetch_wikipedia(self, session, item_name):
        """Fetch Wikipedia data async"""
        cache_key = item_name.lower()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = {'summary': '', 'found': False, 'languages': 0}
        
        try:
            async with self.semaphore:
                # Use Wikipedia API
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{item_name.replace(' ', '_')}"
                
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        result['found'] = True
                        result['summary'] = data.get('extract', '')[:500]
                        
                        # Get language count
                        lang_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={item_name}&prop=langlinks&lllimit=500&format=json"
                        async with session.get(lang_url, timeout=5) as lang_response:
                            if lang_response.status == 200:
                                lang_data = await lang_response.json()
                                pages = lang_data.get('query', {}).get('pages', {})
                                for page in pages.values():
                                    result['languages'] = len(page.get('langlinks', []))
        
        except Exception as e:
            logger.debug(f"Failed to fetch Wikipedia data for {item_name}: {str(e)}")
        
        self.cache[cache_key] = result
        return result
    
    async def augment_row_async(self, session, row):
        """Augment single row async"""
        wiki_data = await self.fetch_wikipedia(session, row['name'])
        
        enriched_parts = [row['description']]
        
        if wiki_data['summary']:
            enriched_parts.append(f"Wikipedia: {wiki_data['summary']}")
        
        if wiki_data['languages'] > 50:
            enriched_parts.append("Available in 50+ languages.")
        elif wiki_data['languages'] > 20:
            enriched_parts.append(f"Available in {wiki_data['languages']} languages.")
        
        augmented_row = row.copy()
        augmented_row['description'] = ' '.join(enriched_parts)
        augmented_row['wiki_found'] = wiki_data['found']
        augmented_row['wiki_languages'] = wiki_data['languages']
        
        return augmented_row
    
    async def augment_dataframe_async(self, df):
        """Augment entire dataframe async"""
        logger.info(f"Starting augmentation for {len(df)} rows...")
        async with ClientSession() as session:
            tasks = []
            for _, row in df.iterrows():
                task = self.augment_row_async(session, row)
                tasks.append(task)
            
            results = []
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Augmenting"):
                result = await f
                results.append(result)
        
        self.save_cache()
        logger.info("Augmentation complete!")
        return pd.DataFrame(results)
    
    def augment_dataframe(self, df):
        """Sync wrapper for async augmentation"""
        return asyncio.run(self.augment_dataframe_async(df))


class TestDataset(Dataset):
    """Test dataset with augmented descriptions"""
    
    def __init__(self, df, tokenizer, max_length=512, label_map=None):
        self.texts = df['description'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Handle label encoding
        if 'label' in df.columns:
            if label_map is None:
                # Auto-create label mapping
                unique_labels = sorted(df['label'].unique())
                self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            else:
                self.label_map = label_map
            
            # Convert string labels to integers
            self.labels = df['label'].map(self.label_map).values
            self.label_names = {v: k for k, v in self.label_map.items()}
        else:
            self.labels = None
            self.label_map = None
            self.label_names = None
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


def load_model_and_tokenizer(model_name):
    """Load model and tokenizer from HuggingFace"""
    logger.info(f"Loading model from HuggingFace: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        logger.info(f"[OK] Model loaded successfully")
        logger.info(f"  Model type: {model.config.model_type}")
        logger.info(f"  Num labels: {model.config.num_labels}")
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def predict(model, dataloader, device):
    """Run inference on test data"""
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    logger.info("Running inference...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            
            if 'labels' in batch:
                all_labels.extend(batch['labels'].numpy())
    
    logger.info(f"[OK] Inference complete: {len(all_predictions)} predictions")
    
    return np.array(all_predictions), np.array(all_labels) if all_labels else None, np.array(all_logits)


def save_results(test_df, predictions, logits, label_names_map=None, output_dir='test_results'):
    """Save predictions and results"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save predictions
    results_df = test_df.copy()
    results_df['predicted_label'] = predictions
    
    # Add predicted label names if mapping is available
    if label_names_map:
        results_df['predicted_label_name'] = results_df['predicted_label'].map(label_names_map)
    
    results_df['prediction_confidence'] = np.max(logits, axis=1)
    
    # Add probability columns for each class
    num_classes = logits.shape[1]
    for i in range(num_classes):
        col_name = f'prob_class_{i}'
        if label_names_map and i in label_names_map:
            col_name += f'_{label_names_map[i].replace(" ", "_")}'
        results_df[col_name] = logits[:, i]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_dir}/predictions_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"[OK] Saved predictions to {results_file}")
    
    return results_file


def evaluate_and_report(true_labels, predictions, label_names=None):
    """Generate comprehensive evaluation report"""
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    
    # Overall metrics
    accuracy = (true_labels == predictions).mean()
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Accuracy:      {accuracy:.4f}")
    logger.info(f"  F1 (Macro):    {f1_macro:.4f}")
    logger.info(f"  F1 (Weighted): {f1_weighted:.4f}")
    
    # Classification report
    logger.info(f"\n{'='*80}")
    logger.info("Detailed Classification Report:")
    logger.info(f"{'='*80}\n")
    
    report = classification_report(
        true_labels, 
        predictions, 
        target_names=label_names,
        digits=4
    )
    logger.info(report)
    
    # Confusion matrix
    logger.info(f"\n{'='*80}")
    logger.info("Confusion Matrix:")
    logger.info(f"{'='*80}\n")
    
    cm = confusion_matrix(true_labels, predictions)
    logger.info(f"\n{cm}\n")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_results/classification_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST EVALUATION REPORT\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:      {accuracy:.4f}\n")
        f.write(f"  F1 (Macro):    {f1_macro:.4f}\n")
        f.write(f"  F1 (Weighted): {f1_weighted:.4f}\n\n")
        f.write("="*80 + "\n")
        f.write("Classification Report:\n")
        f.write("="*80 + "\n\n")
        f.write(report)
        f.write("\n\n" + "="*80 + "\n")
        f.write("Confusion Matrix:\n")
        f.write("="*80 + "\n\n")
        f.write(str(cm))
    
    logger.info(f"\n[OK] Saved detailed report to {report_file}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'report': report,
        'confusion_matrix': cm
    }


def main():
    # Configuration
    MODEL_NAME = "ArchitRastogi/NLP_HW1_LM_tuned"
    TEST_FILE = "test.csv"
    BATCH_SIZE = 16
    MAX_LENGTH = 512
    
    logger.info("="*80)
    logger.info("NLP HW1 - Model Testing with Data Augmentation")
    logger.info("="*80)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nDevice: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load test data
    logger.info(f"\nLoading test data from {TEST_FILE}...")
    test_df = pd.read_csv(TEST_FILE)
    logger.info(f"[OK] Loaded {len(test_df)} test samples")
    logger.info(f"  Columns: {list(test_df.columns)}")
    
    if 'label' in test_df.columns:
        logger.info(f"  Label distribution:")
        for label, count in test_df['label'].value_counts().sort_index().items():
            logger.info(f"    Label {label}: {count} samples")
    
    # Augment test data (same as training)
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Data Augmentation")
    logger.info("="*80)
    
    augmenter = AsyncWikipediaAugmenter(max_concurrent=50)
    test_aug = augmenter.augment_dataframe(test_df)
    
    # Save augmented test data
    test_aug.to_csv("test_augmented.csv", index=False)
    logger.info("[OK] Saved augmented test data to test_augmented.csv")
    
    # Show augmentation stats
    if 'wiki_found' in test_aug.columns:
        found = test_aug['wiki_found'].sum()
        logger.info(f"\nAugmentation Statistics:")
        logger.info(f"  Wikipedia data found: {found}/{len(test_aug)} ({found/len(test_aug)*100:.1f}%)")
        logger.info(f"  Avg description length:")
        logger.info(f"    Before: {test_df['description'].str.len().mean():.0f} chars")
        logger.info(f"    After:  {test_aug['description'].str.len().mean():.0f} chars")
    
    # Load model
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Model Loading")
    logger.info("="*80)
    
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    # Prepare dataset
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Dataset Preparation")
    logger.info("="*80)
    
    test_dataset = TestDataset(test_aug, tokenizer, max_length=MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"[OK] Created DataLoader")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Num batches: {len(test_loader)}")
    
    # Log label mapping if available
    if test_dataset.label_map:
        logger.info(f"\n  Label mapping:")
        for label_str, label_idx in sorted(test_dataset.label_map.items(), key=lambda x: x[1]):
            logger.info(f"    {label_idx}: {label_str}")
    
    # Run inference
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Inference")
    logger.info("="*80)
    
    predictions, true_labels, logits = predict(model, test_loader, device)
    
    # Save predictions
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Saving Results")
    logger.info("="*80)
    
    results_file = save_results(
        test_aug, 
        predictions, 
        logits, 
        label_names_map=test_dataset.label_names if test_dataset.label_names else None
    )
    
    # Evaluate if labels are available
    if true_labels is not None:
        logger.info("\n" + "="*80)
        logger.info("STEP 6: Evaluation")
        logger.info("="*80)
        
        # Get label names from dataset
        label_names = [test_dataset.label_names[i] for i in sorted(test_dataset.label_names.keys())]
        
        metrics = evaluate_and_report(true_labels, predictions, label_names)
        
        # Save metrics to JSON
        metrics_file = f"test_results/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'accuracy': float(metrics['accuracy']),
                'f1_macro': float(metrics['f1_macro']),
                'f1_weighted': float(metrics['f1_weighted']),
                'confusion_matrix': metrics['confusion_matrix'].tolist(),
                'label_mapping': test_dataset.label_map
            }, f, indent=2)
        logger.info(f"[OK] Saved metrics to {metrics_file}")
    
    else:
        logger.info("\nNo ground truth labels found - skipping evaluation")
    
    logger.info("\n" + "="*80)
    logger.info("[OK] TESTING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nResults saved in:")
    logger.info(f"  - {results_file}")
    logger.info(f"  - test_results/ directory")
    logger.info(f"  - Log file: test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


if __name__ == "__main__":
    main()