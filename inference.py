# inference.py - FIXED VERSION

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import json
import logging
from datetime import datetime
from collections import Counter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
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

# ============================================================================
# CONFIGURATION
# ============================================================================

class InferenceConfig:
    MODEL_PATH = "./best_model_enhanced"
    TEST_PATH = "test_augmented.csv"
    OUTPUT_DIR = "./predictions"
    
    BATCH_SIZE = 32
    MAX_LENGTH = 384
    
    # For ensemble (if you trained k-fold)
    USE_ENSEMBLE = True  # Set True to use all fold models
    ENSEMBLE_METHOD = "weighted"  # "weighted", "voting", or "average"
    
    # Analysis
    GENERATE_PLOTS = False  # Set to False if matplotlib not installed
    SAVE_DETAILED_RESULTS = True

config = InferenceConfig()

# ============================================================================
# SINGLE MODEL CLASSIFIER
# ============================================================================

class CulturalClassifier:
    """Single model inference"""
    
    def __init__(self, model_path, tokenizer_path=None):
        logger.info(f"Loading model from: {model_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer from separate path if provided
        if tokenizer_path is None:
            tokenizer_path = model_path
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f" Tokenizer loaded from: {tokenizer_path}")
        except Exception as e:
            logger.error(f" Failed to load tokenizer from {tokenizer_path}: {e}")
            raise
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        logger.info(f" Model loaded successfully!")
        logger.info(f"Model config: {self.model.config.num_labels} classes")
    
    def create_input_text(self, row):
        """Create formatted input text"""
        parts = [f"Item: {row['name']}"]
        
        if pd.notna(row.get('description')) and str(row.get('description')).strip():
            parts.append(f"Description: {row['description']}")
        
        if pd.notna(row.get('type')) and str(row.get('type')).strip():
            parts.append(f"Type: {row['type']}")
        
        if pd.notna(row.get('category')) and str(row.get('category')).strip():
            parts.append(f"Category: {row['category']}")
        
        if pd.notna(row.get('subcategory')) and str(row.get('subcategory')).strip():
            parts.append(f"Subcategory: {row['subcategory']}")
        
        return ". ".join(parts) + "."
    
    def predict_batch(self, texts, batch_size=32):
        """Predict batch of texts"""
        all_predictions = []
        all_confidences = []
        all_probabilities = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting batches"):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=config.MAX_LENGTH,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
            
            pred_classes = torch.argmax(probs, dim=-1).cpu().numpy()
            batch_probs = probs.cpu().numpy()
            
            for pred_class, prob_dist in zip(pred_classes, batch_probs):
                pred_label = self.id2label[pred_class]
                confidence = prob_dist[pred_class]
                
                all_predictions.append(pred_label)
                all_confidences.append(confidence)
                all_probabilities.append(prob_dist)
        
        return all_predictions, all_confidences, all_probabilities
    
    def predict_dataframe(self, df):
        """Predict for entire dataframe"""
        logger.info(f"Predicting for {len(df)} samples...")
        
        # Create input texts
        texts = [self.create_input_text(row) for _, row in df.iterrows()]
        
        # Predict
        predictions, confidences, probabilities = self.predict_batch(
            texts, 
            batch_size=config.BATCH_SIZE
        )
        
        # Create result dataframe
        result_df = df.copy()
        result_df['predicted_label'] = predictions
        result_df['confidence'] = confidences
        
        # Add probability columns
        for idx, label in self.id2label.items():
            result_df[f'prob_{label.replace(" ", "_")}'] = [p[idx] for p in probabilities]
        
        # Add input text for reference
        result_df['input_text'] = texts
        
        return result_df

# ============================================================================
# ENSEMBLE CLASSIFIER - FIXED VERSION
# ============================================================================

class EnsembleClassifier:
    """Ensemble of multiple models - FIXED to use shared tokenizer"""
    
    def __init__(self, model_paths, tokenizer_path, weights=None, method="weighted"):
        """
        Args:
            model_paths: list of model directories
            tokenizer_path: path to tokenizer (usually best_model)
            weights: list of weights (if None, use equal weights)
            method: "weighted", "voting", or "average"
        """
        logger.info(f"Loading ensemble of {len(model_paths)} models...")
        logger.info(f"Using shared tokenizer from: {tokenizer_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.method = method
        
        # Load shared tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f" Shared tokenizer loaded")
        
        # Load all models
        for i, path in enumerate(tqdm(model_paths, desc="Loading models")):
            if os.path.exists(path):
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(path)
                    model.to(self.device)
                    model.eval()
                    
                    self.models.append(model)
                    logger.info(f"   Loaded model {i+1}: {path}")
                except Exception as e:
                    logger.warning(f"    Failed to load model from {path}: {e}")
            else:
                logger.warning(f"    Model path not found: {path}")
        
        if len(self.models) == 0:
            raise ValueError("No models loaded successfully!")
        
        # Set weights
        if weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()
        
        logger.info(f" Successfully loaded {len(self.models)} models")
        logger.info(f"Ensemble method: {self.method}")
        logger.info(f"Ensemble weights: {self.weights}")
        
        # Use first model's label mapping
        self.id2label = self.models[0].config.id2label
        self.label2id = self.models[0].config.label2id
    
    def create_input_text(self, row):
        """Same as single model"""
        parts = [f"Item: {row['name']}"]
        
        if pd.notna(row.get('description')) and str(row.get('description')).strip():
            parts.append(f"Description: {row['description']}")
        
        if pd.notna(row.get('type')) and str(row.get('type')).strip():
            parts.append(f"Type: {row['type']}")
        
        if pd.notna(row.get('category')) and str(row.get('category')).strip():
            parts.append(f"Category: {row['category']}")
        
        if pd.notna(row.get('subcategory')) and str(row.get('subcategory')).strip():
            parts.append(f"Subcategory: {row['subcategory']}")
        
        return ". ".join(parts) + "."
    
    def predict_batch(self, texts, batch_size=32):
        """Ensemble prediction"""
        all_predictions = []
        all_confidences = []
        all_probabilities = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Ensemble predicting"):
            batch_texts = texts[i:i + batch_size]
            batch_size_actual = len(batch_texts)
            
            # Tokenize once for all models
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=config.MAX_LENGTH,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions from all models
            if self.method == "voting":
                votes = []
                for model in self.models:
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    votes.append(preds)
                
                votes = np.array(votes)  # shape: (n_models, batch_size)
                
                # Majority voting
                for j in range(batch_size_actual):
                    vote_counts = Counter(votes[:, j])
                    pred_class = vote_counts.most_common(1)[0][0]
                    pred_label = self.id2label[pred_class]
                    
                    all_predictions.append(pred_label)
                    all_confidences.append(vote_counts[pred_class] / len(self.models))
                    
                    # Create pseudo-probabilities from votes
                    pseudo_probs = np.zeros(3)
                    for cls, count in vote_counts.items():
                        pseudo_probs[cls] = count / len(self.models)
                    all_probabilities.append(pseudo_probs)
            else:
                # Weighted or average method
                ensemble_probs = np.zeros((batch_size_actual, 3))
                
                for model_idx, model in enumerate(self.models):
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        probs = F.softmax(logits, dim=-1).cpu().numpy()
                    
                    if self.method == "weighted":
                        ensemble_probs += probs * self.weights[model_idx]
                    elif self.method == "average":
                        ensemble_probs += probs / len(self.models)
                
                pred_classes = np.argmax(ensemble_probs, axis=-1)
                
                for pred_class, prob_dist in zip(pred_classes, ensemble_probs):
                    pred_label = self.id2label[pred_class]
                    confidence = prob_dist[pred_class]
                    
                    all_predictions.append(pred_label)
                    all_confidences.append(confidence)
                    all_probabilities.append(prob_dist)
        
        return all_predictions, all_confidences, all_probabilities
    
    def predict_dataframe(self, df):
        """Predict for entire dataframe"""
        logger.info(f"Ensemble predicting for {len(df)} samples...")
        
        texts = [self.create_input_text(row) for _, row in df.iterrows()]
        predictions, confidences, probabilities = self.predict_batch(
            texts, 
            batch_size=config.BATCH_SIZE
        )
        
        result_df = df.copy()
        result_df['predicted_label'] = predictions
        result_df['confidence'] = confidences
        
        for idx, label in self.id2label.items():
            result_df[f'prob_{label.replace(" ", "_")}'] = [p[idx] for p in probabilities]
        
        result_df['input_text'] = texts
        
        return result_df

# ============================================================================
# ANALYSIS
# ============================================================================

def generate_analysis(result_df, output_dir):
    """Generate comprehensive analysis"""
    logger.info("\n" + "="*70)
    logger.info(" Generating Analysis")
    logger.info("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    analysis = {}
    
    # 1. Basic Statistics
    logger.info("\n1⃣ Basic Statistics:")
    logger.info("-" * 50)
    
    pred_dist = result_df['predicted_label'].value_counts()
    logger.info(f"\nPrediction Distribution:")
    for label, count in pred_dist.items():
        pct = (count / len(result_df)) * 100
        logger.info(f"  {label:30s}: {count:4d} ({pct:5.2f}%)")
    
    analysis['prediction_distribution'] = pred_dist.to_dict()
    
    # 2. Confidence Analysis
    logger.info("\n2⃣ Confidence Analysis:")
    logger.info("-" * 50)
    
    conf_stats = {
        'mean': float(result_df['confidence'].mean()),
        'std': float(result_df['confidence'].std()),
        'min': float(result_df['confidence'].min()),
        'max': float(result_df['confidence'].max()),
        'median': float(result_df['confidence'].median()),
        'q25': float(result_df['confidence'].quantile(0.25)),
        'q75': float(result_df['confidence'].quantile(0.75)),
    }
    
    for metric, value in conf_stats.items():
        logger.info(f"  {metric:10s}: {value:.4f}")
    
    analysis['confidence_stats'] = conf_stats
    
    # 3. Confidence by Class
    logger.info("\n3⃣ Confidence by Predicted Class:")
    logger.info("-" * 50)
    
    conf_by_class = {}
    for label in result_df['predicted_label'].unique():
        class_conf = result_df[result_df['predicted_label'] == label]['confidence']
        conf_by_class[label] = {
            'mean': float(class_conf.mean()),
            'std': float(class_conf.std()),
            'count': int(len(class_conf))
        }
        logger.info(f"  {label:30s}: {class_conf.mean():.4f} ± {class_conf.std():.4f} (n={len(class_conf)})")
    
    analysis['confidence_by_class'] = conf_by_class
    
    # 4. Low Confidence Predictions
    logger.info("\n4⃣ Low Confidence Predictions:")
    logger.info("-" * 50)
    
    low_conf_analysis = {}
    thresholds = [0.5, 0.6, 0.7, 0.8]
    for threshold in thresholds:
        low_conf = result_df[result_df['confidence'] < threshold]
        count = len(low_conf)
        pct = (count / len(result_df)) * 100
        low_conf_analysis[f'below_{threshold}'] = {'count': count, 'percentage': pct}
        logger.info(f"  Confidence < {threshold}: {count:4d} ({pct:5.1f}%)")
    
    analysis['low_confidence'] = low_conf_analysis
    
    # 5. Category Analysis
    if 'category' in result_df.columns:
        logger.info("\n5⃣ Top Categories by Prediction:")
        logger.info("-" * 50)
        
        category_analysis = {}
        for category in result_df['category'].value_counts().head(10).index:
            if pd.notna(category) and category:
                cat_data = result_df[result_df['category'] == category]
                cat_dist = cat_data['predicted_label'].value_counts()
                category_analysis[str(category)] = cat_dist.to_dict()
                
                logger.info(f"\n  {category} (n={len(cat_data)}):")
                for label, count in cat_dist.items():
                    logger.info(f"    {label}: {count}")
        
        analysis['top_categories'] = category_analysis
    
    # 6. Type Analysis
    if 'type' in result_df.columns:
        logger.info("\n6⃣ Predictions by Type:")
        logger.info("-" * 50)
        
        type_analysis = {}
        for item_type in result_df['type'].value_counts().index:
            if pd.notna(item_type) and item_type:
                type_data = result_df[result_df['type'] == item_type]
                type_dist = type_data['predicted_label'].value_counts()
                type_analysis[str(item_type)] = type_dist.to_dict()
                
                logger.info(f"\n  {item_type} (n={len(type_data)}):")
                for label, count in type_dist.items():
                    logger.info(f"    {label}: {count}")
        
        analysis['type_analysis'] = type_analysis
    
    # 7. Most/Least Confident Predictions
    logger.info("\n7⃣ Most Confident Predictions (Top 10):")
    logger.info("-" * 50)
    
    most_confident = result_df.nlargest(10, 'confidence')[['name', 'predicted_label', 'confidence']]
    for idx, row in most_confident.iterrows():
        logger.info(f"  {row['name'][:45]:45s} → {row['predicted_label']:30s} ({row['confidence']:.4f})")
    
    logger.info("\n8⃣ Least Confident Predictions (Bottom 10):")
    logger.info("-" * 50)
    
    least_confident = result_df.nsmallest(10, 'confidence')[['name', 'predicted_label', 'confidence']]
    for idx, row in least_confident.iterrows():
        logger.info(f"  {row['name'][:45]:45s} → {row['predicted_label']:30s} ({row['confidence']:.4f})")
    
    # Save analysis
    analysis_path = os.path.join(output_dir, 'analysis_summary.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"\n Analysis saved to: {analysis_path}")
    
    return analysis

# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================

def main():
    """Main inference pipeline"""
    logger.info("="*70)
    logger.info(" Cultural Classification Inference Pipeline")
    logger.info("="*70)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load test data
    logger.info(f"\n Loading test data from: {config.TEST_PATH}")
    test_df = pd.read_csv(config.TEST_PATH)
    logger.info(f" Loaded {len(test_df)} test samples")
    
    # Check for ensemble models
    if config.USE_ENSEMBLE:
        fold_models = [f"./results/fold_{i}_model" for i in range(1, 6)]
        existing_models = [m for m in fold_models if os.path.exists(m)]
        
        if len(existing_models) > 1:
            logger.info(f"\n Using ensemble of {len(existing_models)} models")
            try:
                # Use best_model for tokenizer
                classifier = EnsembleClassifier(
                    existing_models,
                    tokenizer_path=config.MODEL_PATH,
                    method=config.ENSEMBLE_METHOD
                )
            except Exception as e:
                logger.warning(f"  Ensemble failed: {e}")
                logger.info("Falling back to single best model")
                classifier = CulturalClassifier(config.MODEL_PATH)
        else:
            logger.info(f"\n Using single best model (not enough fold models)")
            classifier = CulturalClassifier(config.MODEL_PATH)
    else:
        logger.info(f"\n Using single best model from: {config.MODEL_PATH}")
        classifier = CulturalClassifier(config.MODEL_PATH)
    
    # Make predictions
    logger.info("\n Starting predictions...")
    result_df = classifier.predict_dataframe(test_df)
    
    # Save predictions
    prediction_file = os.path.join(config.OUTPUT_DIR, f'predictions_{timestamp}.csv')
    result_df.to_csv(prediction_file, index=False)
    logger.info(f"\n Full predictions saved to: {prediction_file}")
    
    # Save submission format (only required columns)
    submission_df = result_df[['item', 'name', 'predicted_label', 'confidence']].copy()
    submission_file = os.path.join(config.OUTPUT_DIR, f'submission_{timestamp}.csv')
    submission_df.to_csv(submission_file, index=False)
    logger.info(f" Submission file saved to: {submission_file}")
    
    # Generate analysis
    analysis = generate_analysis(result_df, config.OUTPUT_DIR)
    
    # Save detailed results if enabled
    if config.SAVE_DETAILED_RESULTS:
        # Save low confidence predictions
        low_conf_df = result_df[result_df['confidence'] < 0.7].copy()
        if len(low_conf_df) > 0:
            low_conf_file = os.path.join(config.OUTPUT_DIR, f'low_confidence_{timestamp}.csv')
            low_conf_df.to_csv(low_conf_file, index=False)
            logger.info(f" Low confidence predictions saved to: {low_conf_file}")
        
        # Save per-class results
        for label in result_df['predicted_label'].unique():
            class_df = result_df[result_df['predicted_label'] == label].copy()
            class_file = os.path.join(config.OUTPUT_DIR, f'{label.replace(" ", "_")}_{timestamp}.csv')
            class_df.to_csv(class_file, index=False)
        
        logger.info(f" Per-class predictions saved")
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info(" Inference Complete!")
    logger.info("="*70)
    logger.info(f" Total predictions: {len(result_df)}")
    logger.info(f" Mean confidence: {result_df['confidence'].mean():.4f}")
    logger.info(f" Results saved to: {config.OUTPUT_DIR}/")
    logger.info(f" Main files:")
    logger.info(f"   - predictions_{timestamp}.csv (full results)")
    logger.info(f"   - submission_{timestamp}.csv (submission format)")
    logger.info(f"   - analysis_summary.json (detailed statistics)")
    logger.info("="*70)
    
    return result_df

if __name__ == "__main__":
    results = main()