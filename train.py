# train.py - ENHANCED VERSION

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from datasets import Dataset as HFDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import wandb
from tqdm.auto import tqdm
import json
from datetime import datetime
import logging
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

# Set random seed for reproducibility
SEED = 42
set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================
class Config:
    # Paths
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    OUTPUT_DIR = "./results"
    BEST_MODEL_DIR = "./best_model"
    
    # Model Selection - Try multiple models for ensemble
    MODELS = [
        "microsoft/mdeberta-v3-base",      # Best single model
        # "xlm-roberta-large",              # Uncomment for ensemble
        # "microsoft/deberta-v3-large",     # Uncomment if you have 32GB GPU
    ]
    PRIMARY_MODEL = "microsoft/mdeberta-v3-base"
    
    # Tokenization
    MAX_LENGTH = 384  # Increased from 256 for better context
    
    # Training Hyperparameters - OPTIMIZED
    BATCH_SIZE = 8  # Reduced for stability, using gradient accumulation
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16
    EVAL_BATCH_SIZE = 16
    
    LEARNING_RATE = 1.5e-5  # Slightly lower for better convergence
    NUM_EPOCHS = 8  # Reduced from 10, use early stopping
    WARMUP_RATIO = 0.06  # 6% warmup steps
    WEIGHT_DECAY = 0.01
    
    # Advanced optimization
    MAX_GRAD_NORM = 1.0
    LABEL_SMOOTHING = 0.1  # Helps with overconfidence
    
    # Learning rate scheduler
    LR_SCHEDULER_TYPE = "cosine"  # cosine, linear, polynomial
    
    # K-Fold
    USE_KFOLD = True
    N_FOLDS = 5
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3
    
    # Data Augmentation
    USE_AUGMENTATION = False  # Set True for data augmentation
    
    # Ensemble
    USE_ENSEMBLE = False  # Set True to train multiple models
    
    # Logging
    USE_WANDB = True
    PROJECT_NAME = "cultural-classification"
    
    # Hardware
    FP16 = True
    DATALOADER_NUM_WORKERS = 4
    
    # Regularization
    DROPOUT = 0.1
    ATTENTION_DROPOUT = 0.1
    HIDDEN_DROPOUT = 0.1

config = Config()

# Label mapping
LABEL2ID = {
    "cultural agnostic": 0,
    "cultural representative": 1,
    "cultural exclusive": 2
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(train_path, test_path):
    """Load and validate data with enhanced preprocessing"""
    logger.info("Loading data...")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Test size: {len(test_df)}")
    logger.info(f"Train label distribution:\n{train_df['label'].value_counts()}")
    
    # Enhanced missing value handling
    for col in ['description', 'category', 'subcategory', 'type']:
        train_df[col] = train_df[col].fillna('').astype(str)
        test_df[col] = test_df[col].fillna('').astype(str)
    
    # Clean text
    train_df['name'] = train_df['name'].str.strip()
    train_df['description'] = train_df['description'].str.strip()
    test_df['name'] = test_df['name'].str.strip()
    test_df['description'] = test_df['description'].str.strip()
    
    return train_df, test_df

def create_input_text_v1(row):
    """Simple format - good baseline"""
    text = f"{row['name']}. {row['description']}"
    return text.strip()

def create_input_text_v2(row):
    """Structured format - BEST for classification"""
    parts = [f"Item: {row['name']}"]
    
    if row['description']:
        parts.append(f"Description: {row['description']}")
    
    if row['type']:
        parts.append(f"Type: {row['type']}")
    
    if row['category']:
        parts.append(f"Category: {row['category']}")
    
    if row['subcategory']:
        parts.append(f"Subcategory: {row['subcategory']}")
    
    return ". ".join(parts) + "."

def create_input_text_v3(row):
    """Template-based format with explicit cultural framing"""
    text = f"""Classify the cultural specificity of this item:
Name: {row['name']}
Description: {row['description']}
Category: {row['category']}
Type: {row['type']}"""
    return text.strip()

# Use the best format
create_input_text = create_input_text_v2

# ============================================================================
# DATA AUGMENTATION (Optional)
# ============================================================================

def augment_text(text):
    """Simple text augmentation techniques"""
    import random
    
    augmented = []
    
    # Original
    augmented.append(text)
    
    # Paraphrase templates (simple version)
    if random.random() > 0.5:
        # Shuffle sentence order slightly
        sentences = text.split('. ')
        if len(sentences) > 2:
            random.shuffle(sentences)
            augmented.append('. '.join(sentences))
    
    return augmented

# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_dataset(df, tokenizer, is_test=False, augment=False):
    """Prepare dataset with enhanced preprocessing"""
    texts = df.apply(create_input_text, axis=1).tolist()
    
    # Data augmentation
    if augment and not is_test and config.USE_AUGMENTATION:
        logger.info("Applying data augmentation...")
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, df['label']):
            aug_texts = augment_text(text)
            augmented_texts.extend(aug_texts)
            augmented_labels.extend([label] * len(aug_texts))
        
        texts = augmented_texts
        data_dict = {'text': texts, 'label': [LABEL2ID[l] for l in augmented_labels]}
    else:
        data_dict = {'text': texts}
        if not is_test:
            data_dict['label'] = [LABEL2ID[label] for label in df['label']]
    
    dataset = HFDataset.from_dict(data_dict)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_attention_mask=True,
            return_token_type_ids=False  # Not needed for DeBERTa
        )
    
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing",
        remove_columns=['text']
    )
    
    return dataset

# ============================================================================
# ENHANCED METRICS
# ============================================================================

def compute_metrics(eval_pred):
    """Compute comprehensive evaluation metrics"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    per_class_precision, per_class_recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1,
        'f1_weighted': f1_weighted,
        'precision_macro': precision,
        'recall_macro': recall,
    }
    
    # Add per-class metrics
    for idx, class_name in ID2LABEL.items():
        if idx < len(per_class_f1):
            metrics[f'f1_{class_name.replace(" ", "_")}'] = per_class_f1[idx]
            metrics[f'precision_{class_name.replace(" ", "_")}'] = per_class_precision[idx]
            metrics[f'recall_{class_name.replace(" ", "_")}'] = per_class_recall[idx]
    
    return metrics

# ============================================================================
# MODEL INITIALIZATION WITH CUSTOM CONFIG
# ============================================================================

def create_model(model_name):
    """Create model with custom configuration for better performance"""
    
    # Load config
    model_config = AutoConfig.from_pretrained(model_name)
    
    # Customize config for better performance
    model_config.num_labels = 3
    model_config.id2label = ID2LABEL
    model_config.label2id = LABEL2ID
    model_config.problem_type = "single_label_classification"
    
    # Enhanced regularization
    model_config.hidden_dropout_prob = config.HIDDEN_DROPOUT
    model_config.attention_probs_dropout_prob = config.ATTENTION_DROPOUT
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=model_config,
        ignore_mismatched_sizes=True
    )
    
    return model

# ============================================================================
# WEIGHTED LOSS FOR CLASS IMBALANCE
# ============================================================================

class WeightedTrainer(Trainer):
    """Custom Trainer with weighted loss for class imbalance"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply class weights
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                label_smoothing=config.LABEL_SMOOTHING
            )
        else:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def create_trainer(model, tokenizer, train_dataset, eval_dataset, fold=None, class_weights=None):
    """Create Trainer with enhanced configuration"""
    
    # Create output directory
    if fold is not None:
        output_dir = f"{config.OUTPUT_DIR}/fold_{fold}"
    else:
        output_dir = config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Training hyperparameters
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        
        # Optimization
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        max_grad_norm=config.MAX_GRAD_NORM,
        
        # Mixed precision
        fp16=config.FP16,
        fp16_full_eval=config.FP16,
        
        # Evaluation and saving
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=20,
        logging_first_step=True,
        report_to=["wandb"] if config.USE_WANDB else ["none"],
        
        # Hardware optimization
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=True,
        group_by_length=False,  # Can help with efficiency
        
        # Reproducibility
        seed=SEED,
        data_seed=SEED,
        
        # Advanced
        remove_unused_columns=True,
        label_smoothing_factor=config.LABEL_SMOOTHING,
    )
    
    # Callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=0.0001
        )
    ]
    
    # Create trainer with weighted loss
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights
    )
    
    return trainer

# ============================================================================
# K-FOLD TRAINING
# ============================================================================

def train_single_fold(train_df, fold_idx, train_indices, val_indices, tokenizer, class_weights=None):
    """Train a single fold with enhanced logging"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Training Fold {fold_idx + 1}/{config.N_FOLDS}")
    logger.info(f"{'='*70}")
    
    # Split data
    fold_train_df = train_df.iloc[train_indices].reset_index(drop=True)
    fold_val_df = train_df.iloc[val_indices].reset_index(drop=True)
    
    logger.info(f"Fold {fold_idx + 1} - Train: {len(fold_train_df)}, Val: {len(fold_val_df)}")
    logger.info(f"Train label distribution:\n{fold_train_df['label'].value_counts()}")
    
    # Prepare datasets
    fold_train_dataset = prepare_dataset(fold_train_df, tokenizer, augment=True)
    fold_val_dataset = prepare_dataset(fold_val_df, tokenizer)
    
    # Initialize model
    model = create_model(config.PRIMARY_MODEL)
    
    # Initialize WandB run
    if config.USE_WANDB:
        run_name = f"fold_{fold_idx + 1}_{datetime.now().strftime('%H%M%S')}"
        wandb.init(
            project=config.PROJECT_NAME,
            name=run_name,
            config={
                "model": config.PRIMARY_MODEL,
                "fold": fold_idx + 1,
                "batch_size": config.BATCH_SIZE,
                "effective_batch_size": config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS,
                "learning_rate": config.LEARNING_RATE,
                "epochs": config.NUM_EPOCHS,
                "max_length": config.MAX_LENGTH,
                "label_smoothing": config.LABEL_SMOOTHING,
            },
            reinit=True
        )
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=fold_train_dataset,
        eval_dataset=fold_val_dataset,
        fold=fold_idx,
        class_weights=class_weights
    )
    
    # Train
    logger.info(f"Starting training for fold {fold_idx + 1}...")
    train_result = trainer.train()
    
    # Log training metrics
    logger.info(f"Training completed. Final loss: {train_result.training_loss:.4f}")
    
    # Evaluate
    logger.info(f"Evaluating fold {fold_idx + 1}...")
    eval_results = trainer.evaluate()
    
    # Detailed results logging
    logger.info(f"\nFold {fold_idx + 1} Results:")
    logger.info("-" * 50)
    for key, value in sorted(eval_results.items()):
        if 'loss' not in key:
            logger.info(f"  {key:30s}: {value:.4f}")
    
    # Confusion matrix
    predictions = trainer.predict(fold_val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\nConfusion Matrix for Fold {fold_idx + 1}:")
    logger.info(f"\n{cm}")
    
    # Classification report
    logger.info(f"\nClassification Report for Fold {fold_idx + 1}:")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=list(ID2LABEL.values()))}")
    
    # Save model
    fold_model_dir = f"{config.OUTPUT_DIR}/fold_{fold_idx + 1}_model"
    trainer.save_model(fold_model_dir)
    logger.info(f"Model saved to {fold_model_dir}")
    
    if config.USE_WANDB:
        wandb.finish()
    
    return eval_results, trainer.model

def train_with_kfold(train_df, tokenizer):
    """Enhanced K-Fold cross-validation with class weights"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting {config.N_FOLDS}-Fold Cross-Validation")
    logger.info(f"{'='*70}")
    
    # Calculate class weights for imbalanced data
    label_counts = train_df['label'].value_counts()
    total = len(train_df)
    class_weights = torch.FloatTensor([
        total / (len(LABEL2ID) * label_counts[label]) 
        for label in sorted(LABEL2ID.keys())
    ])
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS,
        shuffle=True,
        random_state=SEED
    )
    
    fold_results = []
    best_models = []
    
    # K-Fold training with progress bar
    for fold_idx, (train_indices, val_indices) in enumerate(
        skf.split(train_df, train_df['label'])
    ):
        eval_results, model = train_single_fold(
            train_df, fold_idx, train_indices, val_indices, 
            tokenizer, class_weights
        )
        fold_results.append(eval_results)
        best_models.append(model)
    
    # Aggregate results
    logger.info(f"\n{'='*70}")
    logger.info("K-Fold Cross-Validation Summary")
    logger.info(f"{'='*70}")
    
    metrics_summary = {}
    for metric in fold_results[0].keys():
        if 'loss' not in metric:
            values = [result[metric] for result in fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            metrics_summary[metric] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'values': [float(v) for v in values]
            }
            logger.info(f"{metric:30s}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save summary
    with open(f"{config.OUTPUT_DIR}/kfold_summary.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Select best fold
    best_fold_idx = np.argmax([r['eval_f1_macro'] for r in fold_results])
    best_score = fold_results[best_fold_idx]['eval_f1_macro']
    logger.info(f"\n🏆 Best Fold: {best_fold_idx + 1} with F1 Macro: {best_score:.4f}")
    
    return best_models[best_fold_idx], fold_results

# ============================================================================
# TEST SET PREDICTION
# ============================================================================

def predict_test_set(model, tokenizer, test_df):
    """Generate predictions with confidence scores"""
    logger.info("\n" + "="*70)
    logger.info("Generating predictions for test set...")
    logger.info("="*70)
    
    # Prepare test dataset
    test_dataset = prepare_dataset(test_df, tokenizer, is_test=True)
    
    # Create trainer for prediction
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        fp16=config.FP16,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
    )
    
    # Predict
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    pred_probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    
    # Create submission dataframe
    submission_df = test_df.copy()
    submission_df['predicted_label'] = [ID2LABEL[pred] for pred in pred_labels]
    submission_df['confidence'] = np.max(pred_probs, axis=-1)
    
    # Add probabilities for each class
    for idx, label in ID2LABEL.items():
        submission_df[f'prob_{label.replace(" ", "_")}'] = pred_probs[:, idx]
    
    # Save predictions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = f"{config.OUTPUT_DIR}/test_predictions_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"✅ Predictions saved to {submission_path}")
    
    # Statistics
    logger.info("\n📊 Prediction Statistics:")
    logger.info("-" * 50)
    logger.info(f"\nPrediction distribution:")
    logger.info(submission_df['predicted_label'].value_counts())
    logger.info(f"\nConfidence statistics:")
    logger.info(f"  Mean: {submission_df['confidence'].mean():.4f}")
    logger.info(f"  Std:  {submission_df['confidence'].std():.4f}")
    logger.info(f"  Min:  {submission_df['confidence'].min():.4f}")
    logger.info(f"  Max:  {submission_df['confidence'].max():.4f}")
    
    # Low confidence predictions
    low_conf_threshold = 0.6
    low_conf = submission_df[submission_df['confidence'] < low_conf_threshold]
    logger.info(f"\n⚠️  Low confidence predictions (< {low_conf_threshold}): {len(low_conf)}")
    
    return submission_df

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main training pipeline"""
    logger.info("="*70)
    logger.info("🚀 Cultural Classification Training Pipeline - ENHANCED")
    logger.info("="*70)
    
    # Print configuration
    logger.info(f"\n📋 Configuration:")
    logger.info(f"  Model: {config.PRIMARY_MODEL}")
    logger.info(f"  Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Effective Batch Size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"  LR Scheduler: {config.LR_SCHEDULER_TYPE}")
    logger.info(f"  Epochs: {config.NUM_EPOCHS}")
    logger.info(f"  Max Length: {config.MAX_LENGTH}")
    logger.info(f"  Label Smoothing: {config.LABEL_SMOOTHING}")
    logger.info(f"  Use K-Fold: {config.USE_KFOLD}")
    logger.info(f"  N Folds: {config.N_FOLDS if config.USE_KFOLD else 'N/A'}")
    logger.info(f"  FP16: {config.FP16}")
    logger.info(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Create directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.BEST_MODEL_DIR, exist_ok=True)
    
    # Load data
    train_df, test_df = load_data(config.TRAIN_PATH, config.TEST_PATH)
    
    # Load tokenizer
    logger.info(f"\n📚 Loading tokenizer: {config.PRIMARY_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.PRIMARY_MODEL, use_fast=True)
    
    # Train model
    if config.USE_KFOLD:
        best_model, results = train_with_kfold(train_df, tokenizer)
    else:
        # Simple split training (not shown for brevity, similar to before)
        raise NotImplementedError("Use K-Fold for best results")
    
    # Save best model
    logger.info(f"\n💾 Saving best model to {config.BEST_MODEL_DIR}")
    best_model.save_pretrained(config.BEST_MODEL_DIR)
    tokenizer.save_pretrained(config.BEST_MODEL_DIR)
    
    # Generate test predictions
    test_predictions = predict_test_set(best_model, tokenizer, test_df)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("✅ Training Complete!")
    logger.info("="*70)
    logger.info(f"📁 Best model saved to: {config.BEST_MODEL_DIR}")
    logger.info(f"📊 Test predictions saved to: {config.OUTPUT_DIR}/test_predictions_*.csv")
    logger.info(f"📝 Training logs saved to: training.log")
    logger.info(f"📈 K-Fold summary saved to: {config.OUTPUT_DIR}/kfold_summary.json")
    logger.info("="*70)

if __name__ == "__main__":
    main()