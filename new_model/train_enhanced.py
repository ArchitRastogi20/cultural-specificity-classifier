# train_enhanced_fixed.py - FIXED VERSION

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # ✅ Fix tokenizer warning

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
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
        logging.FileHandler('training_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seed
SEED = 42
set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================
class Config:
    # Paths - USE AUGMENTED DATA
    TRAIN_PATH = "train_augmented.csv"
    TEST_PATH = "test_augmented.csv"
    OUTPUT_DIR = "./results_enhanced"
    BEST_MODEL_DIR = "./best_model_enhanced"
    
    # Model Selection
    PRIMARY_MODEL = "microsoft/deberta-v3-large"
    
    # Tokenization
    MAX_LENGTH = 512
    
    # Training Hyperparameters
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    EVAL_BATCH_SIZE = 8
    
    LEARNING_RATE = 8e-6
    NUM_EPOCHS = 12
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Advanced optimization
    MAX_GRAD_NORM = 0.5
    LABEL_SMOOTHING = 0.1
    
    # Learning rate scheduler
    LR_SCHEDULER_TYPE = "cosine_with_restarts"
    
    # K-Fold
    USE_KFOLD = True
    N_FOLDS = 5
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 4
    
    # Focal Loss
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    
    # Logging
    USE_WANDB = True
    PROJECT_NAME = "cultural-classification-enhanced"
    
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
# FOCAL LOSS FOR CLASS IMBALANCE - FIXED
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - FIXED device handling"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # Will be moved to device when needed
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # ✅ FIX: Move alpha to same device as inputs
        alpha = self.alpha
        if alpha is not None:
            alpha = alpha.to(inputs.device)
        
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none', 
            weight=alpha,
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ============================================================================
# ENHANCED TEXT PREPROCESSING
# ============================================================================

def create_input_text_enhanced(row):
    """Enhanced preprocessing with explicit cultural signals"""
    
    parts = []
    
    # Item name
    parts.append(f"Item: {row['name']}")
    
    # Description (already augmented with Wikipedia!)
    desc = row['description']
    parts.append(f"Description: {desc}")
    
    # Type with more context
    if row['type'] == 'entity':
        parts.append(f"Type: Named Entity (specific instance)")
    elif row['type'] == 'concept':
        parts.append(f"Type: General Concept (abstract/general)")
    
    # Category and subcategory
    if pd.notna(row['category']) and row['category']:
        parts.append(f"Category: {row['category']}")
    
    if pd.notna(row['subcategory']) and row['subcategory']:
        parts.append(f"Subcategory: {row['subcategory']}")
    
    # Add Wikipedia language availability as cultural signal
    if 'wiki_languages' in row and pd.notna(row['wiki_languages']):
        lang_count = row['wiki_languages']
        if lang_count > 50:
            parts.append("[Global reach: 50+ languages]")
        elif lang_count > 20:
            parts.append(f"[International: {lang_count} languages]")
        elif lang_count > 0 and lang_count < 5:
            parts.append(f"[Regional: {lang_count} languages]")
    
    # Combine with explicit task instruction
    text = ". ".join(parts) + ". "
    text += "Task: Classify cultural specificity as agnostic, representative, or exclusive."
    
    return text

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(train_path, test_path):
    """Load augmented data"""
    logger.info("Loading augmented data...")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Test size: {len(test_df)}")
    logger.info(f"Train label distribution:\n{train_df['label'].value_counts()}")
    
    # Check if Wikipedia augmentation worked
    if 'wiki_found' in train_df.columns:
        wiki_found = train_df['wiki_found'].sum()
        logger.info(f"Wikipedia data found: {wiki_found}/{len(train_df)} ({wiki_found/len(train_df)*100:.1f}%)")
    
    # Fill missing values
    for col in ['description', 'category', 'subcategory', 'type']:
        train_df[col] = train_df[col].fillna('').astype(str)
        test_df[col] = test_df[col].fillna('').astype(str)
    
    return train_df, test_df

def prepare_dataset(df, tokenizer, is_test=False):
    """Prepare dataset with enhanced preprocessing"""
    texts = df.apply(create_input_text_enhanced, axis=1).tolist()
    
    data_dict = {'text': texts}
    
    if not is_test:
        labels = [LABEL2ID[label] for label in df['label']]
        data_dict['label'] = labels
    
    dataset = HFDataset.from_dict(data_dict)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_attention_mask=True,
            return_token_type_ids=False
        )
    
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing",
        remove_columns=['text']
    )
    
    return dataset

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_pred):
    """Compute comprehensive evaluation metrics"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
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
    
    for idx, class_name in ID2LABEL.items():
        if idx < len(per_class_f1):
            metrics[f'f1_{class_name.replace(" ", "_")}'] = per_class_f1[idx]
            metrics[f'precision_{class_name.replace(" ", "_")}'] = per_class_precision[idx]
            metrics[f'recall_{class_name.replace(" ", "_")}'] = per_class_recall[idx]
    
    return metrics

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def create_model(model_name):
    """Create model with custom configuration"""
    
    model_config = AutoConfig.from_pretrained(model_name)
    
    model_config.num_labels = 3
    model_config.id2label = ID2LABEL
    model_config.label2id = LABEL2ID
    model_config.problem_type = "single_label_classification"
    
    model_config.hidden_dropout_prob = config.HIDDEN_DROPOUT
    model_config.attention_probs_dropout_prob = config.ATTENTION_DROPOUT
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=model_config,
        ignore_mismatched_sizes=True
    )
    
    return model

# ============================================================================
# CUSTOM TRAINER WITH FOCAL LOSS - FIXED
# ============================================================================

class EnhancedTrainer(Trainer):
    """Trainer with Focal Loss - FIXED device handling"""
    
    def __init__(self, *args, class_weights=None, use_focal_loss=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # Keep on CPU, will move in loss
        self.use_focal_loss = use_focal_loss
        
        if use_focal_loss:
            self.focal_loss = FocalLoss(
                alpha=class_weights,  # Will be moved to device in forward
                gamma=config.FOCAL_GAMMA,
                label_smoothing=config.LABEL_SMOOTHING
            )
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.use_focal_loss:
            loss = self.focal_loss(logits, labels)
        else:
            # ✅ FIX: Move class_weights to device
            weight = None
            if self.class_weights is not None:
                weight = self.class_weights.to(logits.device)
            
            loss_fct = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=config.LABEL_SMOOTHING
            )
            loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def create_trainer(model, tokenizer, train_dataset, eval_dataset, fold=None, class_weights=None):
    """Create enhanced trainer"""
    
    if fold is not None:
        output_dir = f"{config.OUTPUT_DIR}/fold_{fold}"
    else:
        output_dir = config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Training
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
        logging_steps=50,
        logging_first_step=True,
        report_to=["wandb"] if config.USE_WANDB else ["none"],
        
        # Hardware
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=True,
        
        # Reproducibility
        seed=SEED,
        data_seed=SEED,
        
        # Advanced
        remove_unused_columns=True,
        label_smoothing_factor=config.LABEL_SMOOTHING,
    )
    
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=0.0001
        )
    ]
    
    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights,
        use_focal_loss=config.USE_FOCAL_LOSS
    )
    
    return trainer

# ============================================================================
# K-FOLD TRAINING
# ============================================================================

def train_single_fold(train_df, fold_idx, train_indices, val_indices, tokenizer, class_weights=None):
    """Train a single fold"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Training Fold {fold_idx + 1}/{config.N_FOLDS}")
    logger.info(f"{'='*70}")
    
    fold_train_df = train_df.iloc[train_indices].reset_index(drop=True)
    fold_val_df = train_df.iloc[val_indices].reset_index(drop=True)
    
    logger.info(f"Fold {fold_idx + 1} - Train: {len(fold_train_df)}, Val: {len(fold_val_df)}")
    logger.info(f"Train label distribution:\n{fold_train_df['label'].value_counts()}")
    
    fold_train_dataset = prepare_dataset(fold_train_df, tokenizer)
    fold_val_dataset = prepare_dataset(fold_val_df, tokenizer)
    
    model = create_model(config.PRIMARY_MODEL)
    
    if config.USE_WANDB:
        run_name = f"enhanced_fold_{fold_idx + 1}_{datetime.now().strftime('%H%M%S')}"
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
                "focal_loss": config.USE_FOCAL_LOSS,
            },
            reinit=True
        )
    
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=fold_train_dataset,
        eval_dataset=fold_val_dataset,
        fold=fold_idx,
        class_weights=class_weights
    )
    
    logger.info(f"Starting training for fold {fold_idx + 1}...")
    train_result = trainer.train()
    
    logger.info(f"Training completed. Final loss: {train_result.training_loss:.4f}")
    
    logger.info(f"Evaluating fold {fold_idx + 1}...")
    eval_results = trainer.evaluate()
    
    logger.info(f"\nFold {fold_idx + 1} Results:")
    logger.info("-" * 50)
    for key, value in sorted(eval_results.items()):
        if 'loss' not in key:
            logger.info(f"  {key:30s}: {value:.4f}")
    
    predictions = trainer.predict(fold_val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\nConfusion Matrix for Fold {fold_idx + 1}:")
    logger.info(f"\n{cm}")
    
    logger.info(f"\nClassification Report for Fold {fold_idx + 1}:")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=list(ID2LABEL.values()))}")
    
    fold_model_dir = f"{config.OUTPUT_DIR}/fold_{fold_idx + 1}_model"
    trainer.save_model(fold_model_dir)
    tokenizer.save_pretrained(fold_model_dir)  # ✅ Save tokenizer too
    logger.info(f"Model saved to {fold_model_dir}")
    
    if config.USE_WANDB:
        wandb.finish()
    
    return eval_results, trainer.model

def train_with_kfold(train_df, tokenizer):
    """K-Fold cross-validation"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting {config.N_FOLDS}-Fold Cross-Validation")
    logger.info(f"{'='*70}")
    
    # Calculate class weights
    label_counts = train_df['label'].value_counts()
    total = len(train_df)
    class_weights = torch.FloatTensor([
        total / (len(LABEL2ID) * label_counts[label]) 
        for label in sorted(LABEL2ID.keys())
    ])
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS,
        shuffle=True,
        random_state=SEED
    )
    
    fold_results = []
    best_models = []
    
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
    
    with open(f"{config.OUTPUT_DIR}/kfold_summary.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    best_fold_idx = np.argmax([r['eval_f1_macro'] for r in fold_results])
    best_score = fold_results[best_fold_idx]['eval_f1_macro']
    logger.info(f"\n🏆 Best Fold: {best_fold_idx + 1} with F1 Macro: {best_score:.4f}")
    
    return best_models[best_fold_idx], fold_results

# ============================================================================
# TEST PREDICTIONS
# ============================================================================

def predict_test_set(model, tokenizer, test_df):
    """Generate predictions"""
    logger.info("\n" + "="*70)
    logger.info("Generating predictions for test set...")
    logger.info("="*70)
    
    test_dataset = prepare_dataset(test_df, tokenizer, is_test=True)
    
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
    
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    pred_probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    
    submission_df = test_df.copy()
    submission_df['predicted_label'] = [ID2LABEL[pred] for pred in pred_labels]
    submission_df['confidence'] = np.max(pred_probs, axis=-1)
    
    for idx, label in ID2LABEL.items():
        submission_df[f'prob_{label.replace(" ", "_")}'] = pred_probs[:, idx]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_path = f"{config.OUTPUT_DIR}/test_predictions_{timestamp}.csv"
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"✅ Predictions saved to: {submission_path}")
    
    logger.info("\n📊 Prediction Statistics:")
    logger.info("-" * 50)
    logger.info(f"\nPrediction distribution:")
    logger.info(submission_df['predicted_label'].value_counts())
    logger.info(f"\nConfidence statistics:")
    logger.info(f"  Mean: {submission_df['confidence'].mean():.4f}")
    logger.info(f"  Std:  {submission_df['confidence'].std():.4f}")
    logger.info(f"  Min:  {submission_df['confidence'].min():.4f}")
    logger.info(f"  Max:  {submission_df['confidence'].max():.4f}")
    
    return submission_df

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline"""
    logger.info("="*70)
    logger.info("🚀 ENHANCED Cultural Classification Training")
    logger.info("="*70)
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"  Model: {config.PRIMARY_MODEL}")
    logger.info(f"  Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Effective Batch Size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"  LR Scheduler: {config.LR_SCHEDULER_TYPE}")
    logger.info(f"  Epochs: {config.NUM_EPOCHS}")
    logger.info(f"  Max Length: {config.MAX_LENGTH}")
    logger.info(f"  Focal Loss: {config.USE_FOCAL_LOSS}")
    logger.info(f"  FP16: {config.FP16}")
    logger.info(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.BEST_MODEL_DIR, exist_ok=True)
    
    train_df, test_df = load_data(config.TRAIN_PATH, config.TEST_PATH)
    
    logger.info(f"\n📚 Loading tokenizer: {config.PRIMARY_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.PRIMARY_MODEL, use_fast=True)
    
    best_model, results = train_with_kfold(train_df, tokenizer)
    
    logger.info(f"\n💾 Saving best model to {config.BEST_MODEL_DIR}")
    best_model.save_pretrained(config.BEST_MODEL_DIR)
    tokenizer.save_pretrained(config.BEST_MODEL_DIR)
    
    test_predictions = predict_test_set(best_model, tokenizer, test_df)
    
    logger.info("\n" + "="*70)
    logger.info("✅ Training Complete!")
    logger.info("="*70)
    logger.info(f"📁 Best model saved to: {config.BEST_MODEL_DIR}")
    logger.info(f"📊 Test predictions saved to: {config.OUTPUT_DIR}/test_predictions_*.csv")
    logger.info(f"📈 K-Fold summary saved to: {config.OUTPUT_DIR}/kfold_summary.json")
    logger.info("="*70)

if __name__ == "__main__":
    main()