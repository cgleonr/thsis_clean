#!/usr/bin/env python3
"""
Improved Hierarchical HS Code Classifier Training
- Uses DistilBERT instead of BERT-tiny
- Increased epochs and better hyperparameters
- Handles merged training data
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import logging
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'base_model': 'distilbert-base-uncased',  # Better than bert-tiny
    'max_length': 128,
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 2e-5,
    'warmup_steps': 500,
    'dropout': 0.1,
    'chapter_weight': 0.2,
    'heading_weight': 0.3,
    'hs6_weight': 0.5,  # Focus on final classification
    'device': 'cuda' if torch.cuda.is_available() else 'cpu' #try to use GPU unlesss not available
}


class HierarchicalHSClassifier(nn.Module):
    """Hierarchical HS code classifier with 3 prediction heads"""
    
    def __init__(self, base_model_name, num_chapters, num_headings, num_hs6, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        
        # Three classification heads
        self.chapter_head = nn.Linear(hidden_size, num_chapters)
        self.heading_head = nn.Linear(hidden_size, num_headings)
        self.hs6_head = nn.Linear(hidden_size, num_hs6)
    
    def forward(self, input_ids, attention_mask):
        # Get embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        
        # Three predictions
        chapter_logits = self.chapter_head(pooled)
        heading_logits = self.heading_head(pooled)
        hs6_logits = self.hs6_head(pooled)
        
        return chapter_logits, heading_logits, hs6_logits


class HSDataset(Dataset):
    """Dataset for HS code classification"""
    
    def __init__(self, texts, chapters, headings, hs6s, tokenizer, max_length):
        self.texts = texts
        self.chapters = chapters
        self.headings = headings
        self.hs6s = hs6s
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text.lower().strip(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'chapter': torch.tensor(self.chapters[idx], dtype=torch.long),
            'heading': torch.tensor(self.headings[idx], dtype=torch.long),
            'hs6': torch.tensor(self.hs6s[idx], dtype=torch.long)
        }


def load_and_prepare_data(data_file):
    """Load data and create label mappings"""
    
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    
    # Ensure HS6 codes are 6 digits
    df['hs6'] = df['hs6'].astype(str).str.zfill(6)
    df['chapter'] = df['hs6'].str[:2]
    df['heading'] = df['hs6'].str[:4]
    
    logger.info(f"Total examples: {len(df):,}")
    logger.info(f"Unique chapters: {df['chapter'].nunique()}")
    logger.info(f"Unique headings: {df['heading'].nunique()}")
    logger.info(f"Unique HS6 codes: {df['hs6'].nunique()}")
    
    # Create label mappings
    chapter_to_idx = {ch: idx for idx, ch in enumerate(sorted(df['chapter'].unique()))}
    heading_to_idx = {hd: idx for idx, hd in enumerate(sorted(df['heading'].unique()))}
    hs6_to_idx = {hs: idx for idx, hs in enumerate(sorted(df['hs6'].unique()))}
    
    # Map labels
    df['chapter_idx'] = df['chapter'].map(chapter_to_idx)
    df['heading_idx'] = df['heading'].map(heading_to_idx)
    df['hs6_idx'] = df['hs6'].map(hs6_to_idx)
    
    mappings = {
        'chapter_to_idx': chapter_to_idx,
        'heading_to_idx': heading_to_idx,
        'hs6_to_idx': hs6_to_idx
    }
    
    return df, mappings


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    """Train for one epoch"""
    
    model.train()
    total_loss = 0
    chapter_criterion = nn.CrossEntropyLoss()
    heading_criterion = nn.CrossEntropyLoss()
    hs6_criterion = nn.CrossEntropyLoss()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        chapter_labels = batch['chapter'].to(device)
        heading_labels = batch['heading'].to(device)
        hs6_labels = batch['hs6'].to(device)
        
        # Forward pass
        chapter_logits, heading_logits, hs6_logits = model(input_ids, attention_mask)
        
        # Calculate losses
        chapter_loss = chapter_criterion(chapter_logits, chapter_labels)
        heading_loss = heading_criterion(heading_logits, heading_labels)
        hs6_loss = hs6_criterion(hs6_logits, hs6_labels)
        
        # Weighted combination
        loss = (
            config['chapter_weight'] * chapter_loss +
            config['heading_weight'] * heading_loss +
            config['hs6_weight'] * hs6_loss
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, config):
    """Evaluate the model"""
    
    model.eval()
    total_loss = 0
    chapter_correct = 0
    heading_correct = 0
    hs6_correct = 0
    total = 0
    
    chapter_criterion = nn.CrossEntropyLoss()
    heading_criterion = nn.CrossEntropyLoss()
    hs6_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            chapter_labels = batch['chapter'].to(device)
            heading_labels = batch['heading'].to(device)
            hs6_labels = batch['hs6'].to(device)
            
            chapter_logits, heading_logits, hs6_logits = model(input_ids, attention_mask)
            
            chapter_loss = chapter_criterion(chapter_logits, chapter_labels)
            heading_loss = heading_criterion(heading_logits, heading_labels)
            hs6_loss = hs6_criterion(hs6_logits, hs6_labels)
            
            loss = (
                config['chapter_weight'] * chapter_loss +
                config['heading_weight'] * heading_loss +
                config['hs6_weight'] * hs6_loss
            )
            
            total_loss += loss.item()
            
            # Calculate accuracies
            chapter_preds = torch.argmax(chapter_logits, dim=1)
            heading_preds = torch.argmax(heading_logits, dim=1)
            hs6_preds = torch.argmax(hs6_logits, dim=1)
            
            chapter_correct += (chapter_preds == chapter_labels).sum().item()
            heading_correct += (heading_preds == heading_labels).sum().item()
            hs6_correct += (hs6_preds == hs6_labels).sum().item()
            total += len(chapter_labels)
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'chapter_acc': chapter_correct / total,
        'heading_acc': heading_correct / total,
        'hs6_acc': hs6_correct / total
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed/wco_hs_training_data.csv')
    parser.add_argument('--output', type=str, default='models/hierarchical')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(CONFIG['device'])
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df, mappings = load_and_prepare_data(args.data)
    
    # Train/val split (stratified by chapter)
    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df['chapter'],
        random_state=42
    )
    
    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {CONFIG['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'])
    
    # Create datasets
    train_dataset = HSDataset(
        train_df['description'].values,
        train_df['chapter_idx'].values,
        train_df['heading_idx'].values,
        train_df['hs6_idx'].values,
        tokenizer,
        CONFIG['max_length']
    )
    
    val_dataset = HSDataset(
        val_df['description'].values,
        val_df['chapter_idx'].values,
        val_df['heading_idx'].values,
        val_df['hs6_idx'].values,
        tokenizer,
        CONFIG['max_length']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    logger.info(f"Initializing model: {CONFIG['base_model']}")
    model = HierarchicalHSClassifier(
        CONFIG['base_model'],
        num_chapters=len(mappings['chapter_to_idx']),
        num_headings=len(mappings['heading_to_idx']),
        num_hs6=len(mappings['hs6_to_idx']),
        dropout=CONFIG['dropout']
    )
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    total_steps = len(train_loader) * CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Training loop
    best_hs6_acc = 0
    
    # Open training log file
    log_file = output_dir / 'training_log.txt'
    with open(log_file, 'w') as f:
        f.write(f"Training started: {CONFIG['base_model']}\n")
        f.write(f"Epochs: {CONFIG['epochs']}, Batch size: {CONFIG['batch_size']}, LR: {CONFIG['learning_rate']}\n")
        f.write(f"Train examples: {len(train_df):,}, Val examples: {len(val_df):,}\n")
        f.write("="*80 + "\n\n")
    
    logger.info(f"Starting training for {CONFIG['epochs']} epochs")
    
    for epoch in range(CONFIG['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, CONFIG)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device, CONFIG)
        logger.info(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Chapter: {val_metrics['chapter_acc']:.3f} | "
            f"Heading: {val_metrics['heading_acc']:.3f} | "
            f"HS6: {val_metrics['hs6_acc']:.3f}"
        )
        
        # Write to log file
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{CONFIG['epochs']}\n")
            f.write(f"  Train Loss: {train_loss:.4f}\n")
            f.write(f"  Val Loss: {val_metrics['loss']:.4f}\n")
            f.write(f"  Chapter Acc: {val_metrics['chapter_acc']:.4f}\n")
            f.write(f"  Heading Acc: {val_metrics['heading_acc']:.4f}\n")
            f.write(f"  HS6 Acc: {val_metrics['hs6_acc']:.4f}\n")
        
        # Save best model
        if val_metrics['hs6_acc'] > best_hs6_acc:
            best_hs6_acc = val_metrics['hs6_acc']
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'model_config': CONFIG
            }
            
            torch.save(checkpoint, output_dir / 'best_model.pt')
            logger.info(f"✓ Saved best model (HS6 acc: {best_hs6_acc:.3f})")
            
            # Log best model save
            with open(log_file, 'a') as f:
                f.write(f"  >>> BEST MODEL SAVED (HS6: {best_hs6_acc:.4f})\n")
        
        with open(log_file, 'a') as f:
            f.write("\n")
    
    # Save mappings
    with open(output_dir / 'label_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)
    
    # Final log entry
    with open(log_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("TRAINING COMPLETE\n")
        f.write(f"Best HS6 Accuracy: {best_hs6_acc:.4f}\n")
        f.write(f"Model saved to: {output_dir}\n")
    
    logger.info(f"\n✓ Training complete!")
    logger.info(f"✓ Best HS6 accuracy: {best_hs6_acc:.3f}")
    logger.info(f"✓ Model saved to: {output_dir}")
    logger.info(f"✓ Training log: {log_file}")


if __name__ == '__main__':
    main()