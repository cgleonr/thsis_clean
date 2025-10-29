"""
Hierarchical HS Classifier - COMPLETE TRAINING SCRIPT
Trains the hierarchical model on your processed data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
import json
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalHSDataset(Dataset):
    """Dataset for hierarchical HS classification"""
    
    def __init__(
        self,
        descriptions: List[str],
        hs6_codes: List[str],
        tokenizer,
        max_length: int = 128
    ):
        self.descriptions = descriptions
        self.hs6_codes = hs6_codes
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract hierarchical labels
        self.chapters = [code[:2] for code in hs6_codes]
        self.headings = [code[:4] for code in hs6_codes]
        
        # Create label mappings
        self.chapter_to_idx = {ch: idx for idx, ch in enumerate(sorted(set(self.chapters)))}
        self.heading_to_idx = {hd: idx for idx, hd in enumerate(sorted(set(self.headings)))}
        self.hs6_to_idx = {hs: idx for idx, hs in enumerate(sorted(set(hs6_codes)))}
        
        # Reverse mappings
        self.idx_to_chapter = {v: k for k, v in self.chapter_to_idx.items()}
        self.idx_to_heading = {v: k for k, v in self.heading_to_idx.items()}
        self.idx_to_hs6 = {v: k for k, v in self.hs6_to_idx.items()}
        
        logger.info(f"Dataset size: {len(self.descriptions)}")
        logger.info(f"Unique chapters: {len(self.chapter_to_idx)}")
        logger.info(f"Unique headings: {len(self.heading_to_idx)}")
        logger.info(f"Unique HS6 codes: {len(self.hs6_to_idx)}")
    
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, idx):
        description = self.descriptions[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            description,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get labels at each level
        chapter_label = self.chapter_to_idx[self.chapters[idx]]
        heading_label = self.heading_to_idx[self.headings[idx]]
        hs6_label = self.hs6_to_idx[self.hs6_codes[idx]]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'chapter_label': torch.tensor(chapter_label, dtype=torch.long),
            'heading_label': torch.tensor(heading_label, dtype=torch.long),
            'hs6_label': torch.tensor(hs6_label, dtype=torch.long)
        }


class HierarchicalHSClassifier(nn.Module):
    """Hierarchical neural classifier for HS codes"""
    
    def __init__(
        self,
        base_model_name: str = 'distilbert-base-uncased',
        num_chapters: int = 97,
        num_headings: int = 1200,
        num_hs6: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Hierarchical classification heads
        self.chapter_head = nn.Linear(hidden_size, num_chapters)
        self.heading_head = nn.Linear(hidden_size, num_headings)
        self.hs6_head = nn.Linear(hidden_size, num_hs6)
        
        logger.info(f"Initialized HierarchicalHSClassifier with {base_model_name}")
        logger.info(f"  Chapters: {num_chapters}")
        logger.info(f"  Headings: {num_headings}")
        logger.info(f"  HS6 codes: {num_hs6}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        
        # Hierarchical predictions
        chapter_logits = self.chapter_head(pooled)
        heading_logits = self.heading_head(pooled)
        hs6_logits = self.hs6_head(pooled)
        
        return chapter_logits, heading_logits, hs6_logits


def hierarchical_loss(
    chapter_logits: torch.Tensor,
    heading_logits: torch.Tensor,
    hs6_logits: torch.Tensor,
    chapter_labels: torch.Tensor,
    heading_labels: torch.Tensor,
    hs6_labels: torch.Tensor,
    alpha: float = 0.2,
    beta: float = 0.3,
    gamma: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Hierarchical loss function"""
    
    # Individual losses
    loss_chapter = F.cross_entropy(chapter_logits, chapter_labels)
    loss_heading = F.cross_entropy(heading_logits, heading_labels)
    loss_hs6 = F.cross_entropy(hs6_logits, hs6_labels)
    
    # Weighted combination
    total_loss = alpha * loss_chapter + beta * loss_heading + gamma * loss_hs6
    
    # Return loss components for logging
    loss_dict = {
        'loss_total': total_loss.item(),
        'loss_chapter': loss_chapter.item(),
        'loss_heading': loss_heading.item(),
        'loss_hs6': loss_hs6.item()
    }
    
    return total_loss, loss_dict


def load_training_data(data_file: str = "data/processed/wco_hs_descriptions_clean.csv"):
    """Load and prepare training data"""
    
    logger.info(f"Loading training data from {data_file}")
    
    if not Path(data_file).exists():
        raise FileNotFoundError(
            f"Training data not found: {data_file}\n"
            "Please run preprocessing first: python src/data/preprocessing.py"
        )
    
    df = pd.read_csv(data_file, dtype={'hs6': str})
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Unique HS6 codes: {df['hs6'].nunique()}")
    
    # Extract descriptions and codes
    descriptions = df['description'].tolist()
    hs6_codes = df['hs6'].tolist()
    
    # Split into train/val
    train_desc, val_desc, train_hs6, val_hs6 = train_test_split(
        descriptions, hs6_codes,
        test_size=0.2,
        random_state=42,
        stratify=[code[:2] for code in hs6_codes]  # Stratify by chapter
    )
    
    logger.info(f"Train samples: {len(train_desc)}")
    logger.info(f"Val samples: {len(val_desc)}")
    
    return train_desc, train_hs6, val_desc, val_hs6


def train_model(
    train_desc: List[str],
    train_hs6: List[str],
    val_desc: List[str],
    val_hs6: List[str],
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    model_dir: str = "models/hierarchical"
):
    """Train the hierarchical model"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = HierarchicalHSDataset(train_desc, train_hs6, tokenizer)
    val_dataset = HierarchicalHSDataset(val_desc, val_hs6, tokenizer)
    
    # Save label mappings
    label_mappings = {
        'chapter_to_idx': train_dataset.chapter_to_idx,
        'heading_to_idx': train_dataset.heading_to_idx,
        'hs6_to_idx': train_dataset.hs6_to_idx,
        'idx_to_chapter': train_dataset.idx_to_chapter,
        'idx_to_heading': train_dataset.idx_to_heading,
        'idx_to_hs6': train_dataset.idx_to_hs6
    }
    
    with open(model_dir / 'label_mappings.json', 'w') as f:
        json.dump(label_mappings, f)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    logger.info("Initializing model...")
    model = HierarchicalHSClassifier(
        base_model_name='distilbert-base-uncased',
        num_chapters=len(train_dataset.chapter_to_idx),
        num_headings=len(train_dataset.heading_to_idx),
        num_hs6=len(train_dataset.hs6_to_idx)
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            chapter_labels = batch['chapter_label'].to(device)
            heading_labels = batch['heading_label'].to(device)
            hs6_labels = batch['hs6_label'].to(device)
            
            # Forward pass
            chapter_logits, heading_logits, hs6_logits = model(input_ids, attention_mask)
            
            # Compute loss
            loss, _ = hierarchical_loss(
                chapter_logits, heading_logits, hs6_logits,
                chapter_labels, heading_labels, hs6_labels,
                alpha=0.2, beta=0.3, gamma=0.5
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        model.eval()
        val_loss = 0
        chapter_correct = 0
        heading_correct = 0
        hs6_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                chapter_labels = batch['chapter_label'].to(device)
                heading_labels = batch['heading_label'].to(device)
                hs6_labels = batch['hs6_label'].to(device)
                
                # Forward pass
                chapter_logits, heading_logits, hs6_logits = model(input_ids, attention_mask)
                
                # Compute loss
                loss, _ = hierarchical_loss(
                    chapter_logits, heading_logits, hs6_logits,
                    chapter_labels, heading_labels, hs6_labels,
                    alpha=0.2, beta=0.3, gamma=0.5
                )
                
                val_loss += loss.item()
                
                # Compute accuracy
                chapter_pred = torch.argmax(chapter_logits, dim=1)
                heading_pred = torch.argmax(heading_logits, dim=1)
                hs6_pred = torch.argmax(hs6_logits, dim=1)
                
                chapter_correct += (chapter_pred == chapter_labels).sum().item()
                heading_correct += (heading_pred == heading_labels).sum().item()
                hs6_correct += (hs6_pred == hs6_labels).sum().item()
                total_samples += len(chapter_labels)
        
        val_loss /= len(val_loader)
        chapter_acc = chapter_correct / total_samples
        heading_acc = heading_correct / total_samples
        hs6_acc = hs6_correct / total_samples
        
        logger.info(f"Val loss: {val_loss:.4f}")
        logger.info(f"Val accuracy - Chapter: {chapter_acc:.4f}, Heading: {heading_acc:.4f}, HS6: {hs6_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_chapter_acc': chapter_acc,
                'val_heading_acc': heading_acc,
                'val_hs6_acc': hs6_acc
            }, model_dir / 'best_model.pt')
            logger.info("✓ Saved best model")
    
    logger.info("\n" + "="*60)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {model_dir / 'best_model.pt'}")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical HS Classifier')
    parser.add_argument('--data', type=str, default='data/processed/wco_hs_descriptions_clean.csv',
                       help='Path to training data CSV')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--model_dir', type=str, default='models/hierarchical',
                       help='Directory to save model')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("HIERARCHICAL HS CLASSIFIER TRAINING")
    logger.info("=" * 60)
    
    try:
        # Load data
        train_desc, train_hs6, val_desc, val_hs6 = load_training_data(args.data)
        
        # Train model
        train_model(
            train_desc, train_hs6,
            val_desc, val_hs6,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            model_dir=args.model_dir
        )
        
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("\nMake sure you've run preprocessing first!")
        return
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()