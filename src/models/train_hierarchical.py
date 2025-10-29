"""
Standalone Hierarchical Model Training Script
All classes included - no imports needed from project
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATASET CLASS
# ============================================================================

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


# ============================================================================
# MODEL CLASS
# ============================================================================

class HierarchicalHSClassifier(nn.Module):
    """
    Hierarchical neural classifier for HS codes
    
    Architecture:
    - Shared BERT encoder
    - Three separate classification heads (Chapter, Heading, HS6)
    - Multi-task learning with weighted losses
    """
    
    def __init__(
        self,
        base_model_name: str = 'prajjwal1/bert-tiny',
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
        """
        Forward pass
        
        Returns:
            Tuple of (chapter_logits, heading_logits, hs6_logits)
        """
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


# ============================================================================
# LOSS FUNCTION
# ============================================================================

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
    """
    Hierarchical loss function
    
    Combines losses at each level with configurable weights
    """
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


# ============================================================================
# TRAINER CLASS
# ============================================================================

class HierarchicalTrainer:
    """Trainer for hierarchical HS classifier"""
    
    def __init__(
        self,
        model: HierarchicalHSClassifier,
        tokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        model_dir: str = "models/hierarchical"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using device: {device}")
    
    def train(
        self,
        train_dataset: HierarchicalHSDataset,
        val_dataset: Optional[HierarchicalHSDataset] = None,
        batch_size: int = 16,
        num_epochs: int = 10,
        learning_rate: float = 2e-5,
        loss_weights: Tuple[float, float, float] = (0.2, 0.3, 0.5)
    ):
        """Train the model"""
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self._train_epoch(
                train_loader,
                optimizer,
                loss_weights
            )
            
            logger.info(f"  Train loss: {train_loss:.4f}")
            
            # Validate
            if val_loader is not None:
                val_loss, val_metrics = self._validate(
                    val_loader,
                    loss_weights
                )
                
                logger.info(f"  Val loss: {val_loss:.4f}")
                logger.info(f"  Val accuracy - Chapter: {val_metrics['chapter_acc']:.4f}, "
                          f"Heading: {val_metrics['heading_acc']:.4f}, "
                          f"HS6: {val_metrics['hs6_acc']:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    logger.info("  ✓ Saved best model")
        
        logger.info("\nTraining complete!")
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer,
        loss_weights: Tuple[float, float, float]
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            chapter_labels = batch['chapter_label'].to(self.device)
            heading_labels = batch['heading_label'].to(self.device)
            hs6_labels = batch['hs6_label'].to(self.device)
            
            # Forward pass
            chapter_logits, heading_logits, hs6_logits = self.model(
                input_ids,
                attention_mask
            )
            
            # Compute loss
            loss, _ = hierarchical_loss(
                chapter_logits, heading_logits, hs6_logits,
                chapter_labels, heading_labels, hs6_labels,
                *loss_weights
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate(
        self,
        val_loader: DataLoader,
        loss_weights: Tuple[float, float, float]
    ) -> Tuple[float, Dict[str, float]]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        chapter_correct = 0
        heading_correct = 0
        hs6_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                chapter_labels = batch['chapter_label'].to(self.device)
                heading_labels = batch['heading_label'].to(self.device)
                hs6_labels = batch['hs6_label'].to(self.device)
                
                # Forward pass
                chapter_logits, heading_logits, hs6_logits = self.model(
                    input_ids,
                    attention_mask
                )
                
                # Compute loss
                loss, _ = hierarchical_loss(
                    chapter_logits, heading_logits, hs6_logits,
                    chapter_labels, heading_labels, hs6_labels,
                    *loss_weights
                )
                
                total_loss += loss.item()
                
                # Compute accuracy
                chapter_pred = torch.argmax(chapter_logits, dim=1)
                heading_pred = torch.argmax(heading_logits, dim=1)
                hs6_pred = torch.argmax(hs6_logits, dim=1)
                
                chapter_correct += (chapter_pred == chapter_labels).sum().item()
                heading_correct += (heading_pred == heading_labels).sum().item()
                hs6_correct += (hs6_pred == hs6_labels).sum().item()
                total_samples += len(chapter_labels)
        
        metrics = {
            'chapter_acc': chapter_correct / total_samples,
            'heading_acc': heading_correct / total_samples,
            'hs6_acc': hs6_correct / total_samples
        }
        
        return total_loss / len(val_loader), metrics
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'base_model_name': 'prajjwal1/bert-tiny',
            }
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_hierarchical_model():
    """Complete training pipeline"""
    
    logger.info("=" * 60)
    logger.info("HIERARCHICAL MODEL TRAINING (STANDALONE)")
    logger.info("=" * 60)
    
    # Configuration
    config = {
        'base_model': 'prajjwal1/bert-tiny',  # Fast, small. Use 'distilbert-base-uncased' for better quality
        'model_dir': 'models/hierarchical',
        'batch_size': 16,  # Reduce to 8 if OOM
        'num_epochs': 10,
        'learning_rate': 2e-5,
        'train_split': 0.8,
        'val_split': 0.15,
        'test_split': 0.05,
        'max_length': 128,
        'loss_weights': (0.2, 0.3, 0.5),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info("\n1. CONFIGURATION")
    for key, value in config.items():
        logger.info(f"   {key}: {value}")
    
    # Check device
    if config['device'] == 'cuda':
        logger.info(f"\n   🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("\n   ⚠️  Using CPU (training will be slower)")
    
    # Load Data
    logger.info("\n2. LOADING DATA")
    
    # Try multiple possible paths
    possible_paths = [
        Path('data/processed/wco_hs_descriptions_fixed.csv'),
        Path('data/processed/wco_hs_descriptions_clean.csv'),
        Path('../data/processed/wco_hs_descriptions_fixed.csv'),
        Path('../data/processed/wco_hs_descriptions_clean.csv'),
    ]
    
    data_file = None
    for path in possible_paths:
        if path.exists():
            data_file = path
            logger.info(f"   Found data: {data_file}")
            break
    
    if data_file is None:
        logger.error("   ❌ Could not find data file!")
        logger.error("   Please ensure one of these files exists:")
        for p in possible_paths:
            logger.error(f"      - {p}")
        return
    
    # Load and validate
    df = pd.read_csv(data_file, dtype={'hs6': str})
    logger.info(f"   Loaded {len(df)} records")
    
    df = df.dropna(subset=['hs6', 'description'])
    df = df[df['hs6'].str.len() == 6]
    logger.info(f"   After validation: {len(df)} records")
    
    descriptions = df['description'].tolist()
    hs6_codes = df['hs6'].tolist()
    
    # Initialize Tokenizer
    logger.info("\n3. INITIALIZING TOKENIZER")
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    logger.info("   ✅ Tokenizer loaded")
    
    # Create Dataset
    logger.info("\n4. CREATING DATASET")
    dataset = HierarchicalHSDataset(
        descriptions=descriptions,
        hs6_codes=hs6_codes,
        tokenizer=tokenizer,
        max_length=config['max_length']
    )
    
    # Split Data
    logger.info("\n5. SPLITTING DATA")
    train_size = int(config['train_split'] * len(dataset))
    val_size = int(config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    logger.info(f"   Train: {train_size} ({config['train_split']*100:.0f}%)")
    logger.info(f"   Val:   {val_size} ({config['val_split']*100:.0f}%)")
    logger.info(f"   Test:  {test_size} ({config['test_split']*100:.0f}%)")
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Initialize Model
    logger.info("\n6. INITIALIZING MODEL")
    model = HierarchicalHSClassifier(
        base_model_name=config['base_model'],
        num_chapters=len(dataset.chapter_to_idx),
        num_headings=len(dataset.heading_to_idx),
        num_hs6=len(dataset.hs6_to_idx),
        dropout=0.1
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Total parameters: {num_params:,}")
    
    # Initialize Trainer
    logger.info("\n7. INITIALIZING TRAINER")
    trainer = HierarchicalTrainer(
        model=model,
        tokenizer=tokenizer,
        device=config['device'],
        model_dir=config['model_dir']
    )
    
    # Train
    logger.info("\n8. TRAINING MODEL")
    logger.info("   This may take 30-120 minutes...\n")
    
    try:
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            loss_weights=config['loss_weights']
        )
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("\n   ❌ GPU Out of Memory!")
            logger.error("   Try reducing batch_size to 8")
        raise
    
    # Save Mappings
    logger.info("\n9. SAVING LABEL MAPPINGS")
    mappings = {
        'chapter_to_idx': dataset.chapter_to_idx,
        'heading_to_idx': dataset.heading_to_idx,
        'hs6_to_idx': dataset.hs6_to_idx,
        'idx_to_chapter': dataset.idx_to_chapter,
        'idx_to_heading': dataset.idx_to_heading,
        'idx_to_hs6': dataset.idx_to_hs6
    }
    
    model_dir = Path(config['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    mappings_file = model_dir / 'label_mappings.json'
    with open(mappings_file, 'w') as f:
        json.dump(mappings, f, indent=2)
    
    logger.info(f"   Saved to: {mappings_file}")
    
    # Test
    logger.info("\n10. TESTING MODEL")
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    model.eval()
    correct_hs6 = 0
    correct_heading = 0
    correct_chapter = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(config['device'])
            attention_mask = batch['attention_mask'].to(config['device'])
            chapter_labels = batch['chapter_label'].to(config['device'])
            heading_labels = batch['heading_label'].to(config['device'])
            hs6_labels = batch['hs6_label'].to(config['device'])
            
            chapter_logits, heading_logits, hs6_logits = model(input_ids, attention_mask)
            
            chapter_pred = torch.argmax(chapter_logits, dim=1)
            heading_pred = torch.argmax(heading_logits, dim=1)
            hs6_pred = torch.argmax(hs6_logits, dim=1)
            
            correct_chapter += (chapter_pred == chapter_labels).sum().item()
            correct_heading += (heading_pred == heading_labels).sum().item()
            correct_hs6 += (hs6_pred == hs6_labels).sum().item()
            total += len(chapter_labels)
    
    logger.info(f"\n   Test Set Results:")
    logger.info(f"   Chapter Accuracy:  {correct_chapter/total*100:.2f}%")
    logger.info(f"   Heading Accuracy:  {correct_heading/total*100:.2f}%")
    logger.info(f"   HS6 Accuracy:      {correct_hs6/total*100:.2f}%")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info("\nModel files saved:")
    logger.info(f"   ✅ {config['model_dir']}/best_model.pt")
    logger.info(f"   ✅ {config['model_dir']}/label_mappings.json")
    logger.info("\nNext: Test in web app with 'Hierarchical' model selected")


if __name__ == "__main__":
    train_hierarchical_model()