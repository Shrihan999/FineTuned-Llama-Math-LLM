from datasets import load_dataset, Dataset
from typing import Dict, Tuple, Optional
import torch
from transformers import PreTrainedTokenizer
import logging
import json
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

class MetaMathDataLoader:
    def __init__(self, config: Dict):
        """
        Initialize the MetaMathQA data loader
        Args:
            config: Configuration dictionary containing data settings
        """
        self.config = config
        self.processed_dir = Path(config['paths']['processed_data'])
        self.raw_dir = Path(config['paths']['raw_data'])
        self.max_samples = config['data'].get('max_samples')
        
    def load_dataset(self) -> Dataset:
        """Load MetaMathQA dataset with optional size limiting"""
        dataset = load_dataset("meta-math/MetaMathQA")
        
        if self.max_samples and 'train' in dataset:
            # Randomly sample if max_samples is set
            train_size = len(dataset['train'])
            indices = np.random.choice(
                train_size, 
                size=min(self.max_samples, train_size), 
                replace=False
            )
            dataset['train'] = dataset['train'].select(indices)
            
        return dataset
    
    def prepare_data(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        Prepare dataset for training and validation
        Args:
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
        Returns:
            Tuple of train and validation datasets
        """
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        logging.info("Loading dataset...")
        dataset = self.load_dataset()
        
        def tokenize_function(examples):
            # Format inputs for instruction fine-tuning
            prompts = [
                f"You are a helpful math tutor. Solve this problem step by step:\n\nQuestion: {q}\n\nSolution:"
                for q in examples["query"]
            ]
            
            # Tokenize inputs
            model_inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )
            
            # Tokenize responses
            labels = tokenizer(
                examples["response"],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )
            
            # Create the final dictionary
            return {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "labels": labels["input_ids"],
                "type": examples["type"]
            }
        
        logging.info("Processing training dataset...")
        processed_dataset = dataset["train"].map(
            tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=dataset["train"].column_names,
            desc="Processing dataset"
        )
        
        # Set format to PyTorch tensors
        processed_dataset.set_format(type="torch")
        
        # Split into train and validation
        train_size = int(len(processed_dataset) * self.config['data']['train_split'])
        val_size = len(processed_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            processed_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logging.info(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def save_processed_data(self, split_name: str, data: Dict):
        """Save processed data to disk"""
        output_file = self.processed_dir / f"{split_name}_processed.json"
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Saved processed data to {output_file}")
    
    def load_processed_data(self, split_name: str) -> Optional[Dict]:
        """Load processed data from disk"""
        input_file = self.processed_dir / f"{split_name}_processed.json"
        if input_file.exists():
            with input_file.open('r', encoding='utf-8') as f:
                return json.load(f)
        return None